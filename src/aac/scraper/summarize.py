"""Summarisation that scales with page length, plus the conciseness guardrail.

Fix for PROBLEM 4 (the bottleneck): instead of one prompt holding the whole page, the clean text is
cut into chunks that fit comfortably in the context window. Each chunk is condensed to short notes
(map, run concurrently with a bounded pool), and the notes are combined (reduce; repeated as a tree
if the notes themselves are still too long). Every call stays a fixed, small size, so a page twice as
long costs twice the calls instead of overflowing.

Fix for PROBLEM 5 (the guardrail), three layers:
  1. the prompt states the budget (at most N bullets, M words) and the model's output tokens are capped;
  2. `check_concise` validates the result: word count, bullet count, no preamble, not empty;
  3. on a violation the model gets ONE rewrite with the exact violations; if it still fails,
     `enforce_concise` cuts it deterministically (drop preamble, keep the first N bullets, cut at a
     sentence boundary). The summary that leaves this module always meets the budget.
"""

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import LLMError, call, text_of
from ..obs import Tracer
from .fetch import Page

_PREAMBLE = re.compile(
    r"^\s*(?:here(?:'s| is| are)\b[^\n:]*:?|sure[,!.][^\n]*|certainly[,!.][^\n]*|(?:in )?summary:?|tl;?dr:?)\s*\n?",
    re.IGNORECASE,
)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

MAP_PROMPT = """You are condensing one part ({i} of {n}) of a web page titled "{title}".
Write at most 5 short bullet points with the key facts, names and numbers from THIS part only.
Skip navigation text, adverts and anything that is not the page's actual content. No introduction."""

REDUCE_PROMPT = """Below are notes taken from consecutive parts of a web page titled "{title}".
Merge them into at most {bullets} bullet points and at most {words} words in total.
Keep the most important facts and any key numbers. Do not add anything that is not in the notes.
Start directly with the first bullet ("- "). No title, no introduction, no closing remark."""

DIRECT_PROMPT = """Summarise the web page titled "{title}" below.
Write at most {bullets} bullet points and at most {words} words in total, with the most important facts first.
Do not add anything that is not in the page. Start directly with the first bullet ("- "). No introduction."""

FIX_PROMPT = """This summary breaks the length rules: {violations}.
Rewrite it as at most {bullets} bullet points and at most {words} words in total. Keep the most important points.
Return only the bullets.

{summary}"""


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # about 4 characters per token for English; good enough for budgeting


def chunk_text(text: str, max_tokens: int) -> list[str]:
    """Pack whole lines (blocks) into chunks under `max_tokens`; split an oversized block by sentences."""
    limit = max_tokens * 4
    pieces: list[str] = []
    for line in text.splitlines():
        if len(line) <= limit:
            pieces.append(line)
            continue
        buf = ""
        for sent in _SENTENCE_END.split(line):
            while len(sent) > limit:  # one "sentence" longer than a chunk: hard split
                pieces.append(sent[:limit])
                sent = sent[limit:]
            if len(buf) + len(sent) + 1 > limit:
                pieces.append(buf)
                buf = ""
            buf = f"{buf} {sent}".strip()
        if buf:
            pieces.append(buf)
    chunks, buf = [], ""
    for piece in pieces:
        if buf and len(buf) + len(piece) + 1 > limit:
            chunks.append(buf)
            buf = ""
        buf = f"{buf}\n{piece}" if buf else piece
    if buf.strip():
        chunks.append(buf)
    return chunks


def check_concise(summary: str, max_words: int, max_bullets: int) -> list[str]:
    violations = []
    words = len(summary.split())
    bullets = sum(1 for line in summary.splitlines() if _BULLET.match(line))
    if not summary.strip():
        violations.append("summary is empty")
    if words > max_words:
        violations.append(f"{words} words (limit {max_words})")
    if bullets > max_bullets:
        violations.append(f"{bullets} bullets (limit {max_bullets})")
    if _PREAMBLE.match(summary):
        violations.append("starts with an introduction instead of the content")
    return violations


def enforce_concise(summary: str, max_words: int, max_bullets: int) -> str:
    """Deterministic last resort: always returns text within both limits."""
    text = _PREAMBLE.sub("", summary.strip(), count=1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    bullets = [ln for ln in lines if _BULLET.match(ln)]
    lines = bullets[:max_bullets] if bullets else lines
    out, used = [], 0
    for line in lines:
        words = line.split()
        if used + len(words) <= max_words:
            out.append(line)
            used += len(words)
            continue
        room = max_words - used
        if room >= 5:
            # cut at the last sentence end inside the remaining budget, else at the word limit
            cut = " ".join(words[:room])
            ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", cut)]
            out.append(cut[: ends[-1]] if ends else cut.rstrip(",;:") + "...")
        break
    return "\n".join(out)


@dataclass
class SummaryResult:
    summary: str
    words: int
    bullets: int
    chunks_total: int
    chunks_used: int
    llm_calls: int
    raw_tokens_est: int
    sent_tokens_est: int
    guardrail: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class Summarizer:
    def __init__(
        self,
        llm: BaseChatModel,
        *,
        max_words: int = 120,
        max_bullets: int = 5,
        chunk_tokens: int = 1500,
        max_chunks: int = 24,
        workers: int = 4,
        retries: int = 2,
        tracer: Tracer | None = None,
    ):
        self.llm, self.max_words, self.max_bullets = llm, max_words, max_bullets
        self.chunk_tokens, self.max_chunks, self.workers, self.retries = chunk_tokens, max_chunks, workers, retries
        self.tracer = tracer or Tracer()
        self.calls = 0
        self._lock = threading.Lock()

    def _ask(self, system: str, user: str, step: str) -> str:
        with self._lock:  # map calls run on several threads
            self.calls += 1
        return text_of(
            call(
                self.llm,
                [SystemMessage(system), HumanMessage(user)],
                tracer=self.tracer,
                step=step,
                retries=self.retries,
            )
        )

    def _reduce(self, notes: list[str], title: str, final: bool) -> str:
        prompt = REDUCE_PROMPT.format(
            title=title, bullets=self.max_bullets if final else 8, words=self.max_words if final else 200
        )
        return self._ask(prompt, "\n\n".join(notes), "reduce")

    def summarize(self, page: Page) -> SummaryResult:
        notes: list[str] = []
        chunks = chunk_text(page.text, self.chunk_tokens)
        total = len(chunks)
        if total > self.max_chunks:
            # Very long page: sample chunks evenly instead of reading only the top, and say so.
            step = total / self.max_chunks
            chunks = [chunks[int(i * step)] for i in range(self.max_chunks)]
            notes.append(f"page has {total} chunks; summarised an even sample of {self.max_chunks}")
        sent_tokens = sum(estimate_tokens(c) for c in chunks)

        if len(chunks) == 1:
            draft = self._ask(
                DIRECT_PROMPT.format(title=page.title, bullets=self.max_bullets, words=self.max_words),
                chunks[0],
                "summarize",
            )
        else:
            draft = self._map_reduce(chunks, page.title, notes)

        summary, actions = self._guard(draft)
        return SummaryResult(
            summary=summary,
            words=len(summary.split()),
            bullets=sum(1 for ln in summary.splitlines() if _BULLET.match(ln)),
            chunks_total=total,
            chunks_used=len(chunks),
            llm_calls=self.calls,
            raw_tokens_est=estimate_tokens("x" * page.raw_chars),
            sent_tokens_est=sent_tokens,
            guardrail=actions,
            notes=notes,
        )

    def _map_reduce(self, chunks: list[str], title: str, notes: list[str]) -> str:
        n = len(chunks)

        def condense(item: tuple[int, str]) -> str | None:
            i, chunk = item
            try:
                return self._ask(MAP_PROMPT.format(i=i, n=n, title=title), chunk, "map")
            except LLMError:
                return None  # one bad chunk should not lose the whole summary

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            parts = list(pool.map(condense, enumerate(chunks, 1)))
        failed = sum(p is None for p in parts)
        if failed > n // 2:
            raise LLMError(f"{failed} of {n} chunks could not be summarised")
        if failed:
            notes.append(f"{failed} of {n} chunks failed and were skipped")
        parts = [p for p in parts if p]

        # Tree reduce: group notes so each reduce call also fits in one chunk.
        while estimate_tokens("\n\n".join(parts)) > self.chunk_tokens and len(parts) > 1:
            groups = chunk_text("\n\n".join(p.replace("\n", " ") for p in parts), self.chunk_tokens)
            if len(groups) >= len(parts):  # cannot shrink by grouping; stop and let the final reduce cope
                break
            parts = [self._reduce([g], title, final=False) for g in groups]
        return self._reduce(parts, title, final=True)

    def _guard(self, draft: str) -> tuple[str, list[str]]:
        violations = check_concise(draft, self.max_words, self.max_bullets)
        if not violations:
            return draft.strip(), []
        actions = [f"violation: {'; '.join(violations)}"]
        try:
            fixed = self._ask(
                "You shorten summaries to fit strict length limits.",
                FIX_PROMPT.format(
                    violations="; ".join(violations), bullets=self.max_bullets, words=self.max_words, summary=draft
                ),
                "guardrail_rewrite",
            )
        except LLMError:
            fixed = draft
        if not check_concise(fixed, self.max_words, self.max_bullets):
            actions.append("rewritten by the model")
            self.tracer.event("guardrail", status="rewritten", guardrails=["summary_rewrite"])
            return fixed.strip(), actions
        actions.append("truncated deterministically")
        self.tracer.event("guardrail", status="truncated", guardrails=["summary_truncate"])
        return enforce_concise(fixed, self.max_words, self.max_bullets), actions
