"""Knowledge base: load Markdown/text/PDF files, split them by section, search with BM25.

BM25 instead of embeddings is a deliberate trade-off for this size of KB: no model download,
deterministic results that tests can pin, and exact matching on terms like "annual" or "503".
`grounded-rag` (my other repo) shows the hybrid dense + BM25 + reranker version; swapping it in
only changes `KnowledgeBase.search`.
"""

from __future__ import annotations

import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from .support.models import Passage

_HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from has have how i if in is it its my of on or our so that the "
    "their them there this to was we what when where which who why will with you your me us".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def split_markdown(text: str, source: str) -> list[Passage]:
    """One passage per heading section; the document title is kept out of the body."""
    matches = list(_HEADING.finditer(text))
    passages = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip()
        if body:
            passages.append(
                Passage(id=f"{source}#{len(passages) + 1}", source=source, section=m.group(2).strip(), text=body)
            )
    if not passages and text.strip():
        passages.append(Passage(id=f"{source}#1", source=source, section=source, text=text.strip()))
    return passages


def _split_pdf(path: Path) -> list[Passage]:
    from pypdf import PdfReader

    passages = []
    for page_no, page in enumerate(PdfReader(str(path)).pages, 1):
        for j, para in enumerate(p.strip() for p in re.split(r"\n\s*\n", page.extract_text() or "")):
            if len(para) > 40:
                passages.append(
                    Passage(id=f"{path.stem}#p{page_no}.{j}", source=path.stem, section=f"page {page_no}", text=para)
                )
    return passages


def load_passages(kb_dir: Path) -> list[Passage]:
    passages: list[Passage] = []
    for path in sorted(kb_dir.iterdir()):
        if path.suffix.lower() in (".md", ".txt"):
            passages += split_markdown(path.read_text(encoding="utf-8"), path.stem)
        elif path.suffix.lower() == ".pdf":
            passages += _split_pdf(path)
    if not passages:
        raise FileNotFoundError(f"no .md/.txt/.pdf knowledge-base files in {kb_dir}")
    return passages


class KnowledgeBase:
    def __init__(self, passages: list[Passage]):
        self.passages = passages
        self._bm25 = BM25Okapi([tokenize(f"{p.section} {p.text}") for p in passages])

    @classmethod
    def from_dir(cls, kb_dir: Path) -> KnowledgeBase:
        return cls(load_passages(kb_dir))

    def search(self, query: str, k: int = 4) -> list[Passage]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(scores, self.passages, strict=True), key=lambda x: -x[0])
        return [p.model_copy(update={"score": round(float(s), 3)}) for s, p in ranked[:k] if s > 0]

    def by_source(self, needle: str) -> list[Passage]:
        return [p for p in self.passages if needle in p.source.lower()]
