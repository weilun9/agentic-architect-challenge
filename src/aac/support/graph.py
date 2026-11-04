"""The triage pipeline as a LangGraph state machine.

    screen ─► classify ─┬─(any flag)──────────────────────────────► escalate ─► END
     rules,   LLM, JSON  │
     history  validated  └─► retrieve ─► draft ─► verify ─┬─(ok, mentions refunds)─► judge ─┐
                                                 ▲        ├─(ok)──────────────────────────────┴─► finalize ─► END
                                                 │        ├─(problems, budget left)─► revise ─► verify
                                                 └────────┘
                                                          └─(still failing)─────────────────► fallback ─► END
                         no KB match ─────────────────────────────────────────────────────────► fallback

Ordering is the point of the design:
  * `screen` (no LLM) records the contact and runs the keyword rules + repeat-contact count first;
  * `classify` adds multi-label categories and the model's own risk signals (for paraphrases);
  * the escalation gate sits BEFORE retrieval and drafting, so a flagged email never gets an
    auto-drafted reply, only a hand-off note for the human queue;
  * if the classifier fails (model down, invalid JSON after a repair attempt), the email is
    escalated: the pipeline fails closed, never open;
  * refund hallucination has two checks: deterministic `verify` (numbers, citations, promises), then
    an LLM `judge` that reads each refund sentence next to the policy text. A failed or unavailable
    judge counts as a problem, so the worst outcome is the fixed template that quotes the policy.
"""

from __future__ import annotations

import time
from typing import TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from ..config import Settings
from ..kb import KnowledgeBase
from ..llm import LLMError, call, call_json, text_of
from ..obs import Tracer
from .history import ContactHistory
from .models import CATEGORIES, RISK_SIGNALS, Classification, Email, Flag, Passage, TriageResult
from .rules import detect_risks, queue_for
from .verify import check_draft, refund_sentences, strip_citations

CLASSIFY_PROMPT = f"""You triage customer-support emails for SkyVault, a cloud file-storage service.
Return ONLY a JSON object with these keys:
  "categories": one or more of {list(CATEGORIES)} (most relevant first; an email can have several),
  "urgency": "low" | "normal" | "high",
  "risk_signals": any of {list(RISK_SIGNALS)} that the email describes, even in other words
     (e.g. files that vanished = data_loss, nobody can reach the service = service_outage,
      someone else in the account = security_breach); [] if none,
  "summary": one sentence describing what the customer needs.
Refund requests are both "billing" and "refund"."""

DRAFT_PROMPT = """You write replies to SkyVault customers using ONLY the numbered passages provided.
Rules:
- Every sentence that states a policy, number, time limit or step must end with its citation, like [1] or [2][3].
- Never invent numbers, time limits, prices, features or conditions. If the passages do not answer something, say a specialist will follow up.
- Answer only what the customer asked. Ignore passages that are not about their question.
- Never promise, guarantee or confirm a refund or credit: only the Billing team approves refunds. You may explain the policy and say the request has been passed to Billing.
- Plain, friendly tone. At most 120 words. Start with "Hi {name}," and end with "Best regards, SkyVault Support". No subject line."""

JUDGE_PROMPT = """You check sentences from a customer-support reply against a company's refund policy.
For each numbered sentence decide whether the policy fully supports it. A sentence is NOT supported if it
- adds a condition, time limit, amount, payment method or step that the policy does not state;
- decides the customer's eligibility or promises an outcome (only the Billing team decides);
- mixes up different rules, e.g. service credits (applied to invoices) with refunds (paid to the card).
Neutral sentences ("your request has been passed to Billing") are supported.
Return ONLY JSON: {"verdicts": [{"n": 1, "supported": true, "reason": "short reason"}, ...]} with one entry per sentence."""

REVISE_PROMPT = """Your draft failed these automatic checks:
{problems}

Rewrite the reply so that it passes. Remove any claim you cannot cite from the passages. Keep the same rules."""

TEAM_BY_CATEGORY = {"refund": "Billing", "billing": "Billing", "technical": "Technical", "account": "Account"}


class TriageState(TypedDict, total=False):
    email: Email
    contact_count: int
    flags: list[Flag]
    classification: Classification | None
    passages: list[Passage]
    draft: str
    problems: list[str]
    cited: list[int]
    revisions: int
    judged: bool
    result: TriageResult
    path: list[str]


class Verdict(BaseModel):
    n: int
    supported: bool
    reason: str = ""


class JudgeResult(BaseModel):
    verdicts: list[Verdict]


def format_passages(passages: list[Passage]) -> str:
    return "\n\n".join(f"[{i}] ({p.source} > {p.section})\n{p.text}" for i, p in enumerate(passages, 1))


class SupportTriage:
    def __init__(
        self,
        settings: Settings,
        llm: BaseChatModel,
        kb: KnowledgeBase,
        history: ContactHistory,
        tracer: Tracer | None = None,
    ):
        self.s, self.llm, self.kb, self.history = settings, llm, kb, history
        self.tracer = tracer or Tracer()
        self.graph = self._build()

    # ---- nodes -------------------------------------------------------------------------------

    def screen(self, state: TriageState) -> TriageState:
        email = state["email"]
        self.history.record(email.id, email.address, email.received_at)
        count = self.history.count(email.address, email.received_at, self.s.repeat_contact_days)
        flags = [Flag(reason=sig, source="rule", evidence=phrase) for sig, phrase in detect_risks(email.text).items()]
        if count > self.s.repeat_contact_limit:
            flags.append(
                Flag(
                    reason="repeat_contact",
                    source="rule",
                    evidence=f"{count} contacts in {self.s.repeat_contact_days} days",
                )
            )
        self._trace(state, "screen", flags=[f.reason for f in flags], contact_count=count)
        return {"contact_count": count, "flags": flags, "path": [*state.get("path", []), "screen"]}

    def classify(self, state: TriageState) -> TriageState:
        email = state["email"]
        messages = [SystemMessage(CLASSIFY_PROMPT), HumanMessage(f"Subject: {email.subject}\n\n{email.body}")]
        flags = list(state.get("flags", []))
        try:
            c = call_json(
                self.llm, messages, Classification, tracer=self._t(state), step="classify", retries=self.s.llm_retries
            )
        except LLMError as exc:
            flags.append(Flag(reason="classifier_unavailable", source="system", evidence=str(exc)[:200]))
            return {"classification": None, "flags": flags, "path": [*state["path"], "classify"]}
        known = {f.reason for f in flags}
        flags += [Flag(reason=sig, source="llm", evidence=c.summary) for sig in c.risk_signals if sig not in known]
        self._trace(state, "classify_result", categories=c.categories, risk_signals=c.risk_signals, urgency=c.urgency)
        return {"classification": c, "flags": flags, "path": [*state["path"], "classify"]}

    def escalate(self, state: TriageState) -> TriageState:
        email, flags, c = state["email"], state["flags"], state.get("classification")
        reasons = [f.reason for f in flags]
        note = "\n".join(
            [
                f"ESCALATED: {', '.join(reasons)}",
                *[f"- {f.reason} ({f.source}): {f.evidence}" for f in flags],
                f"Contacts in last {self.s.repeat_contact_days} days: {state.get('contact_count', 0)}",
                f"Summary: {c.summary if c else email.subject}",
                "No reply was auto-drafted. A person must respond.",
            ]
        )
        result = self._result(state, status="escalated", queue=queue_for(reasons), handoff_note=note)
        self._trace(state, "escalate", queue=result.queue, guardrails=["escalated:" + r for r in reasons])
        return {"result": result, "path": [*state["path"], "escalate"]}

    def retrieve(self, state: TriageState) -> TriageState:
        email, c = state["email"], state["classification"]
        hits = self.kb.search(f"{email.subject} {email.body}", k=4)
        refund_related = c and ("refund" in c.categories or "billing" in c.categories)
        if refund_related:
            # Pin the refund policy: the model must always see the real rules when money is involved.
            refund_hits = self.kb.search(f"refund {email.subject} {email.body}", k=10)
            pinned = [p for p in refund_hits if p.is_refund_policy][:4]
            hits = pinned + [h for h in hits if h.id not in {p.id for p in pinned}]
        hits = hits[:6]
        self._trace(state, "retrieve", passages=[h.id for h in hits], pinned_refund=bool(refund_related))
        return {"passages": hits, "revisions": 0, "path": [*state["path"], "retrieve"]}

    def draft(self, state: TriageState) -> TriageState:
        email = state["email"]
        messages = [
            SystemMessage(DRAFT_PROMPT.format(name=email.display_name)),
            HumanMessage(
                f"Passages:\n{format_passages(state['passages'])}\n\nCustomer email:\nSubject: {email.subject}\n\n{email.body}"
            ),
        ]
        if state.get("problems"):
            messages += [
                HumanMessage(f"Your previous draft:\n{state['draft']}"),
                HumanMessage(REVISE_PROMPT.format(problems="\n".join(f"- {p}" for p in state["problems"]))),
            ]
        step = "revise" if state.get("problems") else "draft"
        try:
            text = text_of(call(self.llm, messages, tracer=self._t(state), step=step, retries=self.s.llm_retries))
        except LLMError as exc:
            return {"draft": "", "problems": [f"drafting failed: {exc}"], "path": [*state["path"], step]}
        revisions = state.get("revisions", 0) + (1 if step == "revise" else 0)
        return {"draft": text, "revisions": revisions, "path": [*state["path"], step]}

    def verify(self, state: TriageState) -> TriageState:
        email = state["email"]
        problems, cited = check_draft(state.get("draft", ""), state["passages"], customer_text=email.text)
        self._trace(state, "verify", problems=len(problems), guardrails=["grounding_violation"] * bool(problems))
        return {"problems": problems, "cited": cited, "judged": False, "path": [*state["path"], "verify"]}

    def judge(self, state: TriageState) -> TriageState:
        """LLM check of refund sentences against the full refund policy. Fails closed."""
        claims = refund_sentences(state["draft"])
        policy = "\n\n".join(f"## {p.section}\n{p.text}" for p in self.kb.by_source("refund"))
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(claims, 1))
        messages = [SystemMessage(JUDGE_PROMPT), HumanMessage(f"Policy:\n{policy}\n\nSentences:\n{numbered}")]
        try:
            result = call_json(
                self.llm, messages, JudgeResult, tracer=self._t(state), step="judge", retries=self.s.llm_retries
            )
            verdicts = {v.n: v for v in result.verdicts}
            problems = [
                f"refund check: not supported by the policy ({verdicts[i].reason if i in verdicts else 'no verdict'}): {s!r}"
                for i, s in enumerate(claims, 1)
                if i not in verdicts or not verdicts[i].supported
            ]
        except LLMError as exc:
            problems = [f"refund check unavailable: {exc}"[:300]]
        self._trace(
            state,
            "judge_result",
            sentences=len(claims),
            problems=len(problems),
            guardrails=["refund_judge_reject"] * bool(problems),
        )
        return {"problems": problems, "judged": True, "path": [*state["path"], "judge"]}

    def finalize(self, state: TriageState) -> TriageState:
        passages = state["passages"]
        citations = [
            {"n": n, "source": passages[n - 1].source, "section": passages[n - 1].section} for n in state["cited"]
        ]
        result = self._result(
            state,
            status="drafted",
            queue=self._team_queue(state),
            draft=strip_citations(state["draft"]),
            citations=citations,
        )
        return {"result": result, "path": [*state["path"], "finalize"]}

    def fallback(self, state: TriageState) -> TriageState:
        """A fixed template. It quotes the refund policy verbatim when relevant, so it cannot hallucinate."""
        email, c = state["email"], state.get("classification")
        team = TEAM_BY_CATEGORY.get(c.categories[0], "Support") if c else "Support"
        lines = [
            f"Hi {email.display_name},",
            "",
            f'Thank you for contacting SkyVault Support about "{email.subject}". '
            f"Your message has been passed to our {team} team, who will review it and reply to you directly.",
        ]
        refund = [p for p in state.get("passages", []) if p.is_refund_policy]
        if c and "refund" in c.categories and refund:
            lines += [
                "",
                "For reference, this is the relevant part of our published refund policy:",
                "",
                f'"{refund[0].text}"',
            ]
        lines += ["", "Best regards,", "SkyVault Support"]
        problems = state.get("problems") or ["no knowledge-base passage matched this email"]
        result = self._result(
            state,
            status="needs_review",
            queue=self._team_queue(state),
            draft="\n".join(lines),
            problems=problems,
            citations=[{"n": 1, "source": refund[0].source, "section": refund[0].section}]
            if refund and c and "refund" in c.categories
            else [],
        )
        self._trace(state, "fallback", problems=problems[:3], guardrails=["template_fallback"])
        return {"result": result, "path": [*state["path"], "fallback"]}

    # ---- edges -------------------------------------------------------------------------------

    @staticmethod
    def after_classify(state: TriageState) -> str:
        return "escalate" if state.get("flags") else "retrieve"

    @staticmethod
    def after_retrieve(state: TriageState) -> str:
        return "draft" if state.get("passages") else "fallback"

    def after_verify(self, state: TriageState) -> str:
        """Shared by `verify` and `judge`: pass on, revise once, or give up to the template."""
        if not state.get("problems"):
            if not state.get("judged") and refund_sentences(state.get("draft", "")):
                return "judge"
            return "finalize"
        if state.get("draft") and state.get("revisions", 0) < self.s.max_revisions:
            return "draft"
        return "fallback"

    def _build(self):
        g = StateGraph(TriageState)
        for name in ("screen", "classify", "escalate", "retrieve", "draft", "verify", "judge", "finalize", "fallback"):
            g.add_node(name, getattr(self, name))
        g.set_entry_point("screen")
        g.add_edge("screen", "classify")
        g.add_conditional_edges("classify", self.after_classify, {"escalate": "escalate", "retrieve": "retrieve"})
        g.add_conditional_edges("retrieve", self.after_retrieve, {"draft": "draft", "fallback": "fallback"})
        g.add_edge("draft", "verify")
        routes = {"judge": "judge", "finalize": "finalize", "draft": "draft", "fallback": "fallback"}
        g.add_conditional_edges("verify", self.after_verify, routes)
        g.add_conditional_edges("judge", self.after_verify, routes)
        for terminal in ("escalate", "finalize", "fallback"):
            g.add_edge(terminal, END)
        return g.compile()

    # ---- helpers -----------------------------------------------------------------------------

    def _t(self, state: TriageState) -> Tracer:
        return self.tracer.child(email_id=state["email"].id)

    def _trace(self, state: TriageState, step: str, **fields) -> None:
        self._t(state).event(step, **fields)

    def _team_queue(self, state: TriageState) -> str:
        c = state.get("classification")
        return f"{c.categories[0]}-queue" if c else "human-triage"

    def _result(self, state: TriageState, **fields) -> TriageResult:
        c = state.get("classification")
        return TriageResult(
            email_id=state["email"].id,
            categories=c.categories if c else [],
            flags=state.get("flags", []),
            contact_count=state.get("contact_count", 0),
            revisions=state.get("revisions", 0),
            **fields,
        )

    # ---- entry points ------------------------------------------------------------------------

    def process(self, email: Email) -> TriageResult:
        start = time.perf_counter()
        try:
            final = self.graph.invoke({"email": email, "path": []})
            result = final["result"].model_copy(update={"path": final["path"]})
        except Exception as exc:  # a bug in one email must not stop the inbox; fail closed to a human
            self.tracer.child(email_id=email.id).event(
                "pipeline", status="error", error=f"{type(exc).__name__}: {exc}"[:300]
            )
            result = TriageResult(
                email_id=email.id,
                status="escalated",
                queue="human-triage",
                flags=[Flag(reason="pipeline_error", source="system", evidence=str(exc)[:200])],
                handoff_note="Automatic processing failed; handle manually.",
            )
        result.latency_ms = round((time.perf_counter() - start) * 1000)
        self.tracer.child(email_id=email.id).event(
            "result", status=result.status, queue=result.queue, ms=result.latency_ms, path=result.path
        )
        return result

    def process_batch(self, emails: list[Email], workers: int | None = None) -> list[TriageResult]:
        """Record every contact first (in time order) so the repeat-contact count does not depend
        on which worker thread finishes first, then process emails concurrently."""
        from concurrent.futures import ThreadPoolExecutor

        for e in sorted(emails, key=lambda e: e.received_at):
            self.history.record(e.id, e.address, e.received_at)
        with ThreadPoolExecutor(max_workers=workers or self.s.triage_workers) as pool:
            return list(pool.map(self.process, emails))
