"""Grounding checks on a draft reply. Pure functions, no LLM: they cannot be talked out of a decision.

What a draft must satisfy:
  1. every [n] marker points at a passage that was actually given to the model;
  2. every number in the draft (days, %, amounts...) appears in a passage that sentence cites,
     unless the customer wrote that number themselves (an invoice id, an amount they quote);
  3. any sentence that states refund *terms* (eligibility, windows, amounts, timing) cites the
     refund policy;
  4. the draft never commits to a refund. Only the Billing team approves refunds, so
     "we will refund you" / "you are eligible for a full refund" / "guaranteed" are blocked
     unless the sentence is conditional ("if approved, ...") or negated ("we cannot guarantee").

Known limit: these checks see numbers, citations and wording, not meaning. "Once approved, credits
are applied to your next invoice" cites a real section and contains no number, yet mixes up service
credits and refunds (a live qwen3.5:9b run produced exactly that). So the graph adds a second
layer for refund sentences only: `refund_sentences` picks them out and an LLM judge compares each
one with the refund-policy text (graph.py, `judge` step). The judge fails closed.
"""

from __future__ import annotations

import re

from .models import Passage

_CITE = re.compile(r"\[(\d+)\]")
_CITE_AFTER_STOP = re.compile(r"([.!?])\s*((?:\[\d+\])+)")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_NUMBER_UNIT = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%|percent|business days?|days?|months?|hours?|years?|gb|tb)?(?![\w])", re.IGNORECASE
)
_REFUND = re.compile(
    r"\b(refund\w*|money back|reimburs\w*|chargeback|service credits?|credits|credited|prorat\w*)\b", re.IGNORECASE
)
_REFUND_TERMS = re.compile(
    r"\b(eligib\w*|within|only|not refundable|non-refundable|in full|full refund|partial|prorat\w*|"
    r"business days|approv\w*|original payment method|policy)\b",
    re.IGNORECASE,
)
_CONDITIONAL = re.compile(r"\b(if|once|after|may|can request|would)\b", re.IGNORECASE)
_PROMISES = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bwe(?:'ll| will| are going to| have| can)\s+(?:\w+\s+){0,3}?(?:refund|reimburse|credit)\w*",
        r"\b(?:i|we)(?:'ve| have)\s+(?:\w+\s+){0,2}?(?:refunded|issued|processed|approved)\b",
        r"\bguarantee\w*\b",
        r"\brefund (?:has been|was|is being|will be) (?:processed|issued|approved|sent)\b",
        r"\byou(?:'ll| will) (?:get|receive) (?:a |your )?(?:full )?refund\b",
        r"\byou(?:'re| are| will be)\s+(?:\w+\s+)?eligible\b",  # eligibility is Billing's decision
        r"\byou qualify for (?:a |the )?(?:full |partial )?refund\b",
    )
]
_NEGATOR = r"(?:cannot|can't|can not|do not|don't|won't|will not|unable to|not able to|never|not)"
_NEGATED_INSIDE = re.compile(rf"\b{_NEGATOR}\b", re.IGNORECASE)
_NEGATED_BEFORE = re.compile(rf"\b{_NEGATOR}\s+(?:\w+\s+)?$", re.IGNORECASE)


def is_promise(sentence: str) -> bool:
    """A commitment to a refund. "We cannot guarantee" or "if approved, ..." is not one,
    but "Don't worry, we will refund you" still is: the negation must govern the promise itself."""
    if _CONDITIONAL.search(sentence):
        return False
    for p in _PROMISES:
        for m in p.finditer(sentence):
            if not (_NEGATED_INSIDE.search(m.group(0)) or _NEGATED_BEFORE.search(sentence[: m.start()])):
                return True
    return False


def refund_sentences(draft: str) -> list[str]:
    """Sentences that talk about refunds or credits: the ones the LLM judge re-checks."""
    return [strip_citations(s) for s in sentences(draft) if _REFUND.search(s)]


def sentences(text: str) -> list[str]:
    """Split into sentences, keeping a trailing "[n]" with the sentence it belongs to."""
    text = _CITE_AFTER_STOP.sub(lambda m: f"{m.group(2)}{m.group(1)}", text.strip())
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


def strip_citations(text: str) -> str:
    return re.sub(r"\s*(?:\[\d+\])+", "", text)


def _unit_root(unit: str) -> str:
    unit = unit.lower()
    return "%" if unit in ("%", "percent") else unit.rstrip("s")


def _supported(number: str, unit: str | None, haystack: str) -> bool:
    if unit:
        root = _unit_root(unit)
        if root == "%":
            return re.search(rf"(?<![\d.]){re.escape(number)}\s*(?:%|percent)", haystack, re.I) is not None
        return re.search(rf"(?<![\d.]){re.escape(number)}\s*{re.escape(root)}", haystack, re.I) is not None
    return re.search(rf"(?<![\d.]){re.escape(number)}(?![\d])", haystack) is not None


def check_draft(draft: str, passages: list[Passage], customer_text: str = "") -> tuple[list[str], list[int]]:
    """Return (problems, cited passage numbers). No problems means the draft may go out."""
    problems: list[str] = []
    cited_all: list[int] = []
    if not draft.strip():
        return ["the draft is empty"], []

    for sent in sentences(draft):
        cited = [int(n) for n in _CITE.findall(sent)]
        bad = [n for n in cited if not 1 <= n <= len(passages)]
        if bad:
            problems.append(f"cites passage(s) {bad} that do not exist: {sent!r}")
        cited = [n for n in cited if 1 <= n <= len(passages)]
        cited_all += [n for n in cited if n not in cited_all]
        cited_text = "\n".join(passages[n - 1].text for n in cited)
        plain = strip_citations(sent)

        for m in _NUMBER_UNIT.finditer(plain):
            number, unit = m.group(1), m.group(2)
            if _supported(number, unit, customer_text):
                continue
            if not cited:
                problems.append(f"states '{m.group(0).strip()}' without citing a source: {plain!r}")
            elif not _supported(number, unit, cited_text):
                problems.append(f"'{m.group(0).strip()}' is not in the cited passage(s) {cited}: {plain!r}")

        if _REFUND.search(plain):
            states_terms = _REFUND_TERMS.search(plain) or _NUMBER_UNIT.search(plain)
            if states_terms and not any(passages[n - 1].is_refund_policy for n in cited):
                problems.append(f"states refund terms without citing the refund policy: {plain!r}")
        if is_promise(plain):
            problems.append(f"commits to a refund or guarantee; only the Billing team can approve refunds: {plain!r}")
    return problems, cited_all
