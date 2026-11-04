"""Deterministic escalation rules. These run before any LLM call, so they still work when the model is down.

The patterns favour recall over precision: a false escalation costs a human a minute, a missed
security breach costs far more. "No data loss, just a question" will be escalated; that trade-off
is deliberate. Paraphrases the patterns miss ("my folder is empty and not in Trash") are caught by
the classifier's `risk_signals`, and either source is enough to escalate.
"""

from __future__ import annotations

import re

RISK_PATTERNS: dict[str, list[str]] = {
    "data_loss": [
        r"\bdata loss\b",
        r"\blost (?:all |some |most )?(?:of )?(?:my |our |the )?(?:data|files?|photos?|documents?|work)\b",
        r"\b(?:files?|data|photos?|documents?|folders?) (?:are |is |were |was |have |has )?(?:been )?"
        r"(?:gone|missing|deleted|erased|wiped|disappeared|vanished|corrupted)\b",
        r"\b(?:deleted|wiped|erased) (?:all|everything)\b",
        r"\bcorrupt(?:ed|ion)\b",
    ],
    "service_outage": [
        r"\boutage\b",
        r"\b(?:is|are|went|been|was|seems?) (?:completely |totally )?down\b",
        r"\bservice (?:is )?unavailable\b",
        r"\b(?:502|503|504)\b",
        r"\b(?:nobody|no one|none of us) can (?:access|log ?in|sign ?in|open)\b",
        r"\bnot (?:loading|working) for (?:anyone|everyone|all)\b",
    ],
    "security_breach": [
        r"\b(?:security )?breach(?:ed)?\b",
        r"\bhack(?:ed|er|ing)?\b",
        r"\bcompromised\b",
        r"\bunauthori[sz]ed (?:access|login|sign[- ]?in|charges?|activity)\b",
        r"\bphishing\b",
        r"\b(?:data|password|credentials?) (?:was |were |has been )?leak(?:ed)?\b",
        r"\bsomeone (?:else )?(?:signed|logged) in\b",
    ],
}

_COMPILED = {signal: [re.compile(p, re.IGNORECASE) for p in pats] for signal, pats in RISK_PATTERNS.items()}


def detect_risks(text: str) -> dict[str, str]:
    """Map each risk signal found in `text` to the phrase that triggered it."""
    found: dict[str, str] = {}
    for signal, patterns in _COMPILED.items():
        for p in patterns:
            m = p.search(text)
            if m:
                found[signal] = m.group(0)
                break
    return found


# Which human queue owns each escalation reason, in priority order (first match wins).
ESCALATION_QUEUES: list[tuple[str, str]] = [
    ("security_breach", "security-incident"),
    ("data_loss", "tier2-technical"),
    ("service_outage", "tier2-technical"),
    ("repeat_contact", "customer-care-lead"),
    ("classifier_unavailable", "human-triage"),
]


def queue_for(reasons: list[str]) -> str:
    for reason, queue in ESCALATION_QUEUES:
        if reason in reasons:
            return queue
    return "human-triage"
