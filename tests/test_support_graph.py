"""End-to-end paths through the triage graph with a scripted model."""

import json
from datetime import datetime, timedelta

import pytest

from aac.kb import KnowledgeBase
from aac.llm import LLMError
from aac.support.graph import SupportTriage
from aac.support.history import ContactHistory
from aac.support.models import Email

from .conftest import scripted

T0 = datetime(2026, 11, 2, 9, 0)


def email(body, subject="Help", sender="Ana <ana@example.com>", id="e1", at=T0):
    return Email(id=id, sender=sender, subject=subject, body=body, received_at=at)


def classification(categories, risks=(), summary="customer needs help"):
    return json.dumps(
        {"categories": list(categories), "urgency": "normal", "risk_signals": list(risks), "summary": summary}
    )


@pytest.fixture
def kb(settings):
    return KnowledgeBase.from_dir(settings.kb_dir)


def make(settings, kb, llm, history=None):
    return SupportTriage(settings, llm, kb, history or ContactHistory())


def test_rule_flag_escalates_before_any_draft(settings, kb):
    llm = scripted(classification(["technical"]))  # only the classifier may be called
    r = make(settings, kb, llm).process(email("SkyVault is down and every page shows 503", subject="Outage"))
    assert r.status == "escalated" and r.queue == "tier2-technical"
    assert r.draft == "" and "No reply was auto-drafted" in r.handoff_note
    assert r.path == ["screen", "classify", "escalate"]
    assert len(llm.calls) == 1


def test_llm_catches_a_paraphrase_the_rules_miss(settings, kb):
    llm = scripted(classification(["technical"], risks=["data_loss"], summary="folder emptied"))
    r = make(settings, kb, llm).process(email("My Projects folder is completely empty and not in Trash"))
    assert r.status == "escalated"
    assert [(f.reason, f.source) for f in r.flags] == [("data_loss", "llm")]


def test_fourth_contact_in_seven_days_escalates(settings, kb):
    history = ContactHistory()
    for i in range(3):
        history.record(f"old{i}", "luis@example.com", T0 - timedelta(days=i + 1))
    llm = scripted(classification(["billing"]))
    r = make(settings, kb, llm, history).process(email("My payment failed", sender="luis@example.com"))
    assert r.status == "escalated" and r.queue == "customer-care-lead" and r.contact_count == 4


def test_third_contact_is_not_escalated(settings, kb):
    history = ContactHistory()
    for i in range(2):
        history.record(f"old{i}", "luis@example.com", T0 - timedelta(days=i + 1))
    good = "Hi Luis, payments usually fail because the card has expired [1]. Best regards, SkyVault Support"
    llm = scripted(classification(["billing"]), good)
    r = make(settings, kb, llm, history).process(email("Why did my payment fail?", sender="luis@example.com"))
    assert r.status == "drafted" and r.contact_count == 3


def test_classifier_failure_fails_closed(settings, kb):
    llm = scripted(LLMError("connection refused"))
    r = make(settings, kb, llm).process(email("How do I change my email?"))
    assert r.status == "escalated" and r.queue == "human-triage"
    assert r.flags[0].reason == "classifier_unavailable"


def test_invalid_json_gets_one_repair_round(settings, kb):
    draft = "Hi Oliver, go to Settings > Account > Email and enter the new address [1]. Best regards, SkyVault Support"
    llm = scripted("categories: account", classification(["account"]), draft)
    r = make(settings, kb, llm).process(
        email("How do I change my login email address?", subject="Change my login email")
    )
    assert r.status == "drafted"
    assert "not valid" in llm.calls[1][-1].content  # the repair prompt quotes the error


def test_refund_hallucination_is_revised_then_passes(settings, kb):
    bad = "Hi Priya, monthly plans can be refunded within 30 days [1]. Best regards, SkyVault Support"

    def good(messages):
        assert "failed these automatic checks" in messages[-1].content
        return (
            "Hi Priya, a new monthly subscription can be refunded in full if requested within 14 days of the first payment [1]. "
            "I have passed your request to our Billing team. Best regards, SkyVault Support"
        )

    llm = scripted(classification(["billing", "refund"]), bad, good, verdicts(True))
    r = make(settings, kb, llm).process(
        email("I signed up for the monthly plan 5 days ago, can I get a refund?", subject="Refund monthly plan")
    )
    assert r.status == "drafted" and r.revisions == 1
    assert r.path == ["screen", "classify", "retrieve", "draft", "verify", "revise", "verify", "judge", "finalize"]
    assert "[1]" not in r.draft and r.citations[0]["source"] == "refund_policy"


def verdicts(*supported):
    return json.dumps({"verdicts": [{"n": i, "supported": s, "reason": "r"} for i, s in enumerate(supported, 1)]})


# The exact draft qwen3.5:9b wrote in a live run: valid citation, no numbers, but it mixes up
# service credits with refunds. Only the judge can see that.
CREDITS_MIXUP = (
    "Hi Priya, a new monthly subscription can be refunded in full if requested within 14 days [1]. "
    "Once approved, credits are applied to your next invoice [1]. Best regards, SkyVault Support"
)


def test_judge_rejects_a_semantic_mixup_and_the_revision_passes(settings, kb):
    fixed = "Hi Priya, a new monthly subscription can be refunded in full if requested within 14 days [1]. Best regards"
    llm = scripted(classification(["billing", "refund"]), CREDITS_MIXUP, verdicts(True, False), fixed, verdicts(True))
    r = make(settings, kb, llm).process(email("refund for my new monthly plan please", subject="Refund"))
    assert r.status == "drafted" and "credits" not in r.draft
    assert r.path[3:] == ["draft", "verify", "judge", "revise", "verify", "judge", "finalize"]
    assert "not supported by the policy" in llm.calls[3][-1].content  # the revise prompt names the sentence


def test_judge_unavailable_fails_closed_to_template(settings, kb):
    llm = scripted(classification(["billing", "refund"]), CREDITS_MIXUP, LLMError("timeout"))
    triage = make(settings, kb, llm)
    triage.s.max_revisions = 0
    r = triage.process(email("refund for my new monthly plan please", subject="Refund"))
    assert r.status == "needs_review" and any("refund check unavailable" in p for p in r.problems)
    assert "published refund policy" in r.draft


def test_missing_verdict_counts_as_unsupported(settings, kb):
    llm = scripted(classification(["billing", "refund"]), CREDITS_MIXUP, verdicts(True))  # only 1 of 2 verdicts
    triage = make(settings, kb, llm)
    triage.s.max_revisions = 0
    r = triage.process(email("refund for my new monthly plan please", subject="Refund"))
    assert r.status == "needs_review" and any("no verdict" in p for p in r.problems)


def test_refund_hallucination_twice_falls_back_to_verbatim_policy(settings, kb):
    bad = "We will refund you in full within 60 days."
    llm = scripted(classification(["billing", "refund"]), bad, bad)
    r = make(settings, kb, llm).process(email("I want a refund for my monthly plan", subject="Refund please"))
    assert r.status == "needs_review"
    assert "published refund policy" in r.draft and "60" not in r.draft
    assert any("commits to a refund" in p for p in r.problems)


def test_pipeline_bug_is_contained(settings, kb):
    llm = scripted(lambda messages: 1 / 0)  # an unexpected exception inside a node
    r = make(settings, kb, llm).process(email("hello"))
    assert r.status == "escalated" and r.flags[0].reason in ("pipeline_error", "classifier_unavailable")


def test_batch_counts_do_not_depend_on_thread_order(settings, kb):
    emails = [
        email(f"payment question {i}", id=f"b{i}", sender="sam@example.com", at=T0 + timedelta(hours=i))
        for i in range(4)
    ]
    llm = scripted(*[classification(["billing"])] * 8, *["Hi Sam, updating the card fixes most failures [1]."] * 8)
    results = make(settings, kb, llm).process_batch(emails, workers=4)
    assert [r.contact_count for r in results] == [1, 2, 3, 4]
    assert results[3].status == "escalated"


def test_every_step_is_traced(settings, kb):
    from aac.obs import Tracer, summarize_trace

    llm = scripted(classification(["technical"]))
    triage = SupportTriage(settings, llm, kb, ContactHistory(), Tracer(settings.trace_path))
    triage.process(email("our files are gone", subject="data"))
    summary = summarize_trace(settings.trace_path)
    assert {"screen", "classify", "escalate", "result"} <= set(summary["steps"])
    assert summary["guardrails"] == {"escalated:data_loss": 1}
