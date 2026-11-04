"""The refund / grounding guardrail. Each test is one way a model could hallucinate policy."""

from aac.support.models import Passage
from aac.support.verify import check_draft, sentences

MONTHLY = Passage(
    id="refund_policy#2",
    source="refund_policy",
    section="Monthly plans",
    text="A new monthly subscription can be refunded in full if the refund is requested within 14 days of the first payment.",
)
PAYOUT = Passage(
    id="refund_policy#5",
    source="refund_policy",
    section="How refunds are paid",
    text="Refunds usually appear within 5 to 10 business days after approval, depending on the bank.",
)
SYNC = Passage(id="technical_faq#1", source="technical_faq", section="Files are not syncing", text="Click Resume sync.")
PASSAGES = [MONTHLY, PAYOUT, SYNC]


def test_grounded_refund_reply_passes():
    draft = (
        "Hi Priya, a new monthly plan can be refunded in full if you ask within 14 days of the first payment [1]. "
        "I have passed your request to our Billing team. If it is approved, it usually appears within 5 to 10 business days [2]."
    )
    problems, cited = check_draft(draft, PASSAGES)
    assert problems == []
    assert cited == [1, 2]


def test_invented_refund_window_is_caught():
    problems, _ = check_draft("Monthly plans can be refunded within 30 days [1].", PASSAGES)
    assert any("'30 days' is not in the cited passage" in p for p in problems)


def test_number_without_citation_is_caught():
    problems, _ = check_draft("Refunds take 5 to 10 business days.", PASSAGES)
    assert any("without citing" in p for p in problems)


def test_refund_terms_must_cite_the_refund_policy_not_another_page():
    problems, _ = check_draft("You are eligible for a full refund [3].", PASSAGES)
    assert any("without citing the refund policy" in p for p in problems)


def test_refund_promise_is_blocked():
    problems, _ = check_draft("Don't worry, we will refund you today.", PASSAGES)
    assert any("commits to a refund" in p for p in problems)
    problems, _ = check_draft("Your refund has been processed.", PASSAGES)
    assert any("commits to a refund" in p for p in problems)


def test_conditional_refund_statement_is_allowed():
    problems, _ = check_draft("If Billing approves it, you will receive a refund to your card [2].", PASSAGES)
    assert not any("commits" in p for p in problems)


def test_negated_guarantee_is_not_a_promise():
    # Found in a live run: qwen3.5:9b wrote this and the first version of the check blocked it.
    problems, _ = check_draft("Support agents cannot guarantee approval but can open the request.", PASSAGES)
    assert problems == []


def test_negation_elsewhere_does_not_hide_a_promise():
    assert any("commits" in p for p in check_draft("Don't worry, we will refund you today.", PASSAGES)[0])


def test_deciding_eligibility_is_a_promise():
    # Also from a live run: "Since you requested ... within 14 days, you are eligible for a full refund."
    problems, _ = check_draft("Since you asked within 14 days, you are eligible for a full refund [1].", PASSAGES)
    assert any("commits" in p for p in problems)
    assert check_draft("You are not eligible for a refund after 14 days [1].", PASSAGES)[0] == []


def test_citation_to_missing_passage_is_caught():
    problems, _ = check_draft("Click Resume sync [7].", PASSAGES)
    assert any("do not exist" in p for p in problems)


def test_numbers_the_customer_wrote_are_allowed():
    problems, _ = check_draft(
        "We have passed invoice INV-20931 to the Billing team.",
        PASSAGES,
        customer_text="invoice INV-20931 charged twice",
    )
    assert problems == []


def test_empty_draft_fails():
    assert check_draft("  ", PASSAGES)[0] == ["the draft is empty"]


def test_citation_after_full_stop_stays_with_its_sentence():
    assert sentences("First claim. [1] Second claim [2].") == ["First claim[1].", "Second claim [2]."]
