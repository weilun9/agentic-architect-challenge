from datetime import UTC, datetime, timedelta, timezone

import pytest

from aac.support.history import ContactHistory
from aac.support.models import Classification, Email
from aac.support.rules import detect_risks, queue_for


@pytest.mark.parametrize(
    "text, signal",
    [
        ("We suffered data loss after the update", "data_loss"),
        ("I lost all my files yesterday", "data_loss"),
        ("my photos have disappeared", "data_loss"),
        ("SkyVault is down for our whole office", "service_outage"),
        ("every page gives a 503 error", "service_outage"),
        ("there is an outage in Frankfurt", "service_outage"),
        ("I think my account was hacked", "security_breach"),
        ("someone signed in from another country", "security_breach"),
        ("there was unauthorized access to our files", "security_breach"),
    ],
)
def test_detects_risk(text, signal):
    assert signal in detect_risks(text)


@pytest.mark.parametrize(
    "text",
    [
        "How do I download an invoice?",
        "Can I change my login email?",
        "Love the new sharing options",
        "My payment failed",
    ],
)
def test_ordinary_emails_are_not_flagged(text):
    assert detect_risks(text) == {}


def test_queue_priority_security_first():
    assert queue_for(["repeat_contact", "security_breach"]) == "security-incident"
    assert queue_for(["service_outage"]) == "tier2-technical"
    assert queue_for(["repeat_contact"]) == "customer-care-lead"


def test_contact_window_counts_only_last_7_days_and_is_idempotent():
    h = ContactHistory()
    now = datetime(2026, 11, 2, 12, tzinfo=timezone(timedelta(hours=8)))
    for i, days_ago in enumerate([8, 6, 3, 1, 0]):
        h.record(f"e{i}", "A@x.com", now - timedelta(days=days_ago))
    h.record("e4", "a@x.com", now)  # same email processed twice
    assert h.count("a@x.com", now, 7) == 4  # the 8-day-old contact is outside the window
    assert h.count("a@x.com", now - timedelta(days=2), 7) == 3  # window (now-9d, now-2d]: 8, 6 and 3 days ago


def test_timezones_compare_correctly():
    h = ContactHistory()
    sg = timezone(timedelta(hours=8))
    h.record("a", "x@y.com", datetime(2026, 11, 2, 7, 0, tzinfo=sg))  # 23:00 UTC the day before
    assert h.count("x@y.com", datetime(2026, 11, 1, 23, 30, tzinfo=UTC), 7) == 1


def test_classification_normalises_small_model_output():
    c = Classification.model_validate(
        {"categories": ["Billing", "Refund", "nonsense"], "risk_signals": "Service Outage", "urgency": "high"}
    )
    assert c.categories == ["billing", "refund"]
    assert c.risk_signals == ["service_outage"]


def test_email_address_and_name():
    e = Email(id="1", sender="Priya Nair <Priya.Nair@Example.com>", body="hi", received_at=datetime(2026, 1, 1))
    assert e.address == "priya.nair@example.com"
    assert e.display_name == "Priya"
