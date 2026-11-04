"""Data shapes for the triage pipeline. The LLM's output is validated against `Classification`."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal["billing", "refund", "technical", "account", "feedback", "other"]
RiskSignal = Literal["data_loss", "service_outage", "security_breach"]
CATEGORIES: tuple[str, ...] = Category.__args__
RISK_SIGNALS: tuple[str, ...] = RiskSignal.__args__

_ADDRESS = re.compile(r"<([^>]+)>")


class Email(BaseModel):
    id: str
    sender: str
    subject: str = ""
    body: str
    received_at: datetime

    @property
    def address(self) -> str:
        m = _ADDRESS.search(self.sender)
        return (m.group(1) if m else self.sender).strip().lower()

    @property
    def display_name(self) -> str:
        name = _ADDRESS.sub("", self.sender).strip().strip('"')
        return name.split()[0] if name and "@" not in name else "there"

    @property
    def text(self) -> str:
        return f"{self.subject}\n{self.body}"


class Classification(BaseModel):
    categories: list[Category] = Field(min_length=1)
    urgency: Literal["low", "normal", "high"] = "normal"
    risk_signals: list[RiskSignal] = Field(default_factory=list)
    summary: str = ""

    @field_validator("categories", "risk_signals", mode="before")
    @classmethod
    def _normalise(cls, v, info):
        # Small models write "Billing" or "Service Outage"; accept those, drop anything unknown.
        allowed = CATEGORIES if info.field_name == "categories" else RISK_SIGNALS
        if isinstance(v, str):
            v = [v]
        out = []
        for item in v or []:
            key = str(item).strip().lower().replace(" ", "_").replace("-", "_")
            if key in allowed and key not in out:
                out.append(key)
        return out


class Flag(BaseModel):
    reason: str  # data_loss | service_outage | security_breach | repeat_contact | classifier_unavailable
    source: Literal["rule", "llm", "system"]
    evidence: str = ""


class Passage(BaseModel):
    id: str
    source: str
    section: str
    text: str
    score: float = 0.0

    @property
    def is_refund_policy(self) -> bool:
        return "refund" in self.source.lower()


class TriageResult(BaseModel):
    email_id: str
    status: Literal["escalated", "drafted", "needs_review"]
    queue: str
    categories: list[str] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)
    contact_count: int = 0
    draft: str = ""
    handoff_note: str = ""
    citations: list[dict] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    revisions: int = 0
    path: list[str] = Field(default_factory=list)
    latency_ms: int = 0
