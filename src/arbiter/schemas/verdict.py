"""Adjudication and final verdict contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from arbiter.config import Dimension
from arbiter.llm.telemetry import CallRecord
from arbiter.schemas.critique import CriticFailure, Critique, Severity
from arbiter.schemas.disagreement import DisagreementReport


class ConfirmedIssue(BaseModel):
    """An issue the adjudicator (or the union rule) accepted as real."""

    group_id: str
    issue_ids: list[str]
    raised_by: list[str] = Field(description="Critics that raised it.")
    dimensions: list[Dimension]
    severity: Severity
    quote: str
    span_start: int | None = None
    span_end: int | None = None
    evidence: str
    confidence: float = Field(ge=0, le=1, description="Confidence this issue is real.")
    disagreement_ids: list[str] = Field(
        default_factory=list, description="Disagreements the adjudicator resolved here."
    )


class DismissedFlag(BaseModel):
    """A critic flag that was rejected, with the reason (kept for auditability)."""

    issue_ids: list[str]
    raised_by: list[str]
    quote: str
    span_start: int | None = None
    span_end: int | None = None
    rule: Literal["invalid_quote", "adjudicator"]
    reason: str
    disagreement_ids: list[str] = Field(default_factory=list)


class Adjudication(BaseModel):
    """What an adjudicator returns."""

    adjudicator: str
    overall_score: int = Field(ge=1, le=10)
    confidence: float = Field(ge=0, le=1)
    confirmed: list[ConfirmedIssue]
    dismissed: list[DismissedFlag] = Field(default_factory=list)
    summary: str
    calls: list[CallRecord] = Field(default_factory=list)
    fallback_reason: str | None = Field(
        None, description="Set when the LLM adjudicator failed and the union rule was used."
    )


class Verdict(BaseModel):
    """The single, confidence-scored outcome of an arbitration."""

    status: Literal["ok", "degraded", "failed"] = Field(
        description="degraded = a critic or the adjudicator failed; failed = no critic succeeded."
    )
    passed: bool | None
    overall_score: int | None = Field(None, ge=1, le=10)
    confidence: float = Field(ge=0, le=1)
    confirmed_issues: list[ConfirmedIssue] = Field(default_factory=list)
    dismissed_flags: list[DismissedFlag] = Field(default_factory=list)
    summary: str
    low_confidence_dimensions: list[Dimension] = Field(default_factory=list)
    short_circuited: bool = False
    adjudicator: str | None = None


class ArbitrationResult(BaseModel):
    """Full audit record of one arbitration (stored in SQLite in Phase 6)."""

    id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    question: str | None
    output: str
    critiques: list[Critique]
    failures: list[CriticFailure]
    report: DisagreementReport
    adjudication: Adjudication | None
    verdict: Verdict
    latency_s: float

    @property
    def calls(self) -> list[CallRecord]:
        """Every LLM attempt made for this arbitration."""
        calls = [c for cr in self.critiques for c in cr.calls]
        calls += [c for f in self.failures for c in f.calls]
        calls += self.adjudication.calls if self.adjudication else []
        return calls

    @property
    def cost_usd(self) -> float:
        """Total spend for this arbitration."""
        return sum(c.cost_usd for c in self.calls)
