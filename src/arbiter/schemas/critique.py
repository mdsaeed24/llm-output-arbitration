"""Critique contracts.

There are two layers on purpose:

* ``*Draft`` models are what the LLM is asked to produce. They contain only things the
  critic can legitimately judge (quote, problem, severity, score, confidence).
* ``Issue`` / ``Critique`` are what the rest of the system consumes. Fields like ``valid``,
  ``span_start`` and ``dimension`` are stamped on by *our code* after the call, so a critic
  can never mark its own hallucinated quote as valid.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from arbiter.config import Dimension
from arbiter.llm.telemetry import CallRecord


class Severity(StrEnum):
    """How much an issue matters."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        """Numeric level (low=1 … high=3) for comparing severities across critics."""
        return {"low": 1, "medium": 2, "high": 3}[self.value]


MatchMethod = Literal["exact", "normalized", "fragments", "none"]


class IssueDraft(BaseModel):
    """One problem as reported by a critic LLM."""

    quote: str = Field(
        min_length=1,
        description="Exact, verbatim span copied from the evaluated text that shows the problem.",
    )
    problem: str = Field(min_length=1, description="What is wrong and, if known, what is right.")
    severity: Severity


class VerifiedClaimDraft(BaseModel):
    """A span the critic checked and judges CORRECT. Lets us detect opposite judgments: one
    critic flags a span that another critic explicitly verified."""

    quote: str = Field(min_length=1, description="Exact, verbatim span from the evaluated text.")
    note: str = Field(default="", description="Why it is correct (one short sentence).")


class CritiqueDraft(BaseModel):
    """Raw critic output. Field order matters: analysis first, score last, so the score is
    written *after* the model has committed to its issues (cheap 'think before you score')."""

    analysis: str = Field(description="2-4 sentence assessment of this dimension.")
    issues: list[IssueDraft] = Field(default_factory=list, max_length=20)
    verified_claims: list[VerifiedClaimDraft] = Field(default_factory=list, max_length=8)
    score: int = Field(ge=1, le=5, description="5 = no problems … 1 = fundamentally unsound.")
    confidence: float = Field(
        ge=0.0, le=1.0, description="How sure the critic is that its own assessment is right."
    )


class Issue(IssueDraft):
    """An issue after quote validation against the evaluated text."""

    model_config = ConfigDict(frozen=True)

    id: str
    critic: str
    dimension: Dimension
    valid: bool = Field(description="False if the quote was not found in the evaluated text.")
    span_start: int | None = Field(None, description="Char offset of the quote in the text.")
    span_end: int | None = None
    match_method: MatchMethod = "none"
    invalid_reason: str | None = None

    @property
    def span(self) -> tuple[int, int] | None:
        """``(start, end)`` in the evaluated text, or ``None`` for an invalid quote."""
        if self.span_start is None or self.span_end is None:
            return None
        return self.span_start, self.span_end


class VerifiedClaim(VerifiedClaimDraft):
    """A verified claim after quote validation."""

    model_config = ConfigDict(frozen=True)

    critic: str
    dimension: Dimension
    valid: bool
    span_start: int | None = None
    span_end: int | None = None

    @property
    def span(self) -> tuple[int, int] | None:
        """``(start, end)`` in the evaluated text, or ``None`` for an invalid quote."""
        if self.span_start is None or self.span_end is None:
            return None
        return self.span_start, self.span_end


class Critique(BaseModel):
    """A completed critic report, ready for disagreement detection and adjudication."""

    critic: str
    dimension: Dimension
    model_key: str
    model_id: str
    provider: str
    analysis: str
    score: int = Field(ge=1, le=5)
    confidence: float = Field(ge=0.0, le=1.0)
    issues: list[Issue]
    verified_claims: list[VerifiedClaim] = Field(default_factory=list)
    calls: list[CallRecord] = Field(default_factory=list)

    @property
    def valid_issues(self) -> list[Issue]:
        """Issues whose quote was found in the text."""
        return [i for i in self.issues if i.valid]

    @property
    def invalid_issues(self) -> list[Issue]:
        """Issues with a quote that does not appear in the text (likely hallucinated)."""
        return [i for i in self.issues if not i.valid]

    @property
    def cost_usd(self) -> float:
        """Spend across every attempt for this critique."""
        return sum(c.cost_usd for c in self.calls)

    @property
    def latency_s(self) -> float:
        """Wall time spent in LLM calls for this critique."""
        return sum(c.latency_s for c in self.calls)


class CriticFailure(BaseModel):
    """Recorded when a critic could not produce a critique after retries."""

    critic: str
    dimension: Dimension
    model_key: str
    error_type: str
    message: str
    calls: list[CallRecord] = Field(default_factory=list)
