"""Output of the disagreement detector: issue groups, disagreements, short-circuit flag."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from arbiter.config import Dimension
from arbiter.schemas.critique import Severity


class IssueGroup(BaseModel):
    """Valid issues that refer to the same text (or a single omission claim).

    Factual/logic issues from any critics are grouped by span overlap. A completeness issue is
    always its own group: its quote is only an anchor near the gap, so overlapping another
    critic's span says nothing about agreement.
    """

    id: str
    issue_ids: list[str]
    critics: list[str]
    dimensions: list[Dimension]
    span_start: int
    span_end: int
    max_severity: Severity
    min_severity: Severity

    @property
    def corroborated(self) -> bool:
        """Raised independently by two or more critics."""
        return len(self.critics) >= 2


class DisagreementKind(StrEnum):
    """The three disagreement types the adjudicator must resolve."""

    UNCORROBORATED = "uncorroborated"  # one critic raised it, the others did not
    SEVERITY_CONFLICT = "severity_conflict"  # same span, severities 2+ levels apart
    OPPOSITE_JUDGMENT = "opposite_judgment"  # flagged by one, verified correct by another


class Disagreement(BaseModel):
    """One point the adjudicator must reason through explicitly."""

    id: str
    kind: DisagreementKind
    group_id: str
    issue_ids: list[str]
    raised_by: list[str]
    other_critics: list[str] = Field(
        description="Uncorroborated: critics that ran and did not raise it. "
        "Opposite judgment: critics that verified the span as correct."
    )
    detail: str


class DisagreementReport(BaseModel):
    """Everything downstream nodes need to know about how the critics compare."""

    groups: list[IssueGroup]
    disagreements: list[Disagreement]
    invalid_issue_ids: list[str] = Field(description="Quotes not found in the text.")
    unanimous_pass: bool = Field(
        description="All critics ran, none raised a valid issue, all confident."
    )

    @property
    def agreements(self) -> list[IssueGroup]:
        """Groups raised by 2+ critics."""
        return [g for g in self.groups if g.corroborated]
