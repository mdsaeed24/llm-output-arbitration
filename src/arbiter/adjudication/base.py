"""Adjudicator interface and shared scoring helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from pydantic import BaseModel

from arbiter.schemas.critique import Critique, Severity
from arbiter.schemas.disagreement import DisagreementReport
from arbiter.schemas.verdict import Adjudication

SEVERITY_PENALTY = {Severity.LOW: 1, Severity.MEDIUM: 2, Severity.HIGH: 4}
"""Points deducted from 10 per confirmed issue in the deterministic score."""


class AdjudicationInput(BaseModel):
    """Everything an adjudicator may look at."""

    question: str | None
    output: str
    critiques: list[Critique]
    report: DisagreementReport


class Adjudicator(Protocol):
    """Resolves critic disagreements into an :class:`Adjudication`.

    Implementations: ``UnionAdjudicator`` (no LLM; also the "no adjudicator" ablation) and,
    from Phase 3, the LLM adjudicator.
    """

    name: str

    async def adjudicate(self, inp: AdjudicationInput) -> Adjudication:
        """Return confirmed issues, dismissed flags, score, confidence and summary."""
        ...


def deterministic_score(severities: Iterable[Severity]) -> int:
    """10 minus severity penalties, floored at 1 (one high issue → 6; one medium → 8)."""
    return max(1, 10 - sum(SEVERITY_PENALTY[s] for s in severities))
