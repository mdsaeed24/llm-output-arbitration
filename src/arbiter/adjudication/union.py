"""Union "adjudicator": trusts every critic, confirms every valid issue. No LLM call.

Used (a) as the Phase 2 placeholder so the graph runs end to end, and (b) in Phase 5 as the
"system without the adjudicator" ablation — the raw union of critic findings.
"""

from __future__ import annotations

from statistics import mean

from arbiter.adjudication.base import AdjudicationInput, deterministic_score
from arbiter.schemas.critique import Issue
from arbiter.schemas.verdict import Adjudication, ConfirmedIssue


class UnionAdjudicator:
    """Confirms each issue group; severity = the highest any critic gave."""

    name = "union"

    async def adjudicate(self, inp: AdjudicationInput) -> Adjudication:
        """Confirm every valid issue group without any LLM reasoning."""
        issues: dict[str, Issue] = {i.id: i for c in inp.critiques for i in c.issues}
        confidence_of = {c.critic: c.confidence for c in inp.critiques}
        confirmed = [
            ConfirmedIssue(
                group_id=g.id,
                issue_ids=g.issue_ids,
                raised_by=g.critics,
                dimensions=g.dimensions,
                severity=g.max_severity,
                quote=inp.output[g.span_start : g.span_end],
                span_start=g.span_start,
                span_end=g.span_end,
                evidence=" | ".join(
                    f"{issues[i].critic}: {issues[i].problem}" for i in g.issue_ids
                ),
                confidence=max(confidence_of[c] for c in g.critics),
            )
            for g in inp.report.groups
        ]
        n = len(confirmed)
        return Adjudication(
            adjudicator=self.name,
            overall_score=deterministic_score(c.severity for c in confirmed),
            confidence=mean(c.confidence for c in inp.critiques) if inp.critiques else 0.0,
            confirmed=confirmed,
            summary=(
                f"Union of critic findings: {n} issue{'s' * (n != 1)} confirmed without "
                "adjudication."
            ),
        )
