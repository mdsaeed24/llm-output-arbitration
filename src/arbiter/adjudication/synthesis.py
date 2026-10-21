"""Verdict synthesis: combine adjudication (or short-circuit), rule-based dismissals, and
critic failures into one confidence-scored :class:`Verdict`. Pure function, no LLM."""

from __future__ import annotations

from collections.abc import Sequence

from arbiter.config import AppConfig, Dimension
from arbiter.schemas.critique import CriticFailure, Critique, Severity
from arbiter.schemas.disagreement import DisagreementReport
from arbiter.schemas.verdict import Adjudication, ConfirmedIssue, DismissedFlag, Verdict


def _invalid_quote_dismissals(critiques: Sequence[Critique]) -> list[DismissedFlag]:
    return [
        DismissedFlag(
            issue_ids=[i.id],
            raised_by=[i.critic],
            quote=i.quote,
            rule="invalid_quote",
            reason=f"Quote does not appear in the evaluated text ({i.invalid_reason}).",
        )
        for c in critiques
        for i in c.invalid_issues
    ]


def is_pass(confirmed: Sequence[ConfirmedIssue], config: AppConfig) -> bool:
    """An output passes unless a confirmed issue reaches ``verdict.fail_at_severity``."""
    fail_rank = Severity(config.verdict.fail_at_severity).rank
    return not any(ci.severity.rank >= fail_rank for ci in confirmed)


def low_confidence_dimensions(critiques: Sequence[Critique], threshold: float) -> list[Dimension]:
    """Dimensions with no successful critic, or whose critics are all below ``threshold``."""
    out = []
    for dim in Dimension:
        confs = [c.confidence for c in critiques if c.dimension is dim]
        if not confs or max(confs) < threshold:
            out.append(dim)
    return out


def synthesize_verdict(
    *,
    config: AppConfig,
    critiques: Sequence[Critique],
    failures: Sequence[CriticFailure],
    report: DisagreementReport,
    adjudication: Adjudication | None,
) -> Verdict:
    """Build the final verdict.

    Confidence = (adjudicator or critic) confidence × dimension coverage, where coverage is the
    fraction of the three core dimensions with at least one successful critic. So a verdict
    missing one dimension can never claim more than ~67% confidence.
    """
    low_dims = low_confidence_dimensions(critiques, config.disagreement.low_confidence_below)
    covered = {c.dimension for c in critiques}
    coverage = len(covered) / len(Dimension)
    status = "degraded" if failures else "ok"
    dismissed = _invalid_quote_dismissals(critiques)
    failure_note = (
        " Failed critics: "
        + "; ".join(f"{f.critic} ({f.error_type}: {f.message})" for f in failures)
        + "."
        if failures
        else ""
    )

    if not critiques:
        return Verdict(
            status="failed",
            passed=None,
            overall_score=None,
            confidence=0.0,
            summary="No critic produced a report, so no verdict could be reached." + failure_note,
            low_confidence_dimensions=list(Dimension),
        )

    if adjudication is None:  # unanimous pass → adjudicator skipped
        return Verdict(
            status=status,
            passed=True,
            overall_score=10,
            confidence=min(c.confidence for c in critiques) * coverage,
            dismissed_flags=dismissed,
            summary=(
                f"All {len(critiques)} critics found no problems with high confidence; "
                "the adjudicator was skipped." + failure_note
            ),
            low_confidence_dimensions=low_dims,
            short_circuited=True,
        )

    if adjudication.fallback_reason:
        status = "degraded"
    return Verdict(
        status=status,
        passed=is_pass(adjudication.confirmed, config),
        overall_score=adjudication.overall_score,
        confidence=adjudication.confidence * coverage,
        confirmed_issues=adjudication.confirmed,
        dismissed_flags=adjudication.dismissed + dismissed,
        summary=adjudication.summary + failure_note,
        low_confidence_dimensions=low_dims,
        adjudicator=adjudication.adjudicator,
    )
