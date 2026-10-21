"""Normalise every evaluated variant into the same list of :class:`Finding` objects.

Variants (all but the baseline are derived from ONE system run, so ablations cost nothing
and see identical critic outputs):

* ``system``           – the full pipeline: issues confirmed by the adjudicator
* ``baseline``         – the single-reviewer baseline
* ``union``            – the "no adjudicator" ablation: every valid critic issue, grouped
* ``critic:<name>``    – one critic alone: its own valid issues

Invalid-quote issues are excluded everywhere (they are dismissed by rule in every variant);
their counts are reported separately as a hallucination statistic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from arbiter.baseline import BaselineResult
from arbiter.config import AppConfig, Dimension
from arbiter.llm.telemetry import CallRecord
from arbiter.schemas.critique import Critique, Issue, Severity
from arbiter.schemas.verdict import ArbitrationResult, ConfirmedIssue

SYSTEM, BASELINE, UNION = "system", "baseline", "union"


def critic_variant(name: str) -> str:
    """Variant name for one critic alone."""
    return f"critic:{name}"


class Finding(BaseModel):
    """One problem a variant reported, in a variant-neutral shape."""

    id: str
    kind: Literal["span", "omission"] = "span"
    dimension: Dimension
    span_start: int | None = None
    span_end: int | None = None
    severity: Severity
    description: str
    critics: list[str]

    @property
    def span(self) -> tuple[int, int] | None:
        """``(start, end)`` in the evaluated output."""
        if self.span_start is None or self.span_end is None:
            return None
        return self.span_start, self.span_end


class CaseRun(BaseModel):
    """Raw outputs for one test case (persisted, so scoring can be re-run for free)."""

    case_id: str
    system: ArbitrationResult | None = None
    baseline: BaselineResult | None = None
    system_error: str | None = None
    baseline_error: str | None = None


class VariantCase(BaseModel):
    """One variant's outcome on one case."""

    variant: str
    case_id: str
    findings: list[Finding]
    passed: bool | None
    cost_usd: float
    latency_s: float
    invalid_quotes: int = 0


def _kind(dimensions: Sequence[Dimension]) -> tuple[Literal["span", "omission"], Dimension]:
    """Omission only when the issue is purely completeness; else span (factual first)."""
    if list(dimensions) == [Dimension.COMPLETENESS]:
        return "omission", Dimension.COMPLETENESS
    span_dims = [d for d in dimensions if d is not Dimension.COMPLETENESS]
    return "span", span_dims[0]


def _from_issue(issue: Issue, fid: str) -> Finding:
    kind, dim = _kind([issue.dimension])
    return Finding(
        id=fid,
        kind=kind,
        dimension=dim,
        span_start=issue.span_start,
        span_end=issue.span_end,
        severity=issue.severity,
        description=issue.problem,
        critics=[issue.critic],
    )


def _from_confirmed(ci: ConfirmedIssue, fid: str, problems: dict[str, str]) -> Finding:
    kind, dim = _kind(ci.dimensions)
    return Finding(
        id=fid,
        kind=kind,
        dimension=dim,
        span_start=ci.span_start,
        span_end=ci.span_end,
        severity=ci.severity,
        # the critics' own problem statements (concise; what the judge compares)
        description=" | ".join(problems[i] for i in ci.issue_ids if i in problems) or ci.evidence,
        critics=ci.raised_by,
    )


def _latency(calls: Sequence[CallRecord]) -> float:
    return sum(c.latency_s for c in calls)


def critical_path_latency(result: ArbitrationResult, *, with_adjudicator: bool = True) -> float:
    """LLM time on the critical path: the slowest critic (critics run in parallel) plus the
    adjudicator. Excludes queueing and backoff sleeps, so concurrency in the eval run doesn't
    distort the comparison."""
    per_critic = [_latency(c.calls) for c in result.critiques]
    per_critic += [_latency(f.calls) for f in result.failures]
    slowest = max(per_critic, default=0.0)
    adjudication = result.adjudication
    adj = _latency(adjudication.calls) if (with_adjudicator and adjudication) else 0.0
    return slowest + adj


def _passed(findings: Sequence[Finding], config: AppConfig) -> bool:
    fail_rank = Severity(config.verdict.fail_at_severity).rank
    return not any(f.severity.rank >= fail_rank for f in findings)


def system_variant(run: CaseRun, config: AppConfig) -> VariantCase | None:
    """The full system's confirmed issues and verdict."""
    r = run.system
    if r is None:
        return None
    problems = {i.id: i.problem for c in r.critiques for i in c.issues}
    findings = [
        _from_confirmed(ci, f"sys-{n}", problems)
        for n, ci in enumerate(r.verdict.confirmed_issues, start=1)
    ]
    return VariantCase(
        variant=SYSTEM,
        case_id=run.case_id,
        findings=findings,
        passed=r.verdict.passed,
        cost_usd=r.cost_usd,
        latency_s=critical_path_latency(r),
        invalid_quotes=sum(len(c.invalid_issues) for c in r.critiques),
    )


def union_variant(run: CaseRun, config: AppConfig) -> VariantCase | None:
    """Every valid critic issue, grouped by span exactly as the disagreement detector did."""
    r = run.system
    if r is None:
        return None
    issues = {i.id: i for c in r.critiques for i in c.issues}
    findings = []
    for g in r.report.groups:
        kind, dim = _kind(g.dimensions)
        findings.append(
            Finding(
                id=f"union-{g.id}",
                kind=kind,
                dimension=dim,
                span_start=g.span_start,
                span_end=g.span_end,
                severity=g.max_severity,
                description=" | ".join(issues[i].problem for i in g.issue_ids),
                critics=g.critics,
            )
        )
    critic_calls = [c for cr in r.critiques for c in cr.calls] + [
        c for f in r.failures for c in f.calls
    ]
    return VariantCase(
        variant=UNION,
        case_id=run.case_id,
        findings=findings,
        passed=_passed(findings, config) if r.critiques else None,
        cost_usd=sum(c.cost_usd for c in critic_calls),
        latency_s=critical_path_latency(r, with_adjudicator=False),
        invalid_quotes=sum(len(c.invalid_issues) for c in r.critiques),
    )


def critic_alone_variant(run: CaseRun, critic: str, config: AppConfig) -> VariantCase | None:
    """One critic on its own. ``passed`` is None if that critic failed on this case."""
    r = run.system
    if r is None:
        return None
    critique: Critique | None = next((c for c in r.critiques if c.critic == critic), None)
    failure = next((f for f in r.failures if f.critic == critic), None)
    if critique is None:
        calls = failure.calls if failure else []
        return VariantCase(
            variant=critic_variant(critic),
            case_id=run.case_id,
            findings=[],
            passed=None,
            cost_usd=sum(c.cost_usd for c in calls),
            latency_s=_latency(calls),
        )
    findings = [_from_issue(i, i.id) for i in critique.valid_issues]
    return VariantCase(
        variant=critic_variant(critic),
        case_id=run.case_id,
        findings=findings,
        passed=_passed(findings, config),
        cost_usd=critique.cost_usd,
        latency_s=critique.latency_s,
        invalid_quotes=len(critique.invalid_issues),
    )


def baseline_variant(run: CaseRun, config: AppConfig) -> VariantCase | None:
    """The single reviewer's valid issues and verdict."""
    b = run.baseline
    if b is None:
        return None
    findings = [_from_issue(i, i.id) for i in b.issues if i.valid]
    return VariantCase(
        variant=BASELINE,
        case_id=run.case_id,
        findings=findings,
        passed=b.verdict.passed,
        cost_usd=b.cost_usd,
        latency_s=_latency(b.calls),
        invalid_quotes=sum(not i.valid for i in b.issues),
    )


def all_variants(run: CaseRun, critics: Sequence[str], config: AppConfig) -> list[VariantCase]:
    """Every variant for a case (missing ones — e.g. a crashed run — are skipped)."""
    out = [
        system_variant(run, config),
        baseline_variant(run, config),
        union_variant(run, config),
        *[critic_alone_variant(run, c, config) for c in critics],
    ]
    return [v for v in out if v is not None]
