"""Disagreement detection — deterministic, no LLM.

Steps:
1. Ignore invalid-quote issues (they can't be located; they are dismissed by rule later).
2. Group valid factual/logic issues whose spans overlap (union-find), across all critics.
   Each completeness issue is its own group (its quote is just an anchor near a gap).
3. Derive disagreements:
   * UNCORROBORATED     – group raised by one critic only.
   * SEVERITY_CONFLICT  – group whose critics' severities differ by 2+ levels (low vs high).
   * OPPOSITE_JUDGMENT  – group overlapping a span that another critic listed as verified.
4. Unanimous pass – every critic ran, no valid issues, every confidence >= threshold.

Expectation worth stating: because each critic owns one dimension, most issues will be
UNCORROBORATED. That is by design, not a bug — single-source issues are exactly the ones the
adjudicator needs to verify. Corroboration mostly comes from factual/logic overlap (e.g. an
internal contradiction) or from extra critics on the same dimension.
"""

from __future__ import annotations

from collections.abc import Sequence

from arbiter.config import Dimension, DisagreementConfig
from arbiter.schemas.critique import CriticFailure, Critique, Issue, Severity
from arbiter.schemas.disagreement import (
    Disagreement,
    DisagreementKind,
    DisagreementReport,
    IssueGroup,
)
from arbiter.spans import Span, spans_overlap, union


def _group_issues(issues: list[Issue], min_ratio: float) -> list[list[Issue]]:
    """Union-find over span overlap for factual/logic issues; omissions stay singletons."""
    spanned = [i for i in issues if i.dimension is not Dimension.COMPLETENESS]
    parent = list(range(len(spanned)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(spanned)):
        for b in range(a + 1, len(spanned)):
            sa, sb = spanned[a].span, spanned[b].span
            if sa and sb and spans_overlap(sa, sb, min_ratio):
                parent[find(a)] = find(b)

    clusters: dict[int, list[Issue]] = {}
    for idx, issue in enumerate(spanned):
        clusters.setdefault(find(idx), []).append(issue)
    groups = list(clusters.values())
    groups += [[i] for i in issues if i.dimension is Dimension.COMPLETENESS]
    return sorted(groups, key=lambda g: min(i.span_start or 0 for i in g))


def _make_group(gid: str, issues: list[Issue]) -> IssueGroup:
    spans: list[Span] = [i.span for i in issues if i.span]
    start, end = union(spans)
    severities = sorted((i.severity for i in issues), key=lambda s: s.rank)
    return IssueGroup(
        id=gid,
        issue_ids=[i.id for i in issues],
        critics=sorted({i.critic for i in issues}),
        dimensions=sorted({i.dimension for i in issues}),
        span_start=start,
        span_end=end,
        max_severity=severities[-1],
        min_severity=severities[0],
    )


def _critic_severity(group_issues: list[Issue], critic: str) -> Severity:
    return max((i.severity for i in group_issues if i.critic == critic), key=lambda s: s.rank)


def detect_disagreements(
    critiques: Sequence[Critique],
    failures: Sequence[CriticFailure],
    config: DisagreementConfig,
) -> DisagreementReport:
    """Compare critic reports and list what the adjudicator must resolve."""
    valid = [i for c in critiques for i in c.valid_issues]
    invalid_ids = [i.id for c in critiques for i in c.invalid_issues]
    critic_names = [c.critic for c in critiques]
    claims = [
        claim
        for c in critiques
        for claim in c.verified_claims
        if claim.valid and claim.dimension is not Dimension.COMPLETENESS
    ]

    groups: list[IssueGroup] = []
    disagreements: list[Disagreement] = []

    def add(kind: DisagreementKind, group: IssueGroup, others: list[str], detail: str) -> None:
        disagreements.append(
            Disagreement(
                id=f"d{len(disagreements) + 1}",
                kind=kind,
                group_id=group.id,
                issue_ids=group.issue_ids,
                raised_by=group.critics,
                other_critics=others,
                detail=detail,
            )
        )

    for n, members in enumerate(_group_issues(valid, config.span_overlap_min_ratio), start=1):
        group = _make_group(f"g{n}", members)
        groups.append(group)
        span = (group.span_start, group.span_end)

        is_omission = group.dimensions == [Dimension.COMPLETENESS]
        verifiers = sorted(
            {
                claim.critic
                for claim in claims
                if not is_omission
                and claim.critic not in group.critics
                and claim.span
                and spans_overlap(span, claim.span, config.span_overlap_min_ratio)
            }
        )
        if verifiers:
            # Stronger than "uncorroborated": another critic actively judged it correct.
            add(
                DisagreementKind.OPPOSITE_JUDGMENT,
                group,
                verifiers,
                f"Flagged by {', '.join(group.critics)} but verified as correct by "
                f"{', '.join(verifiers)}.",
            )
        elif not group.corroborated:
            others = [c for c in critic_names if c not in group.critics]
            add(
                DisagreementKind.UNCORROBORATED,
                group,
                others,
                f"Raised only by {group.critics[0]} ({group.max_severity.value}); "
                f"not raised by {', '.join(others) or 'any other critic'}.",
            )

        if group.corroborated:
            by_critic = {c: _critic_severity(members, c) for c in group.critics}
            ranks = [s.rank for s in by_critic.values()]
            if max(ranks) - min(ranks) >= 2:
                add(
                    DisagreementKind.SEVERITY_CONFLICT,
                    group,
                    [],
                    "Same span rated "
                    + ", ".join(f"{c}={s.value}" for c, s in sorted(by_critic.items())),
                )

    unanimous_pass = (
        not failures
        and bool(critiques)
        and not valid
        and all(c.confidence >= config.unanimous_pass_min_confidence for c in critiques)
    )
    return DisagreementReport(
        groups=groups,
        disagreements=disagreements,
        invalid_issue_ids=invalid_ids,
        unanimous_pass=unanimous_pass,
    )
