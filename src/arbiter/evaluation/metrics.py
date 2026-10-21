"""Scoring: per-variant metrics, paired significance tests, critic and adjudicator analytics.

Definitions (also printed in the report):
* recall            = planted issues caught / planted issues
* precision         = findings that caught something / all findings
* false flag        = a finding that matches no planted issue (every finding on a clean case
                      is one). Reported as a total, per case, and "≥ medium" only.
* clean flagged     = share of clean cases with at least one finding
* verdict accuracy  = clean cases passed + flawed cases failed, over all cases. A missing
                      verdict (everything failed) counts as wrong.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations

from pydantic import BaseModel

from arbiter.evaluation.findings import (
    SYSTEM,
    UNION,
    CaseRun,
    Finding,
    VariantCase,
    critic_variant,
)
from arbiter.evaluation.matching import Match
from arbiter.schemas.critique import Severity
from arbiter.schemas.testcase import EvalCase, PlantedSpanIssue
from arbiter.spans import overlap_ratio

PLANTED_TYPES = ("factual", "logic", "completeness")


class ScoredCase(BaseModel):
    """One variant on one case, after matching."""

    case_id: str
    variant: str
    is_clean: bool
    planted_types: list[str]
    caught: list[bool]
    caught_by: list[str | None]  # "span" | "judge" | None, per planted issue
    findings: list[Finding]
    matched_finding_ids: list[str]
    matches: list[Match]
    passed: bool | None
    cost_usd: float
    latency_s: float
    invalid_quotes: int
    restatement_ids: list[str] = []

    @property
    def verdict_correct(self) -> bool:
        """Clean → must pass; flawed → must fail. None counts as wrong."""
        if self.passed is None:
            return False
        return self.passed == self.is_clean

    @property
    def false_flags(self) -> list[Finding]:
        """Findings that caught nothing."""
        matched = set(self.matched_finding_ids)
        return [f for f in self.findings if f.id not in matched]

    @property
    def restatements(self) -> list[Finding]:
        """False flags that are an omission-style complaint anchored on a planted factual/
        logic span — usually the completeness critic restating a planted error ("the summary
        reverses the core distinction…"). A sensitivity measure, NOT used by the strict rule."""
        ids = set(self.restatement_ids)
        return [f for f in self.false_flags if f.id in ids]


def score_case(
    case: EvalCase, variant: VariantCase, matches: Sequence[Match], min_overlap: float = 0.5
) -> ScoredCase:
    """Combine a variant's findings with its matches."""
    matched = {m.finding_id for m in matches}
    planted_spans = [
        loc for p in case.planted if isinstance(p, PlantedSpanIssue) and (loc := case.locate(p))
    ]
    restatements = [
        f.id
        for f in variant.findings
        if f.id not in matched
        and f.kind == "omission"
        and f.span is not None
        and any(overlap_ratio(sp, f.span) >= min_overlap for sp in planted_spans)
    ]
    caught_by: list[str | None] = [None] * len(case.planted)
    for m in matches:
        if caught_by[m.planted_index] is None or m.method == "span":
            caught_by[m.planted_index] = m.method
    return ScoredCase(
        case_id=case.id,
        variant=variant.variant,
        is_clean=case.is_clean,
        planted_types=[p.type for p in case.planted],
        caught=[c is not None for c in caught_by],
        caught_by=caught_by,
        findings=variant.findings,
        matched_finding_ids=sorted({m.finding_id for m in matches}),
        matches=list(matches),
        passed=variant.passed,
        cost_usd=variant.cost_usd,
        latency_s=variant.latency_s,
        invalid_quotes=variant.invalid_quotes,
        restatement_ids=restatements,
    )


# --------------------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (well-behaved at small n and near 0/1)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pairs (binomial, p=0.5)."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (1 << n)
    return min(1.0, 2 * tail)


# --------------------------------------------------------------------------- per variant


@dataclass
class VariantScore:
    """Aggregate metrics for one variant."""

    variant: str
    n_cases: int = 0
    planted_total: int = 0
    caught: int = 0
    by_type: dict[str, list[int]] = field(
        default_factory=lambda: {t: [0, 0] for t in PLANTED_TYPES}
    )
    caught_span: int = 0
    caught_judge: int = 0
    findings_total: int = 0
    findings_matched: int = 0
    false_flags: int = 0
    false_flags_ge_medium: int = 0
    restatements: int = 0
    clean_cases: int = 0
    clean_flagged: int = 0
    clean_flagged_ge_medium: int = 0
    verdict_correct: int = 0
    clean_passed: int = 0
    flawed_cases: int = 0
    flawed_failed: int = 0
    no_verdict: int = 0
    invalid_quotes: int = 0
    costs: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)

    @property
    def recall(self) -> float:
        """Planted issues caught / planted issues."""
        return self.caught / self.planted_total if self.planted_total else 0.0

    @property
    def precision(self) -> float:
        """Findings that caught something / all findings."""
        return self.findings_matched / self.findings_total if self.findings_total else 0.0

    @property
    def false_flags_excl_restatements(self) -> int:
        """Sensitivity: false flags if restatements of planted errors count as duplicates."""
        return self.false_flags - self.restatements

    @property
    def precision_excl_restatements(self) -> float:
        """Sensitivity: precision with restatements removed from the denominator."""
        n = self.findings_total - self.restatements
        return self.findings_matched / n if n else 0.0

    @property
    def accuracy(self) -> float:
        """Correct pass/fail verdicts / cases."""
        return self.verdict_correct / self.n_cases if self.n_cases else 0.0

    @property
    def mean_cost(self) -> float:
        """Average USD per case."""
        return statistics.fmean(self.costs) if self.costs else 0.0

    @property
    def mean_latency(self) -> float:
        """Average critical-path LLM seconds per case."""
        return statistics.fmean(self.latencies) if self.latencies else 0.0

    @property
    def median_latency(self) -> float:
        """Median critical-path LLM seconds per case."""
        return statistics.median(self.latencies) if self.latencies else 0.0


def aggregate(variant: str, scored: Sequence[ScoredCase]) -> VariantScore:
    """Roll per-case scores up into a :class:`VariantScore`."""
    medium = Severity.MEDIUM.rank
    s = VariantScore(variant=variant)
    for sc in scored:
        s.n_cases += 1
        s.planted_total += len(sc.caught)
        s.caught += sum(sc.caught)
        for t, c, how in zip(sc.planted_types, sc.caught, sc.caught_by, strict=True):
            s.by_type[t][1] += 1
            s.by_type[t][0] += int(c)
            s.caught_span += how == "span"
            s.caught_judge += how == "judge"
        s.findings_total += len(sc.findings)
        s.findings_matched += len(sc.findings) - len(sc.false_flags)
        s.false_flags += len(sc.false_flags)
        s.false_flags_ge_medium += sum(f.severity.rank >= medium for f in sc.false_flags)
        s.restatements += len(sc.restatements)
        if sc.is_clean:
            s.clean_cases += 1
            s.clean_flagged += bool(sc.findings)
            s.clean_flagged_ge_medium += any(f.severity.rank >= medium for f in sc.findings)
            s.clean_passed += sc.passed is True
        else:
            s.flawed_cases += 1
            s.flawed_failed += sc.passed is False
        s.verdict_correct += sc.verdict_correct
        s.no_verdict += sc.passed is None
        s.invalid_quotes += sc.invalid_quotes
        s.costs.append(sc.cost_usd)
        s.latencies.append(sc.latency_s)
    return s


# --------------------------------------------------------------------------- comparisons


@dataclass
class PairedComparison:
    """A vs B on the same planted issues (recall) and the same cases (verdicts)."""

    a: str
    b: str
    recall_a_only: int
    recall_b_only: int
    recall_both: int
    recall_neither: int
    verdict_a_only: int
    verdict_b_only: int

    @property
    def recall_p(self) -> float:
        """Exact McNemar p-value on planted-issue catches."""
        return mcnemar_exact(self.recall_a_only, self.recall_b_only)

    @property
    def verdict_p(self) -> float:
        """Exact McNemar p-value on correct verdicts."""
        return mcnemar_exact(self.verdict_a_only, self.verdict_b_only)


def compare(a: str, b: str, by_variant: dict[str, dict[str, ScoredCase]]) -> PairedComparison:
    """Paired comparison of variants ``a`` and ``b`` on the cases both have."""
    cases = sorted(set(by_variant[a]) & set(by_variant[b]))
    counts = Counter()  # type: Counter[str]
    for cid in cases:
        sa, sb = by_variant[a][cid], by_variant[b][cid]
        for ca, cb in zip(sa.caught, sb.caught, strict=True):
            counts["both" if ca and cb else "a" if ca else "b" if cb else "neither"] += 1
        if sa.verdict_correct != sb.verdict_correct:
            counts["va" if sa.verdict_correct else "vb"] += 1
    return PairedComparison(
        a=a,
        b=b,
        recall_a_only=counts["a"],
        recall_b_only=counts["b"],
        recall_both=counts["both"],
        recall_neither=counts["neither"],
        verdict_a_only=counts["va"],
        verdict_b_only=counts["vb"],
    )


# --------------------------------------------------------------------------- critics


@dataclass
class CriticStats:
    """How one critic performed, alone and inside the system."""

    critic: str
    model_id: str = ""
    cases_run: int = 0
    cases_failed: int = 0
    valid_issues: int = 0
    invalid_quotes: int = 0
    true_positives: int = 0
    planted_caught: int = 0
    unique_catches: int = 0
    adjudicated: int = 0
    overruled: int = 0
    overruled_false_flags: int = 0  # dismissed issues that really were false flags

    @property
    def hallucination_rate(self) -> float:
        """Share of this critic's issues whose quote wasn't in the text."""
        total = self.valid_issues + self.invalid_quotes
        return self.invalid_quotes / total if total else 0.0

    @property
    def precision(self) -> float:
        """Valid issues that caught a planted issue / valid issues."""
        return self.true_positives / self.valid_issues if self.valid_issues else 0.0

    @property
    def overrule_rate(self) -> float:
        """Share of its adjudicated issues the adjudicator dismissed."""
        return self.overruled / self.adjudicated if self.adjudicated else 0.0


@dataclass
class AdjudicatorStats:
    """What the adjudicator changed relative to the raw union of critic issues."""

    cases_adjudicated: int = 0
    cases_short_circuited: int = 0
    short_circuit_correct: int = 0  # short-circuited AND the case really was clean
    cases_fallback: int = 0
    groups_confirmed: int = 0
    groups_dismissed: int = 0
    dismissed_false_flags: int = 0  # correct dismissals
    dismissed_true_catches: int = 0  # wrong dismissals: lost a real catch
    verdicts_fixed: int = 0  # union wrong → system right
    verdicts_broken: int = 0  # union right → system wrong


@dataclass
class Agreement:
    """How often critics agree."""

    pairwise_verdict: dict[tuple[str, str], tuple[int, int]]  # (agree, total)
    all_agree: tuple[int, int]
    corroborated_groups: tuple[int, int]


def critic_analytics(
    runs: Sequence[CaseRun],
    critics: Sequence[str],
    by_variant: dict[str, dict[str, ScoredCase]],
) -> tuple[list[CriticStats], AdjudicatorStats, Agreement]:
    """Per-critic stats, adjudicator effect, and agreement between critics."""
    stats = {c: CriticStats(critic=c) for c in critics}
    adj = AdjudicatorStats()
    pair_counts: dict[tuple[str, str], list[int]] = {p: [0, 0] for p in combinations(critics, 2)}
    all_agree = [0, 0]
    groups_total = groups_corroborated = 0

    for run in runs:
        r = run.system
        if r is None:
            continue
        cid = run.case_id
        union_sc = by_variant[UNION].get(cid)
        union_matched = set(union_sc.matched_finding_ids) if union_sc else set()

        # critic-level
        caught_sets: dict[str, set[int]] = {}
        for name in critics:
            sc = by_variant[critic_variant(name)].get(cid)
            crit = next((c for c in r.critiques if c.critic == name), None)
            st = stats[name]
            if crit is None:
                st.cases_failed += 1
                continue
            st.model_id = crit.model_id
            st.cases_run += 1
            st.valid_issues += len(crit.valid_issues)
            st.invalid_quotes += len(crit.invalid_issues)
            if sc is not None:
                st.true_positives += len(sc.findings) - len(sc.false_flags)
                caught_sets[name] = {i for i, c in enumerate(sc.caught) if c}
                st.planted_caught += len(caught_sets[name])
        for name, caught in caught_sets.items():
            others = set().union(*(v for k, v in caught_sets.items() if k != name))
            stats[name].unique_catches += len(caught - others)

        # adjudicator-level
        a = r.adjudication
        v = r.verdict
        sys_sc = by_variant[SYSTEM].get(cid)
        if v.short_circuited:
            adj.cases_short_circuited += 1
            adj.short_circuit_correct += bool(sys_sc and sys_sc.is_clean)
        elif a is not None:
            if a.fallback_reason:
                adj.cases_fallback += 1
            else:
                adj.cases_adjudicated += 1
                adj.groups_confirmed += len(a.confirmed)
                adj.groups_dismissed += len(a.dismissed)
                dismissed_ids = {i for d in a.dismissed for i in d.issue_ids}
                for g in r.report.groups:
                    for issue_id in g.issue_ids:
                        name = issue_id.rsplit("-", 1)[0]
                        if name in stats:
                            stats[name].adjudicated += 1
                    if not set(g.issue_ids) & dismissed_ids:
                        continue
                    real = f"union-{g.id}" in union_matched
                    adj.dismissed_true_catches += real
                    adj.dismissed_false_flags += not real
                    for issue_id in g.issue_ids:
                        name = issue_id.rsplit("-", 1)[0]
                        if name in stats:
                            stats[name].overruled += 1
                            stats[name].overruled_false_flags += not real
        if union_sc and sys_sc:
            adj.verdicts_fixed += sys_sc.verdict_correct and not union_sc.verdict_correct
            adj.verdicts_broken += union_sc.verdict_correct and not sys_sc.verdict_correct

        # agreement on pass/fail between critics that both ran
        verdicts = {
            n: sc.passed
            for n in critics
            if (sc := by_variant[critic_variant(n)].get(cid)) is not None and sc.passed is not None
        }
        for pair, cnt in pair_counts.items():
            if pair[0] in verdicts and pair[1] in verdicts:
                cnt[1] += 1
                cnt[0] += verdicts[pair[0]] == verdicts[pair[1]]
        if len(verdicts) == len(critics):
            all_agree[1] += 1
            all_agree[0] += len(set(verdicts.values())) == 1
        groups_total += len(r.report.groups)
        groups_corroborated += sum(g.corroborated for g in r.report.groups)

    agreement = Agreement(
        pairwise_verdict={p: (c[0], c[1]) for p, c in pair_counts.items()},
        all_agree=(all_agree[0], all_agree[1]),
        corroborated_groups=(groups_corroborated, groups_total),
    )
    return list(stats.values()), adj, agreement
