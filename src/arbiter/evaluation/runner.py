"""Run the system and the baseline over the test set, persist raw results, then score.

Each case's raw outputs are written to ``<run_dir>/cases/<id>.json`` as soon as the case
finishes, so a run interrupted by a quota or network failure can be resumed, and scoring can be
re-run (``--from-run``) without spending anything — the judge cache lives in the run dir too.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from arbiter.baseline import SingleReviewer
from arbiter.config import AppConfig
from arbiter.critics.critic import critic_specs
from arbiter.evaluation.findings import (
    BASELINE,
    SYSTEM,
    UNION,
    CaseRun,
    all_variants,
    critic_variant,
)
from arbiter.evaluation.matching import OmissionJudge, match_variant, omission_complaints
from arbiter.evaluation.metrics import (
    AdjudicatorStats,
    Agreement,
    CriticStats,
    PairedComparison,
    ScoredCase,
    VariantScore,
    aggregate,
    compare,
    critic_analytics,
    score_case,
)
from arbiter.llm.client import LLMClient
from arbiter.llm.telemetry import CallLog, CallRecord
from arbiter.pipeline import Arbiter
from arbiter.schemas.testcase import EvalCase, PlantedOmission

Progress = Callable[[str], None]


def new_run_dir(root: Path = Path("reports/runs")) -> Path:
    """``reports/runs/<UTC timestamp>``."""
    return root / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _case_path(run_dir: Path, case_id: str) -> Path:
    return run_dir / "cases" / f"{case_id}.json"


def load_runs(run_dir: Path) -> dict[str, CaseRun]:
    """Every case already saved in ``run_dir``."""
    folder = run_dir / "cases"
    if not folder.exists():
        return {}
    runs = [CaseRun.model_validate_json(p.read_text("utf-8")) for p in folder.glob("*.json")]
    return {r.case_id: r for r in runs}


async def run_case(arbiter: Arbiter, reviewer: SingleReviewer, case: EvalCase) -> CaseRun:
    """System and baseline on one case, concurrently. Errors are recorded, not raised."""
    sys_res, base_res = await asyncio.gather(
        arbiter.arbitrate(case.output, case.question, request_id=case.id),
        reviewer.review(case.output, case.question),
        return_exceptions=True,
    )
    run = CaseRun(case_id=case.id)
    if isinstance(sys_res, BaseException):
        run.system_error = f"{type(sys_res).__name__}: {sys_res}"
    else:
        run.system = sys_res
    if isinstance(base_res, BaseException):
        run.baseline_error = f"{type(base_res).__name__}: {base_res}"
    else:
        run.baseline = base_res
    return run


def _describe(run: CaseRun) -> str:
    parts = []
    if run.system is not None:
        v = run.system.verdict
        parts.append(
            f"system {v.status} pass={v.passed} issues={len(v.confirmed_issues)} "
            f"${run.system.cost_usd:.4f}"
        )
    else:
        parts.append(f"system ERROR {run.system_error}")
    if run.baseline is not None:
        b = run.baseline.verdict
        parts.append(f"baseline pass={b.passed} issues={len(b.confirmed_issues)}")
    else:
        parts.append(f"baseline ERROR {run.baseline_error}")
    return " | ".join(parts)


async def run_cases(
    config: AppConfig,
    cases: Sequence[EvalCase],
    run_dir: Path,
    *,
    llm: LLMClient | None = None,
    progress: Progress = print,
) -> dict[str, CaseRun]:
    """Run every case not already saved in ``run_dir`` (so re-running resumes)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    llm = llm or LLMClient(config, call_log=CallLog(run_dir / "llm_calls.jsonl"))
    arbiter, reviewer = Arbiter(config, llm=llm), SingleReviewer(llm)
    done = load_runs(run_dir)
    todo = [c for c in cases if c.id not in done]
    if done:
        progress(f"resuming: {len(done)} case(s) already done, {len(todo)} to go")
    gate = asyncio.Semaphore(config.evaluation.case_concurrency)
    counter = {"n": len(cases) - len(todo)}

    async def one(case: EvalCase) -> None:
        async with gate:
            run = await run_case(arbiter, reviewer, case)
        path = _case_path(run_dir, case.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(run.model_dump_json(), encoding="utf-8")
        done[case.id] = run
        counter["n"] += 1
        progress(f"[{counter['n']}/{len(cases)}] {case.id}: {_describe(run)}")

    await asyncio.gather(*[one(c) for c in todo])
    return {c.id: done[c.id] for c in cases if c.id in done}


# --------------------------------------------------------------------------- scoring


@dataclass
class JudgeMatchRecord:
    """A completeness match made by the judge — listed in the report for spot-checking."""

    case_id: str
    missing: str
    complaint: str
    reasoning: str
    variants: list[str]


@dataclass
class EvalResults:
    """Everything the report needs."""

    config: AppConfig
    cases: list[EvalCase]
    runs: dict[str, CaseRun]
    critics: list[str]
    scored: dict[str, dict[str, ScoredCase]]
    scores: dict[str, VariantScore]
    comparisons: list[PairedComparison]
    critic_stats: list[CriticStats]
    adjudicator: AdjudicatorStats
    agreement: Agreement
    judge_matches: list[JudgeMatchRecord]
    judge_rejections: int
    judge_calls: list[CallRecord] = field(default_factory=list)
    run_dir: Path | None = None

    @property
    def variants(self) -> list[str]:
        """Report order: system, baseline, union, then each critic alone."""
        return [SYSTEM, BASELINE, UNION, *[critic_variant(c) for c in self.critics]]


async def score_runs(
    config: AppConfig,
    cases: Sequence[EvalCase],
    runs: dict[str, CaseRun],
    judge: OmissionJudge,
    *,
    run_dir: Path | None = None,
) -> EvalResults:
    """Match every variant on every case and aggregate."""
    critics = [s.name for s in critic_specs(config)]
    variants = [SYSTEM, BASELINE, UNION, *[critic_variant(c) for c in critics]]
    scored: dict[str, dict[str, ScoredCase]] = {v: {} for v in variants}
    judge_matches: dict[tuple[str, str], JudgeMatchRecord] = {}
    rejections = 0
    min_overlap = config.evaluation.span_min_overlap

    for case in cases:
        run = runs.get(case.id)
        if run is None:
            continue
        outcomes = all_variants(run, critics, config)
        judged = await judge.judge(case, omission_complaints(outcomes))
        rejections += sum(v.planted_index is None for v in judged.values())
        for outcome in outcomes:
            matches = match_variant(case, outcome, judged, min_overlap)
            scored[outcome.variant][case.id] = score_case(case, outcome, matches, min_overlap)
            for m in matches:
                if m.method != "judge":
                    continue
                f = next(f for f in outcome.findings if f.id == m.finding_id)
                planted = case.planted[m.planted_index]
                assert isinstance(planted, PlantedOmission)
                rec = judge_matches.setdefault(
                    (case.id, f.description),
                    JudgeMatchRecord(
                        case_id=case.id,
                        missing=planted.missing,
                        complaint=f.description,
                        reasoning=m.judge_reasoning or "",
                        variants=[],
                    ),
                )
                if outcome.variant not in rec.variants:
                    rec.variants.append(outcome.variant)

    scores = {v: aggregate(v, list(scored[v].values())) for v in variants}
    comparisons = [compare(SYSTEM, BASELINE, scored), compare(SYSTEM, UNION, scored)]
    critic_stats, adj, agreement = critic_analytics(list(runs.values()), critics, scored)
    return EvalResults(
        config=config,
        cases=list(cases),
        runs=runs,
        critics=critics,
        scored=scored,
        scores=scores,
        comparisons=comparisons,
        critic_stats=critic_stats,
        adjudicator=adj,
        agreement=agreement,
        judge_matches=list(judge_matches.values()),
        judge_rejections=rejections,
        judge_calls=list(judge.calls),
        run_dir=run_dir,
    )


def write_metrics_json(results: EvalResults, path: Path) -> None:
    """Machine-readable metrics (used by the Phase 6 analytics page)."""
    data = {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_cases": len(results.runs),
        "variants": {
            v: {
                "recall": s.recall,
                "precision": s.precision,
                "accuracy": s.accuracy,
                "by_type": {t: {"caught": c, "total": n} for t, (c, n) in s.by_type.items()},
                "caught": s.caught,
                "planted_total": s.planted_total,
                "caught_span": s.caught_span,
                "caught_judge": s.caught_judge,
                "findings_total": s.findings_total,
                "false_flags": s.false_flags,
                "false_flags_ge_medium": s.false_flags_ge_medium,
                "restatements": s.restatements,
                "false_flags_excl_restatements": s.false_flags_excl_restatements,
                "precision_excl_restatements": s.precision_excl_restatements,
                "false_flags_per_case": s.false_flags / s.n_cases if s.n_cases else 0,
                "clean_cases": s.clean_cases,
                "clean_flagged": s.clean_flagged,
                "clean_flagged_ge_medium": s.clean_flagged_ge_medium,
                "clean_passed": s.clean_passed,
                "flawed_failed": s.flawed_failed,
                "flawed_cases": s.flawed_cases,
                "no_verdict": s.no_verdict,
                "invalid_quotes": s.invalid_quotes,
                "mean_cost_usd": s.mean_cost,
                "mean_latency_s": s.mean_latency,
                "median_latency_s": s.median_latency,
            }
            for v, s in results.scores.items()
        },
        "comparisons": [
            {
                "a": c.a,
                "b": c.b,
                "recall_a_only": c.recall_a_only,
                "recall_b_only": c.recall_b_only,
                "recall_p": c.recall_p,
                "verdict_a_only": c.verdict_a_only,
                "verdict_b_only": c.verdict_b_only,
                "verdict_p": c.verdict_p,
            }
            for c in results.comparisons
        ],
        "critics": [
            {
                "critic": s.critic,
                "model_id": s.model_id,
                "valid_issues": s.valid_issues,
                "true_positives": s.true_positives,
                "precision": s.precision,
                "planted_caught": s.planted_caught,
                "unique_catches": s.unique_catches,
                "hallucination_rate": s.hallucination_rate,
                "adjudicated": s.adjudicated,
                "overruled": s.overruled,
                "overrule_rate": s.overrule_rate,
                "overruled_false_flags": s.overruled_false_flags,
                "cases_failed": s.cases_failed,
            }
            for s in results.critic_stats
        ],
        "adjudicator": vars(results.adjudicator),
        "judge": {
            "matches": len(results.judge_matches),
            "rejections": results.judge_rejections,
            "cost_usd": sum(c.cost_usd for c in results.judge_calls),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
