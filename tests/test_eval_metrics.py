"""Metric definitions and the small-sample statistics."""

from __future__ import annotations

import pytest

from arbiter.config import Dimension
from arbiter.evaluation.findings import Finding
from arbiter.evaluation.metrics import ScoredCase, aggregate, compare, mcnemar_exact, wilson
from arbiter.schemas.critique import Severity


def _f(fid: str, severity: str = "high") -> Finding:
    return Finding(
        id=fid,
        dimension=Dimension.FACTUAL,
        severity=Severity(severity),
        description="d",
        critics=["c"],
    )


def _sc(
    cid: str,
    *,
    types: list[str],
    caught_by: list[str | None],
    findings: list[Finding],
    matched: list[str],
    passed: bool | None,
    variant: str = "v",
) -> ScoredCase:
    return ScoredCase(
        case_id=cid,
        variant=variant,
        is_clean=not types,
        planted_types=types,
        caught=[c is not None for c in caught_by],
        caught_by=caught_by,
        findings=findings,
        matched_finding_ids=matched,
        matches=[],
        passed=passed,
        cost_usd=0.01,
        latency_s=2.0,
        invalid_quotes=0,
    )


def test_wilson_interval_known_values() -> None:
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.237, abs=0.001) and hi == pytest.approx(0.763, abs=0.001)
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.278, abs=0.001)
    assert wilson(0, 0) == (0.0, 0.0)


def test_mcnemar_exact() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(6, 0) == pytest.approx(2 / 64)
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 2) == pytest.approx(0.0386, abs=0.0005)


def test_aggregate_definitions() -> None:
    scored = [
        # flawed: 2 planted (factual caught by span, completeness caught by judge),
        # 3 findings of which 1 false flag (low severity), verdict fail → correct
        _sc(
            "a",
            types=["factual", "completeness"],
            caught_by=["span", "judge"],
            findings=[_f("1"), _f("2"), _f("3", "low")],
            matched=["1", "2"],
            passed=False,
        ),
        # flawed: logic missed, no findings, passed → wrong verdict
        _sc("b", types=["logic"], caught_by=[None], findings=[], matched=[], passed=True),
        # clean: one medium finding (false flag) but passed → correct verdict
        _sc("c", types=[], caught_by=[], findings=[_f("4", "medium")], matched=[], passed=True),
        # clean: everything failed → no verdict → counts as wrong
        _sc("d", types=[], caught_by=[], findings=[], matched=[], passed=None),
    ]
    s = aggregate("v", scored)
    assert (s.caught, s.planted_total, s.recall) == (2, 3, pytest.approx(2 / 3))
    assert s.by_type == {"factual": [1, 1], "logic": [0, 1], "completeness": [1, 1]}
    assert (s.caught_span, s.caught_judge) == (1, 1)
    assert (s.findings_total, s.findings_matched) == (4, 2)
    assert s.precision == pytest.approx(0.5)
    assert (s.false_flags, s.false_flags_ge_medium) == (2, 1)
    assert (s.clean_cases, s.clean_flagged, s.clean_flagged_ge_medium) == (2, 1, 1)
    assert (s.verdict_correct, s.no_verdict, s.accuracy) == (2, 1, 0.5)
    assert (s.clean_passed, s.flawed_failed, s.flawed_cases) == (1, 1, 2)
    assert s.mean_cost == pytest.approx(0.01) and s.median_latency == 2.0


def test_paired_comparison_counts_discordant_pairs() -> None:
    a = {
        "x": _sc(
            "x",
            types=["factual", "logic"],
            caught_by=["span", None],
            findings=[],
            matched=[],
            passed=False,
            variant="a",
        ),
        "y": _sc("y", types=[], caught_by=[], findings=[], matched=[], passed=True, variant="a"),
    }
    b = {
        "x": _sc(
            "x",
            types=["factual", "logic"],
            caught_by=[None, "span"],
            findings=[],
            matched=[],
            passed=False,
            variant="b",
        ),
        "y": _sc(
            "y", types=[], caught_by=[], findings=[_f("1")], matched=[], passed=False, variant="b"
        ),
    }
    c = compare("a", "b", {"a": a, "b": b})
    assert (c.recall_a_only, c.recall_b_only, c.recall_both, c.recall_neither) == (1, 1, 0, 0)
    assert (c.verdict_a_only, c.verdict_b_only) == (1, 0)
    assert c.recall_p == 1.0


def test_restatement_of_planted_error_is_tracked_separately() -> None:
    from arbiter.evaluation.findings import VariantCase
    from arbiter.evaluation.metrics import score_case
    from arbiter.schemas.testcase import EvalCase

    text = "The fee is 1.5% a year. Nothing else."
    case = EvalCase.model_validate(
        {
            "id": "c",
            "output": text,
            "planted": [{"type": "factual", "span": "1.5% a year", "description": "too high"}],
        }
    )
    restate = Finding(
        id="r",
        kind="omission",
        dimension=Dimension.COMPLETENESS,
        span_start=0,
        span_end=len("The fee is 1.5% a year."),
        severity=Severity.HIGH,
        description="The cost figure is wrong.",
        critics=["completeness"],
    )
    other = Finding(
        id="o",
        kind="omission",
        dimension=Dimension.COMPLETENESS,
        span_start=text.index("Nothing"),
        span_end=len(text),
        severity=Severity.LOW,
        description="x",
        critics=["completeness"],
    )
    variant = VariantCase(
        variant="v",
        case_id="c",
        findings=[restate, other],
        passed=False,
        cost_usd=0,
        latency_s=0,
    )
    sc = score_case(case, variant, [], min_overlap=0.5)
    assert [f.id for f in sc.false_flags] == ["r", "o"]  # strict rule: both are false flags
    assert [f.id for f in sc.restatements] == ["r"]  # only the anchored one is a restatement
    agg = aggregate("v", [sc])
    assert (agg.false_flags, agg.restatements, agg.false_flags_excl_restatements) == (2, 1, 1)
