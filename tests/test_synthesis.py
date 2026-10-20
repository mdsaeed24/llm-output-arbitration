"""Verdict synthesis and the union adjudicator (both deterministic)."""

from __future__ import annotations

import pytest

from arbiter.adjudication.base import AdjudicationInput, deterministic_score
from arbiter.adjudication.synthesis import synthesize_verdict
from arbiter.adjudication.union import UnionAdjudicator
from arbiter.config import AppConfig, Dimension
from arbiter.graph.disagreement import detect_disagreements
from arbiter.schemas.critique import Severity
from tests.helpers import TEXT, critique, failure

DATE = "July 20, 1971"
CONCLUSION = "competition is the only driver of progress"


async def _verdict(config: AppConfig, critiques, failures=()):  # type: ignore[no-untyped-def]
    report = detect_disagreements(critiques, list(failures), config.disagreement)
    adjudication = (
        None
        if report.unanimous_pass or not critiques
        else await UnionAdjudicator().adjudicate(
            AdjudicationInput(question=None, output=TEXT, critiques=critiques, report=report)
        )
    )
    return synthesize_verdict(
        config=config,
        critiques=critiques,
        failures=list(failures),
        report=report,
        adjudication=adjudication,
    )


def test_deterministic_score() -> None:
    assert deterministic_score([]) == 10
    assert deterministic_score([Severity.HIGH]) == 6
    assert deterministic_score([Severity.MEDIUM, Severity.LOW]) == 7
    assert deterministic_score([Severity.HIGH] * 5) == 1


async def test_union_confirms_every_valid_issue_and_fails_on_medium(config: AppConfig) -> None:
    v = await _verdict(
        config,
        [
            critique("factual", issues=[(DATE, "high"), ("landed on Mars", "high")]),
            critique("logic", issues=[(CONCLUSION, "medium")]),
            critique("completeness"),
        ],
    )
    assert v.status == "ok" and v.passed is False and v.adjudicator == "union"
    assert [c.quote for c in v.confirmed_issues] == [DATE, CONCLUSION]
    assert v.overall_score == deterministic_score([Severity.HIGH, Severity.MEDIUM])
    [dismissed] = v.dismissed_flags
    assert dismissed.rule == "invalid_quote" and dismissed.quote == "landed on Mars"


async def test_low_only_issues_pass(config: AppConfig) -> None:
    v = await _verdict(
        config,
        [critique("factual", issues=[(DATE, "low")]), critique("logic"), critique("completeness")],
    )
    assert v.passed is True and v.overall_score == 9


async def test_short_circuit_verdict(config: AppConfig) -> None:
    v = await _verdict(config, [critique(d.value, confidence=0.95) for d in Dimension])
    assert v.short_circuited and v.passed and v.overall_score == 10
    assert v.adjudicator is None
    assert v.confidence == pytest.approx(0.95)


async def test_failed_critic_marks_dimension_low_confidence_and_caps_confidence(
    config: AppConfig,
) -> None:
    v = await _verdict(
        config,
        [critique("factual", issues=[(DATE, "high")]), critique("completeness")],
        failures=[failure("logic")],
    )
    assert v.status == "degraded"
    assert v.low_confidence_dimensions == [Dimension.LOGIC]
    assert v.confidence == pytest.approx(0.9 * 2 / 3)
    assert "logic (FatalAPIError: boom)" in v.summary


async def test_unsure_critic_marks_its_dimension_low_confidence(config: AppConfig) -> None:
    v = await _verdict(
        config,
        [critique("factual"), critique("logic", confidence=0.3), critique("completeness")],
    )
    assert v.low_confidence_dimensions == [Dimension.LOGIC]
    assert not v.short_circuited  # one unsure critic blocks the shortcut


async def test_all_failed(config: AppConfig) -> None:
    v = await _verdict(config, [], failures=[failure(d.value) for d in Dimension])
    assert v.status == "failed" and v.passed is None and v.overall_score is None
    assert v.confidence == 0.0 and v.low_confidence_dimensions == list(Dimension)
