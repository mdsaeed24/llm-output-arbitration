"""End-to-end evaluation on two mocked cases: run → persist → resume → score → report.

Scripted scenario (case 'apollo', 3 planted issues + case 'water', clean):
* critics flag all three apollo issues; the adjudicator wrongly DISMISSES the logic one
* the baseline catches date + logic but misses the omission, and false-flags the clean case
So by hand: system 2/3, baseline 2/3, union 3/3, adjudicator lost 1 real catch.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from arbiter.config import AppConfig, Dimension
from arbiter.evaluation.findings import BASELINE, SYSTEM, UNION, critic_variant
from arbiter.evaluation.matching import OmissionJudge
from arbiter.evaluation.report import render
from arbiter.evaluation.runner import load_runs, run_cases, score_runs, write_metrics_json
from arbiter.llm.client import LLMClient
from arbiter.schemas.testcase import EvalCase
from tests.conftest import FakeOpenAI, make_completion, no_sleep
from tests.helpers import TEXT

CLEAN_TEXT = "At sea level, pure water boils at 100 °C, which is 212 °F."

CASES = [
    EvalCase.model_validate(
        {
            "id": "apollo",
            "status": "ready",
            "question": "When did Apollo 11 land, who walked on the Moon, and for how long?",
            "output": TEXT,
            "planted": [
                {"type": "factual", "span": "July 20, 1971", "description": "1969"},
                {"type": "completeness", "missing": "How long they stayed on the surface."},
                {
                    "type": "logic",
                    "span": "competition is the only driver of progress",
                    "description": "overgeneralisation",
                },
            ],
        }
    ),
    EvalCase.model_validate(
        {"id": "water", "status": "ready", "question": "Boiling point?", "output": CLEAN_TEXT}
    ),
]

CLEAN_CRITIQUE = {"analysis": "ok", "issues": [], "score": 5, "confidence": 0.95}
APOLLO_CRITIQUES = {
    Dimension.FACTUAL: {
        "analysis": "a",
        "issues": [{"quote": "July 20, 1971", "problem": "It was 1969.", "severity": "high"}],
        "score": 2,
        "confidence": 0.9,
    },
    Dimension.LOGIC: {
        "analysis": "a",
        "issues": [
            {
                "quote": "competition is the only driver of progress",
                "problem": "One event can't prove a universal claim.",
                "severity": "medium",
            }
        ],
        "score": 3,
        "confidence": 0.9,
    },
    Dimension.COMPLETENESS: {
        "analysis": "a",
        "issues": [
            {
                "quote": "Neil Armstrong and Buzz Aldrin walked on the surface.",
                "problem": "Never says how long they stayed.",
                "severity": "high",
            }
        ],
        "score": 2,
        "confidence": 0.9,
    },
}
MARKERS = {
    "ONE dimension only: Factual accuracy": Dimension.FACTUAL,
    "ONE dimension only: Logical consistency": Dimension.LOGIC,
    "ONE dimension only: Completeness": Dimension.COMPLETENESS,
}


def _ruling(gid: str, decision: str, severity: str | None) -> dict[str, Any]:
    return {
        "group_id": gid,
        "reasoning": f"{decision} {gid}",
        "decision": decision,
        "severity": severity,
        "confidence": 0.9,
    }


class Scripted:
    """Routes every request (critic / adjudicator / baseline / judge) to a scripted reply."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.fakes = {
            "deepseek": FakeOpenAI(router=self.route),
            "gemini": FakeOpenAI(router=self.route),
        }

    def _count(self, kind: str) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + 1

    async def route(self, req: dict[str, Any]) -> Any:
        await asyncio.sleep(0)
        system, user = req["messages"][0]["content"], req["messages"][1]["content"]
        apollo = TEXT in user
        for marker, dim in MARKERS.items():
            if marker in system:
                self._count("critic")
                return make_completion(APOLLO_CRITIQUES[dim] if apollo else CLEAN_CRITIQUE)
        if "You are the ADJUDICATOR" in system:
            self._count("adjudicator")
            # groups by position: g1 date, g2 completeness anchor, g3 logic conclusion
            return make_completion(
                {
                    "rulings": [
                        _ruling("g1", "confirm", "high"),
                        _ruling("g2", "confirm", "high"),
                        _ruling("g3", "dismiss", None),  # wrong dismissal
                    ],
                    "overall_score": 4,
                    "confidence": 0.9,
                    "summary": "s",
                }
            )
        if "Review this output for factual errors" in system:
            self._count("baseline")
            issues = (
                [
                    {
                        "quote": "July 20, 1971",
                        "problem": "1969",
                        "severity": "high",
                        "type": "factual",
                    },
                    {
                        "quote": "competition is the only driver of progress",
                        "problem": "non-sequitur",
                        "severity": "medium",
                        "type": "logic",
                    },
                ]
                if apollo
                else [
                    {
                        "quote": "212 °F",
                        "problem": "Should be 210 °F.",
                        "severity": "low",
                        "type": "factual",
                    }
                ]
            )
            return make_completion(
                {
                    "analysis": "a",
                    "issues": issues,
                    "overall_score": 5,
                    "confidence": 0.8,
                    "summary": "s",
                }
            )
        if "grading automatic reviewers" in system:
            self._count("judge")
            return make_completion(
                {"decisions": [{"complaint_id": "c1", "reasoning": "same item", "match": "m1"}]}
            )
        raise AssertionError(f"unrouted request: {system[:60]}")


@pytest.fixture
def scripted() -> Scripted:
    return Scripted()


def _llm(config: AppConfig, scripted: Scripted) -> LLMClient:
    return LLMClient(config, clients=scripted.fakes, sleep=no_sleep)  # type: ignore[arg-type]


async def test_full_evaluation_pipeline(
    config: AppConfig, scripted: Scripted, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    lines: list[str] = []
    runs = await run_cases(
        config, CASES, run_dir, llm=_llm(config, scripted), progress=lines.append
    )

    # persisted per case, progress reported
    assert sorted(load_runs(run_dir)) == ["apollo", "water"]
    assert len(lines) == 2 and all(line.startswith("[") for line in lines)
    # clean case short-circuits: 3 critics × 2 cases, 1 adjudication, 2 baselines
    assert scripted.counts == {"critic": 6, "adjudicator": 1, "baseline": 2}

    judge = OmissionJudge(_llm(config, scripted), cache_path=run_dir / "judge_cache.json")
    results = await score_runs(config, CASES, runs, judge, run_dir=run_dir)
    s = results.scores

    assert (s[SYSTEM].caught, s[SYSTEM].planted_total) == (2, 3)
    assert s[SYSTEM].by_type["logic"] == [0, 1]  # dismissed by the adjudicator
    assert s[SYSTEM].caught_judge == 1  # the omission, via the judge
    assert (s[BASELINE].caught, s[BASELINE].by_type["completeness"]) == (2, [0, 1])
    assert s[BASELINE].false_flags == 1 and s[BASELINE].clean_flagged == 1
    assert s[SYSTEM].false_flags == 0
    assert s[UNION].caught == 3
    assert s[critic_variant("factual")].caught == 1
    # verdicts: apollo must fail, water must pass — both get both right
    assert s[SYSTEM].verdict_correct == 2 and s[BASELINE].verdict_correct == 2
    # adjudicator threw away one real catch; logic critic was the one overruled
    assert results.adjudicator.dismissed_true_catches == 1
    assert results.adjudicator.cases_short_circuited == 1
    assert results.adjudicator.short_circuit_correct == 1
    logic = next(c for c in results.critic_stats if c.critic == "logic")
    assert (logic.overruled, logic.adjudicated, logic.overruled_false_flags) == (1, 1, 0)
    # judge: one call, the match listed for spot-checking
    assert scripted.counts["judge"] == 1
    [jm] = results.judge_matches
    assert jm.case_id == "apollo" and set(jm.variants) >= {SYSTEM, UNION}

    report = render(results)
    assert "## Headline" in report and "caught 67% (2/3)" in report
    assert "the full system tied the single reviewer" in report
    assert "Judge matches — spot-check these" in report and "`apollo`" in report
    assert "dismissals that threw away a real catch: 1/1" in report
    write_metrics_json(results, tmp_path / "metrics.json")
    assert (tmp_path / "metrics.json").read_text().count('"recall"') == len(results.variants)


async def test_resume_skips_finished_cases(
    config: AppConfig, scripted: Scripted, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    await run_cases(config, CASES[:1], run_dir, llm=_llm(config, scripted), progress=lambda _: None)
    before = dict(scripted.counts)
    lines: list[str] = []
    runs = await run_cases(
        config, CASES, run_dir, llm=_llm(config, scripted), progress=lines.append
    )
    assert sorted(runs) == ["apollo", "water"]
    assert lines[0].startswith("resuming: 1 case(s) already done, 1 to go")
    assert scripted.counts["critic"] - before["critic"] == 3  # only the new case ran


async def test_rescoring_uses_judge_cache(
    config: AppConfig, scripted: Scripted, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    runs = await run_cases(
        config, CASES, run_dir, llm=_llm(config, scripted), progress=lambda _: None
    )
    for _ in range(2):
        judge = OmissionJudge(_llm(config, scripted), cache_path=run_dir / "judge_cache.json")
        await score_runs(config, CASES, runs, judge)
    assert scripted.counts["judge"] == 1


async def test_crashed_case_is_recorded_not_fatal(
    config: AppConfig, scripted: Scripted, tmp_path: Path
) -> None:
    broken = EvalCase.model_validate({"id": "huge", "status": "ready", "output": "x" * 30_000})
    runs = await run_cases(
        config, [broken, CASES[1]], tmp_path, llm=_llm(config, scripted), progress=lambda _: None
    )
    assert runs["huge"].system is None and "max_input_chars" in (runs["huge"].system_error or "")
    assert runs["huge"].baseline is None  # same input guard as the system
    assert runs["water"].system is not None
    judge = OmissionJudge(_llm(config, scripted))
    results = await score_runs(config, [broken, CASES[1]], runs, judge)
    assert "huge" not in results.scored[SYSTEM]
    assert "system error" in render(results)
