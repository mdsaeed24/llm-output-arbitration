"""Evaluation matching: span rule, judge rule (blind, cached), anchor isolation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from arbiter.config import Dimension
from arbiter.evaluation.findings import Finding, VariantCase
from arbiter.evaluation.matching import (
    OmissionJudge,
    match_variant,
    omission_complaints,
    span_matches,
)
from arbiter.llm.client import LLMClient
from arbiter.schemas.critique import Severity
from arbiter.schemas.testcase import EvalCase
from tests.conftest import FakeOpenAI, make_completion
from tests.helpers import TEXT

MakeLLM = Callable[..., LLMClient]

CASE = EvalCase.model_validate(
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
)


def _span_finding(fid: str, quote: str, dim: Dimension = Dimension.FACTUAL) -> Finding:
    start = TEXT.index(quote)
    return Finding(
        id=fid,
        kind="span",
        dimension=dim,
        span_start=start,
        span_end=start + len(quote),
        severity=Severity.HIGH,
        description=f"bad {quote}",
        critics=["x"],
    )


def _omission(fid: str, description: str, anchor: str) -> Finding:
    start = TEXT.index(anchor)
    return Finding(
        id=fid,
        kind="omission",
        dimension=Dimension.COMPLETENESS,
        span_start=start,
        span_end=start + len(anchor),
        severity=Severity.HIGH,
        description=description,
        critics=["completeness"],
    )


def _variant(*findings: Finding) -> VariantCase:
    return VariantCase(
        variant="system",
        case_id="apollo",
        findings=list(findings),
        passed=False,
        cost_usd=0,
        latency_s=0,
    )


def test_span_rule_uses_shorter_span_overlap() -> None:
    sentence = _span_finding("f1", "Apollo 11 landed on the Moon on July 20, 1971.")
    wrong_place = _span_finding("f2", "Neil Armstrong and Buzz Aldrin")
    [m] = span_matches(CASE, [sentence, wrong_place], 0.5)
    assert (m.finding_id, m.planted_index, m.method, m.overlap) == ("f1", 0, "span", 1.0)


def test_span_rule_is_type_agnostic_between_factual_and_logic() -> None:
    f = _span_finding("f1", "competition is the only driver", dim=Dimension.FACTUAL)
    [m] = span_matches(CASE, [f], 0.5)
    assert m.planted_index == 2


def test_omission_finding_never_matches_by_span() -> None:
    """Completeness critic anchored on the planted LOGIC sentence — must not 'catch' it."""
    f = _omission("f1", "Says nothing about duration.", "competition is the only driver")
    assert span_matches(CASE, [f], 0.5) == []
    assert match_variant(CASE, _variant(f), {}, 0.5) == []


def _judge_reply(*decisions: tuple[str, str | None]) -> dict[str, Any]:
    return {
        "decisions": [
            {"complaint_id": c, "reasoning": f"why {c}", "match": m} for c, m in decisions
        ]
    }


async def test_judge_maps_matches_and_is_blind(make_llm: MakeLLM, tmp_path: Path) -> None:
    fake = FakeOpenAI(make_completion(_judge_reply(("c1", "m1"), ("c2", None))))
    judge = OmissionJudge(make_llm(deepseek=fake), cache_path=tmp_path / "cache.json")
    out = await judge.judge(CASE, ["Never says how long they stayed.", "Could be more detailed."])
    assert out["Never says how long they stayed."].planted_index == 1  # index in case.planted
    assert out["Could be more detailed."].planted_index is None
    req = fake.requests[0]
    assert req["model"] == "deepseek-flash"
    user = req["messages"][1]["content"]
    assert "m1: How long they stayed on the surface." in user
    for word in ("system", "baseline", "critic", "union"):
        assert word not in user.lower()  # blind to who raised the complaint


async def test_judge_cache_avoids_repeat_calls(make_llm: MakeLLM, tmp_path: Path) -> None:
    cache = tmp_path / "cache.json"
    fake = FakeOpenAI(make_completion(_judge_reply(("c1", "m1"))))
    await OmissionJudge(make_llm(deepseek=fake), cache_path=cache).judge(CASE, ["A"])
    fresh = FakeOpenAI()  # any request would fail: nothing scripted
    again = await OmissionJudge(make_llm(deepseek=fresh), cache_path=cache).judge(CASE, ["A"])
    assert again["A"].planted_index == 1 and fresh.requests == []


async def test_judge_skips_call_without_omissions_or_complaints(make_llm: MakeLLM) -> None:
    fake = FakeOpenAI()
    judge = OmissionJudge(make_llm(deepseek=fake))
    no_omission = CASE.model_copy(update={"planted": CASE.planted[:1]})
    assert (await judge.judge(no_omission, ["x"]))["x"].planted_index is None
    assert await judge.judge(CASE, []) == {}
    assert fake.requests == []


async def test_judge_rejects_incomplete_or_invented_ids(make_llm: MakeLLM) -> None:
    fake = FakeOpenAI(
        make_completion(_judge_reply(("c1", "m9"))),  # unknown omission
        make_completion(_judge_reply()),  # missing decision
        make_completion(_judge_reply(("c1", "m1"))),
    )
    out = await OmissionJudge(make_llm(deepseek=fake)).judge(CASE, ["A"])
    assert out["A"].planted_index == 1 and len(fake.requests) == 3


async def test_match_variant_combines_both_rules(make_llm: MakeLLM) -> None:
    date = _span_finding("f1", "July 20, 1971")
    gap = _omission("f2", "Never says how long they stayed.", "Neil Armstrong")
    variant = _variant(date, gap)
    complaints = omission_complaints([variant])
    assert complaints == ["Never says how long they stayed."]
    fake = FakeOpenAI(make_completion(_judge_reply(("c1", "m1"))))
    judged = await OmissionJudge(make_llm(deepseek=fake)).judge(CASE, complaints)
    matches = match_variant(CASE, variant, judged, 0.5)
    assert {(m.finding_id, m.planted_index, m.method) for m in matches} == {
        ("f1", 0, "span"),
        ("f2", 1, "judge"),
    }


@pytest.mark.parametrize("ratio,expected", [(0.5, 1), (0.95, 0)])
def test_span_threshold_is_configurable(ratio: float, expected: int) -> None:
    # finding = "20, 1971. Neil" (14 chars); overlap with "July 20, 1971" (13) is "20, 1971"
    # = 8 chars → ratio 8/13 ≈ 0.62 of the shorter span
    start = TEXT.index("20, 1971. Neil")
    partial = Finding(
        id="p",
        dimension=Dimension.FACTUAL,
        span_start=start,
        span_end=start + len("20, 1971. Neil"),
        severity=Severity.LOW,
        description="x",
        critics=["x"],
    )
    assert len(span_matches(CASE, [partial], ratio)) == expected
