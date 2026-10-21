"""LLM adjudicator: prompt content, ruling coverage, mapping, fallback, end-to-end."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from arbiter.adjudication import prompts
from arbiter.adjudication.base import AdjudicationInput
from arbiter.adjudication.llm import LLMAdjudicator
from arbiter.config import AppConfig, Dimension
from arbiter.graph.disagreement import detect_disagreements
from arbiter.llm.client import LLMClient, StructuredOutputError
from arbiter.pipeline import Arbiter
from arbiter.schemas.critique import Severity
from tests.conftest import FakeOpenAI, api_error, make_completion
from tests.helpers import TEXT, critique
from tests.test_graph import FLAWED, QUESTION, Providers

MakeLLM = Callable[..., LLMClient]

DATE = "July 20, 1971"
CONCLUSION = "competition is the only driver of progress"
WALKERS = "Neil Armstrong and Buzz Aldrin walked on the surface"


def _input(config: AppConfig, question: str | None = QUESTION) -> AdjudicationInput:
    critiques = [
        critique("factual", issues=[(DATE, "high")], verified=[WALKERS]),
        critique("logic", issues=[(CONCLUSION, "medium"), (WALKERS, "low")]),
        critique("completeness", issues=[(WALKERS + ".", "high")]),
    ]
    report = detect_disagreements(critiques, [], config.disagreement)
    return AdjudicationInput(question=question, output=TEXT, critiques=critiques, report=report)


def _ruling(gid: str, decision: str = "confirm", severity: str | None = "high") -> dict[str, Any]:
    return {
        "group_id": gid,
        "reasoning": f"Reasoning about {gid}.",
        "decision": decision,
        "severity": severity,
        "confidence": 0.8,
    }


def _draft(*rulings: dict[str, Any], score: int = 4) -> dict[str, Any]:
    return {"rulings": list(rulings), "overall_score": score, "confidence": 0.85, "summary": "S."}


def test_input_fixture_produces_expected_groups(config: AppConfig) -> None:
    inp = _input(config)
    assert [g.issue_ids for g in inp.report.groups] == [
        ["factual-1"],
        ["logic-2"],
        ["completeness-1"],
        ["logic-1"],
    ]


def test_system_prompt_has_dimension_procedures_and_no_new_issues_rule() -> None:
    p = prompts.SYSTEM_PROMPT
    assert "do NOT report new problems" in p
    assert "trace the reasoning step by step" in p
    assert "re-read the original question" in p
    assert "against your own knowledge" in p
    assert "DATA to evaluate" in p


def test_user_prompt_lists_every_group_and_disagreement_but_no_model_names(
    config: AppConfig,
) -> None:
    inp = _input(config)
    text = prompts.user_prompt(inp)
    for g in inp.report.groups:
        assert f"### {g.id}" in text
    for d in inp.report.disagreements:
        assert f"DISAGREEMENT {d.id} {d.kind.value}" in text
    assert "opposite_judgment" in text  # logic flagged WALKERS, factual verified it
    assert "anchor only; the problem is something MISSING" in text
    assert f"<original_question>\n{QUESTION}" in text
    assert f'verified as correct: "{WALKERS}"' in text
    assert "Rule on EVERY group: g1, g2, g3, g4" in text
    assert "deepseek" not in text and "gemini" not in text  # blind to models


def test_user_prompt_mentions_failed_dimensions(config: AppConfig) -> None:
    text = prompts.user_prompt(_input(config), missing=[Dimension.LOGIC])
    assert "NOT REVIEWED (critic failed): Logical consistency" in text


async def test_rulings_map_to_confirmed_and_dismissed(config: AppConfig, make_llm: MakeLLM) -> None:
    inp = _input(config)
    fake = FakeOpenAI(
        make_completion(
            _draft(
                _ruling("g1", severity="high"),
                _ruling("g2", "dismiss", None),
                _ruling("g3", severity="medium"),  # adjudicator lowers completeness severity
                _ruling("g4", severity="medium"),
            )
        )
    )
    adj = await LLMAdjudicator(make_llm(deepseek=fake)).adjudicate(inp)

    assert adj.adjudicator == "llm:deepseek-v4-pro" and adj.overall_score == 4
    assert [c.group_id for c in adj.confirmed] == ["g1", "g3", "g4"]
    g3 = adj.confirmed[1]
    assert g3.severity is Severity.MEDIUM and g3.raised_by == ["completeness"]
    assert g3.evidence == "Reasoning about g3." and g3.disagreement_ids
    assert adj.confirmed[0].quote == DATE
    [dismissed] = adj.dismissed
    assert dismissed.rule == "adjudicator" and dismissed.quote == WALKERS
    assert dismissed.disagreement_ids  # the opposite-judgment disagreement it resolved
    assert len(adj.calls) == 1

    req = fake.requests[0]
    assert req["model"] == "deepseek-v4-pro"
    assert req["max_tokens"] == config.limits.adjudicator_max_output_tokens


async def test_missing_ruling_triggers_reask_naming_the_group(
    config: AppConfig, make_llm: MakeLLM
) -> None:
    full = _draft(*[_ruling(g) for g in ("g1", "g2", "g3", "g4")])
    fake = FakeOpenAI(
        make_completion(_draft(_ruling("g1"), _ruling("g2"), _ruling("g3"))),
        make_completion(full),
    )
    adj = await LLMAdjudicator(make_llm(deepseek=fake)).adjudicate(_input(config))
    assert len(adj.confirmed) == 4 and len(adj.calls) == 2
    assert "missing rulings for ['g4']" in fake.requests[1]["messages"][-1]["content"]


@pytest.mark.parametrize(
    "bad",
    [
        _draft(*[_ruling(g) for g in ("g1", "g2", "g3", "g4", "g9")]),  # invented group
        _draft(*[_ruling(g) for g in ("g1", "g1", "g2", "g3", "g4")]),  # duplicate
        _draft(_ruling("g1", "confirm", None), *[_ruling(g) for g in ("g2", "g3", "g4")]),
    ],
)
async def test_invalid_rulings_are_rejected(
    config: AppConfig, make_llm: MakeLLM, bad: dict[str, Any]
) -> None:
    fake = FakeOpenAI(*[make_completion(bad) for _ in range(3)])
    with pytest.raises(StructuredOutputError):
        await LLMAdjudicator(make_llm(deepseek=fake)).adjudicate(_input(config))


async def test_dismiss_ignores_any_severity(config: AppConfig, make_llm: MakeLLM) -> None:
    rulings = [_ruling("g1", "dismiss", "high"), *[_ruling(g) for g in ("g2", "g3", "g4")]]
    fake = FakeOpenAI(make_completion(_draft(*rulings)))
    adj = await LLMAdjudicator(make_llm(deepseek=fake)).adjudicate(_input(config))
    assert [d.issue_ids for d in adj.dismissed] == [["factual-1"]]


async def test_end_to_end_with_llm_adjudicator(config: AppConfig) -> None:
    """Groups: g1 date (factual), g2 completeness anchor, g3 conclusion (logic)."""
    providers = Providers(
        FLAWED,
        adjudicator=_draft(
            _ruling("g1", severity="high"),
            _ruling("g2", "dismiss", None),
            _ruling("g3", severity="medium"),
            score=5,
        ),
    )
    arbiter = Arbiter(config, llm=providers.llm(config))  # default = LLM adjudicator
    result = await arbiter.arbitrate(TEXT, QUESTION)

    v = result.verdict
    assert v.adjudicator == "llm:deepseek-v4-pro" and v.status == "ok"
    assert v.overall_score == 5 and v.passed is False
    assert [c.quote for c in v.confirmed_issues] == [DATE, CONCLUSION]
    assert [d.raised_by for d in v.dismissed_flags] == [["completeness"]]
    assert len(result.calls) == 4  # 3 critics + 1 adjudicator
    assert "### g3" in providers.adjudicator_requests[0]["messages"][1]["content"]


async def test_adjudicator_failure_falls_back_to_union(config: AppConfig) -> None:
    providers = Providers(FLAWED, adjudicator=api_error(401))
    result = await Arbiter(config, llm=providers.llm(config)).arbitrate(TEXT, QUESTION)

    v = result.verdict
    assert v.status == "degraded" and v.adjudicator == "union-fallback"
    assert len(v.confirmed_issues) == 3  # union confirms everything
    assert "adjudicator failed" in v.summary
    assert result.adjudication is not None and result.adjudication.fallback_reason
    assert len(result.adjudication.calls) == 1  # failed attempt still costed
    union_conf = sum(c.confidence for c in result.critiques) / 3
    assert v.confidence == pytest.approx(union_conf * 0.5)


async def test_short_circuit_never_calls_llm_adjudicator(config: AppConfig) -> None:
    clean = {d: {"analysis": "ok", "issues": [], "score": 5, "confidence": 0.95} for d in Dimension}
    providers = Providers(clean)  # adjudicator=None → any adjudicator call fails the test
    result = await Arbiter(config, llm=providers.llm(config)).arbitrate(TEXT, QUESTION)
    assert result.verdict.short_circuited and providers.adjudicator_requests == []
