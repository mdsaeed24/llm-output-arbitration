"""Single-reviewer baseline: same rules as the system, one call, typed issues."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from arbiter.baseline import SYSTEM_PROMPT, SingleReviewer
from arbiter.config import AppConfig, Dimension
from arbiter.critics.prompts import CHECKS, ISSUE_RULES
from arbiter.llm.client import LLMClient
from tests.conftest import FakeOpenAI, make_completion
from tests.helpers import TEXT

MakeLLM = Callable[..., LLMClient]
QUESTION = "When did Apollo 11 land, who walked on the Moon, and for how long?"


def _draft(*issues: dict[str, str], score: int = 4, confidence: float = 0.8) -> dict[str, Any]:
    return {
        "analysis": "Reviewed.",
        "issues": list(issues),
        "overall_score": score,
        "confidence": confidence,
        "summary": "Summary.",
    }


def _issue(quote: str, type_: str, severity: str = "high") -> dict[str, str]:
    return {"quote": quote, "problem": f"bad: {quote}", "severity": severity, "type": type_}


def test_prompt_uses_the_requested_instruction_and_shared_rules() -> None:
    assert "Review this output for factual errors, logical flaws, and missing" in SYSTEM_PROMPT
    assert ISSUE_RULES in SYSTEM_PROMPT
    for dim in Dimension:
        assert CHECKS[dim] in SYSTEM_PROMPT


async def test_review_builds_verdict_with_system_rules(
    config: AppConfig, make_llm: MakeLLM
) -> None:
    fake = FakeOpenAI(
        make_completion(
            _draft(
                _issue("July 20, 1971", "factual"),
                _issue("competition is the only driver of progress", "logic", "medium"),
                _issue("They stayed for three days.", "completeness"),  # not in text
            )
        )
    )
    result = await SingleReviewer(make_llm(deepseek=fake)).review(TEXT, QUESTION)

    assert result.model_id == "deepseek-v4-pro"
    assert [i.dimension for i in result.issues] == [
        Dimension.FACTUAL,
        Dimension.LOGIC,
        Dimension.COMPLETENESS,
    ]
    assert [i.valid for i in result.issues] == [True, True, False]
    v = result.verdict
    assert [c.quote for c in v.confirmed_issues] == [
        "July 20, 1971",
        "competition is the only driver of progress",
    ]
    assert v.confirmed_issues[0].raised_by == ["baseline"]
    assert [d.rule for d in v.dismissed_flags] == ["invalid_quote"]
    assert v.passed is False and v.overall_score == 4 and v.confidence == 0.8
    assert len(result.calls) == 1 and result.cost_usd > 0
    user = fake.requests[0]["messages"][1]["content"]
    assert QUESTION in user and TEXT in user


async def test_clean_review_passes(config: AppConfig, make_llm: MakeLLM) -> None:
    fake = FakeOpenAI(make_completion(_draft(score=10, confidence=0.9)))
    result = await SingleReviewer(make_llm(deepseek=fake)).review(TEXT)
    assert result.verdict.passed is True and result.verdict.confirmed_issues == []
    assert result.question is None


async def test_low_only_issues_pass_like_the_system(config: AppConfig, make_llm: MakeLLM) -> None:
    fake = FakeOpenAI(make_completion(_draft(_issue("July 20, 1971", "factual", "low"))))
    result = await SingleReviewer(make_llm(deepseek=fake)).review(TEXT)
    assert result.verdict.passed is True


async def test_missing_issue_type_triggers_reask(config: AppConfig, make_llm: MakeLLM) -> None:
    untyped = {"quote": "July 20, 1971", "problem": "x", "severity": "high"}
    fake = FakeOpenAI(
        make_completion(_draft(untyped)),
        make_completion(_draft(_issue("July 20, 1971", "factual"))),
    )
    result = await SingleReviewer(make_llm(deepseek=fake)).review(TEXT)
    assert len(result.calls) == 2 and result.issues[0].dimension is Dimension.FACTUAL
