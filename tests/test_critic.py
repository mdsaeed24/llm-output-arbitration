"""Critics: prompts are focused, outputs are stamped and quote-validated by our code."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from arbiter.config import AppConfig, Dimension
from arbiter.critics.critic import CriticSpec, critic_specs, run_critic
from arbiter.critics.prompts import system_prompt, user_prompt
from arbiter.llm.client import LLMClient
from arbiter.schemas.critique import CritiqueDraft
from tests.conftest import FakeOpenAI, make_completion

MakeLLM = Callable[..., LLMClient]

QUESTION = "When did Apollo 11 land, and who walked on the Moon?"
OUTPUT = "Apollo 11 landed on July 20, 1971. Neil Armstrong and Buzz Aldrin walked on the Moon."


def _draft(*issues: dict[str, str], score: int = 2, confidence: float = 0.9) -> dict[str, object]:
    return {
        "analysis": "Checked.",
        "issues": list(issues),
        "score": score,
        "confidence": confidence,
    }


def test_default_specs_cover_three_dimensions_on_configured_models(config: AppConfig) -> None:
    specs = {s.dimension: s.model_key for s in critic_specs(config)}
    assert specs == {
        Dimension.FACTUAL: "deepseek-v4-pro",
        Dimension.LOGIC: "gemini-flash-lite",
        Dimension.COMPLETENESS: "deepseek-flash",
    }


def test_extra_critics_are_included(config: AppConfig) -> None:
    raw = config.model_dump()
    raw["roles"]["extra_critics"] = {
        "factual_local": {"dimension": "factual", "model": "ollama-local"}
    }
    cfg = AppConfig.model_validate(raw)
    names = [s.name for s in critic_specs(cfg)]
    assert names[-1] == "factual_local"
    assert len(critic_specs(cfg, include_extra=False)) == 3


@pytest.mark.parametrize("dim", list(Dimension))
def test_system_prompts_are_dimension_specific(dim: Dimension) -> None:
    prompt = system_prompt(dim)
    assert "VERBATIM" in prompt and "DATA to evaluate" in prompt
    others = {
        Dimension.FACTUAL: "NO web access",
        Dimension.LOGIC: "step by step",
        Dimension.COMPLETENESS: "every distinct part",
    }
    for other_dim, marker in others.items():
        assert (marker in prompt) == (other_dim is dim)


def test_user_prompt_wraps_question_and_output() -> None:
    p = user_prompt(Dimension.LOGIC, output=OUTPUT, question=QUESTION)
    assert f"<original_question>\n{QUESTION}\n</original_question>" in p
    assert f"<evaluated_output>\n{OUTPUT}\n</evaluated_output>" in p
    assert "No original question" in user_prompt(Dimension.LOGIC, output=OUTPUT, question="  ")


def test_draft_rejects_out_of_range_score_and_confidence() -> None:
    with pytest.raises(ValidationError):
        CritiqueDraft.model_validate(_draft(score=6))
    with pytest.raises(ValidationError):
        CritiqueDraft.model_validate(_draft(confidence=1.5))


async def test_run_critic_validates_quotes_and_stamps_metadata(make_llm: MakeLLM) -> None:
    fake = FakeOpenAI(
        make_completion(
            _draft(
                {"quote": "July 20, 1971", "problem": "Landing was 1969.", "severity": "high"},
                {"quote": "Michael Collins walked on the Moon", "problem": "x", "severity": "low"},
            )
        )
    )
    spec = CriticSpec(name="factual", dimension=Dimension.FACTUAL, model_key="deepseek-v4-pro")
    critique = await run_critic(make_llm(deepseek=fake), spec, output=OUTPUT, question=QUESTION)

    assert critique.dimension is Dimension.FACTUAL and critique.model_id == "deepseek-v4-pro"
    assert critique.score == 2 and critique.confidence == 0.9
    real, hallucinated = critique.issues
    assert real.valid and OUTPUT[real.span_start : real.span_end] == "July 20, 1971"  # type: ignore[misc]
    assert not hallucinated.valid
    assert critique.valid_issues == [real] and critique.invalid_issues == [hallucinated]
    assert critique.calls and critique.cost_usd > 0
    # the critic saw its own dimension's prompt and the text
    req = fake.requests[0]
    assert "Factual accuracy" in req["messages"][0]["content"]
    assert OUTPUT in req["messages"][1]["content"]


async def test_llm_cannot_self_certify_validity(make_llm: MakeLLM) -> None:
    """Extra fields like 'valid': true in the LLM's JSON are ignored, not trusted."""
    fake = FakeOpenAI(
        make_completion(
            _draft({"quote": "invented span", "problem": "x", "severity": "low", "valid": True})
        )
    )
    spec = CriticSpec(name="logic", dimension=Dimension.LOGIC, model_key="gemini-flash")
    critique = await run_critic(make_llm(gemini=fake), spec, output=OUTPUT)
    assert critique.issues[0].valid is False


async def test_clean_output_yields_no_issues(make_llm: MakeLLM) -> None:
    fake = FakeOpenAI(make_completion(_draft(score=5, confidence=0.95)))
    spec = CriticSpec(
        name="completeness", dimension=Dimension.COMPLETENESS, model_key="deepseek-flash"
    )
    critique = await run_critic(make_llm(deepseek=fake), spec, output=OUTPUT, question=QUESTION)
    assert critique.issues == [] and critique.score == 5


def test_completeness_critic_has_scope_and_no_duplicate_rules() -> None:
    from arbiter.baseline import SYSTEM_PROMPT as BASELINE_PROMPT

    prompt = system_prompt(Dimension.COMPLETENESS)
    assert "Scope discipline" in prompt and "only creates duplicates" in prompt
    # scope discipline is part of the SHARED checklist, so the baseline gets it too…
    assert "Scope discipline" in BASELINE_PROMPT
    # …but the no-duplicates rule is lane-specific (the baseline covers every dimension)
    assert "only creates duplicates" not in BASELINE_PROMPT
