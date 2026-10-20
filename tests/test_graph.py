"""End-to-end graph runs with mocked providers: fan-out/fan-in, failures, short-circuit."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from arbiter.adjudication.base import AdjudicationInput
from arbiter.adjudication.union import UnionAdjudicator
from arbiter.config import AppConfig, Dimension
from arbiter.graph.nodes import InputError
from arbiter.llm.client import LLMClient
from arbiter.pipeline import Arbiter
from arbiter.schemas.verdict import Adjudication
from tests.conftest import FakeOpenAI, api_error, make_completion, no_sleep
from tests.helpers import TEXT

QUESTION = "When did Apollo 11 land, who walked on the Moon, and for how long?"
MARKERS = {
    "Factual accuracy": Dimension.FACTUAL,
    "Logical consistency": Dimension.LOGIC,
    "Completeness": Dimension.COMPLETENESS,
}

FLAWED = {
    Dimension.FACTUAL: {
        "analysis": "Wrong year.",
        "issues": [{"quote": "July 20, 1971", "problem": "It was 1969.", "severity": "high"}],
        "verified_claims": [{"quote": "Neil Armstrong and Buzz Aldrin", "note": "correct crew"}],
        "score": 2,
        "confidence": 0.95,
    },
    Dimension.LOGIC: {
        "analysis": "Overgeneralisation.",
        "issues": [
            {
                "quote": "competition is the only driver of progress",
                "problem": "Does not follow from one event.",
                "severity": "medium",
            }
        ],
        "score": 3,
        "confidence": 0.9,
    },
    Dimension.COMPLETENESS: {
        "analysis": "Duration missing.",
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
CLEAN = {d: {"analysis": "Fine.", "issues": [], "score": 5, "confidence": 0.95} for d in Dimension}


def _dimension_of(request: dict[str, Any]) -> Dimension:
    system = request["messages"][0]["content"]
    return next(d for marker, d in MARKERS.items() if f"ONE dimension only: {marker}" in system)


class Providers:
    """Routes each request to a scripted reply based on which critic is asking."""

    def __init__(
        self,
        replies: dict[Dimension, dict[str, Any] | Exception],
        delay: float = 0.0,
    ) -> None:
        self.replies = replies
        self.delay = delay
        self.in_flight = self.peak = 0
        self.fakes = {
            "deepseek": FakeOpenAI(router=self._route),
            "gemini": FakeOpenAI(router=self._route),
        }

    async def _route(self, request: dict[str, Any]) -> Any:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            reply = self.replies[_dimension_of(request)]
            if isinstance(reply, Exception):
                raise reply
            return make_completion(reply)
        finally:
            self.in_flight -= 1

    def llm(self, config: AppConfig) -> LLMClient:
        return LLMClient(config, clients=self.fakes, sleep=no_sleep)  # type: ignore[arg-type]


class SpyAdjudicator(UnionAdjudicator):
    """Union adjudicator that records whether it was called."""

    def __init__(self) -> None:
        self.calls = 0

    async def adjudicate(self, inp: AdjudicationInput) -> Adjudication:
        self.calls += 1
        return await super().adjudicate(inp)


def _arbiter(config: AppConfig, providers: Providers) -> tuple[Arbiter, SpyAdjudicator]:
    spy = SpyAdjudicator()
    return Arbiter(config, llm=providers.llm(config), adjudicator=spy), spy


def test_graph_has_expected_nodes_and_edges(config: AppConfig) -> None:
    arbiter, _ = _arbiter(config, Providers(CLEAN))
    graph = arbiter.graph.get_graph()
    assert set(graph.nodes) >= {
        "parse_input",
        "run_critic",
        "collect_critiques",
        "detect_disagreements",
        "adjudicate",
        "synthesize_verdict",
    }
    conditional = {(e.source, e.target) for e in graph.edges if e.conditional}
    assert conditional == {
        ("parse_input", "run_critic"),
        ("detect_disagreements", "adjudicate"),
        ("detect_disagreements", "synthesize_verdict"),
    }


async def test_critics_run_in_parallel_and_results_fan_in(config: AppConfig) -> None:
    providers = Providers(FLAWED, delay=0.05)
    arbiter, spy = _arbiter(config, providers)
    result = await arbiter.arbitrate(TEXT, QUESTION)

    assert providers.peak == 3  # all three critics in flight at once
    assert [c.critic for c in result.critiques] == ["completeness", "factual", "logic"]
    assert result.failures == []
    assert spy.calls == 1
    v = result.verdict
    assert v.status == "ok" and v.passed is False and not v.short_circuited
    assert {c.quote for c in v.confirmed_issues} == {
        "July 20, 1971",
        "competition is the only driver of progress",
        "Neil Armstrong and Buzz Aldrin walked on the surface.",
    }
    assert len(result.calls) == 3 and result.cost_usd > 0
    assert result.id and result.question == QUESTION


async def test_failed_critic_degrades_but_does_not_abort(config: AppConfig) -> None:
    providers = Providers({**FLAWED, Dimension.LOGIC: api_error(401)})
    arbiter, spy = _arbiter(config, providers)
    result = await arbiter.arbitrate(TEXT, QUESTION)

    [failure] = result.failures
    assert failure.critic == "logic" and failure.error_type == "FatalAPIError"
    assert "GEMINI_API_KEY" in failure.message
    assert len(failure.calls) == 1  # the failed attempt is still accounted for
    assert {c.critic for c in result.critiques} == {"factual", "completeness"}
    v = result.verdict
    assert v.status == "degraded"
    assert Dimension.LOGIC in v.low_confidence_dimensions
    assert v.confidence < 0.67
    assert spy.calls == 1


async def test_transient_errors_are_retried_inside_a_branch(config: AppConfig) -> None:
    attempts = {"n": 0}
    providers = Providers(FLAWED)
    original = providers._route

    async def flaky(request: dict[str, Any]) -> Any:
        if _dimension_of(request) is Dimension.LOGIC and attempts["n"] < 2:
            attempts["n"] += 1
            raise api_error(503)
        return await original(request)

    providers.fakes["gemini"].router = flaky
    result = await _arbiter(config, providers)[0].arbitrate(TEXT, QUESTION)
    assert result.failures == []
    logic = next(c for c in result.critiques if c.critic == "logic")
    assert [c.success for c in logic.calls] == [False, False, True]


async def test_unanimous_confident_pass_short_circuits_adjudicator(config: AppConfig) -> None:
    arbiter, spy = _arbiter(config, Providers(CLEAN))
    result = await arbiter.arbitrate(TEXT, QUESTION)
    assert spy.calls == 0 and result.adjudication is None
    assert result.verdict.short_circuited and result.verdict.passed
    assert result.verdict.overall_score == 10


async def test_clean_but_unsure_goes_to_adjudicator(config: AppConfig) -> None:
    unsure = {**CLEAN, Dimension.LOGIC: {**CLEAN[Dimension.LOGIC], "confidence": 0.4}}
    arbiter, spy = _arbiter(config, Providers(unsure))
    result = await arbiter.arbitrate(TEXT, QUESTION)
    assert spy.calls == 1 and not result.verdict.short_circuited


async def test_all_critics_failing_yields_failed_verdict(config: AppConfig) -> None:
    providers = Providers({d: api_error(401) for d in Dimension})
    arbiter, spy = _arbiter(config, providers)
    result = await arbiter.arbitrate(TEXT, QUESTION)
    assert result.verdict.status == "failed" and result.verdict.passed is None
    assert spy.calls == 0 and len(result.failures) == 3


async def test_input_is_normalised(config: AppConfig) -> None:
    arbiter, _ = _arbiter(config, Providers(CLEAN))
    result = await arbiter.arbitrate(f"\n  {TEXT}  \n", "   ")
    assert result.output == TEXT and result.question is None


@pytest.mark.parametrize("output", ["", "   "])
async def test_empty_output_rejected_before_any_call(config: AppConfig, output: str) -> None:
    providers = Providers(CLEAN)
    with pytest.raises(InputError, match="empty"):
        await _arbiter(config, providers)[0].arbitrate(output)
    assert providers.peak == 0


async def test_oversized_input_rejected(config: AppConfig) -> None:
    providers = Providers(CLEAN)
    with pytest.raises(InputError, match="max_input_chars"):
        await _arbiter(config, providers)[0].arbitrate("x" * 20_001)
    assert providers.peak == 0
