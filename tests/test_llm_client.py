"""LLMClient: structured parsing, re-asks, retry policy, guard-rails, telemetry."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import openai
import pytest
from pydantic import BaseModel

from arbiter.config import AppConfig
from arbiter.llm.client import (
    ConfigurationError,
    FatalAPIError,
    LLMClient,
    RetriesExhaustedError,
    StructuredOutputError,
    strip_code_fences,
)
from tests.conftest import FakeOpenAI, api_error, make_completion

MakeLLM = Callable[..., LLMClient]


class Answer(BaseModel):
    verdict: str
    score: int


GOOD = {"verdict": "ok", "score": 4}


async def _ask(llm: LLMClient, model_key: str = "deepseek-v4-pro") -> Any:
    return await llm.structured(
        role="test", model_key=model_key, system="sys", user="usr", schema=Answer
    )


async def test_parses_valid_json_and_logs_cost(make_llm: MakeLLM) -> None:
    fake = FakeOpenAI(make_completion(GOOD, prompt_tokens=1000, completion_tokens=500))
    llm = make_llm(deepseek=fake)
    result = await _ask(llm)
    assert result.value == Answer(verdict="ok", score=4)
    [call] = result.calls
    assert call.success and call.finish_reason == "stop"
    assert call.input_tokens == 1000 and call.output_tokens == 500
    # deepseek-v4-pro: 1000 * 1.32/1M + 500 * 3.96/1M
    assert call.cost_usd == pytest.approx(0.00132 + 0.00198)
    assert llm.call_log.records == result.calls


async def test_request_carries_config_guardrails(make_llm: MakeLLM) -> None:
    fake = FakeOpenAI(make_completion(GOOD))
    await _ask(make_llm(deepseek=fake))
    req = fake.requests[0]
    assert req["model"] == "deepseek-v4-pro"
    assert req["max_tokens"] == 2048
    assert req["temperature"] == 0.0
    assert req["extra_body"] == {"thinking": {"type": "disabled"}}
    assert req["response_format"] == {"type": "json_object"}
    assert "json" in req["messages"][0]["content"].lower()  # DeepSeek JSON-mode requirement


async def test_gemini_uses_json_schema_and_reasoning_effort(make_llm: MakeLLM) -> None:
    fake = FakeOpenAI(make_completion(GOOD))
    await _ask(make_llm(gemini=fake), model_key="gemini-flash")
    req = fake.requests[0]
    assert req["response_format"]["type"] == "json_schema"
    assert req["reasoning_effort"] == "low"
    assert "temperature" not in req


async def test_code_fenced_json_is_accepted(make_llm: MakeLLM) -> None:
    fake = FakeOpenAI(make_completion('```json\n{"verdict": "ok", "score": 4}\n```'))
    assert (await _ask(make_llm(deepseek=fake))).value.score == 4


def test_strip_code_fences_leaves_plain_json() -> None:
    assert strip_code_fences(' {"a": 1} ') == '{"a": 1}'


async def test_schema_violation_triggers_reask_with_error(make_llm: MakeLLM) -> None:
    fake = FakeOpenAI(
        make_completion({"verdict": "ok", "score": "high"}),  # wrong type
        make_completion(GOOD),
    )
    result = await _ask(make_llm(deepseek=fake))
    assert result.value.score == 4
    assert [c.reask for c in result.calls] == [0, 1]
    repair = fake.requests[1]["messages"][-1]["content"]
    assert "score" in repair and "did not match" in repair


async def test_gives_up_after_max_reasks(make_llm: MakeLLM) -> None:
    fake = FakeOpenAI(*[make_completion({"nope": 1}) for _ in range(3)])
    with pytest.raises(StructuredOutputError, match="after 2 re-asks"):
        await _ask(make_llm(deepseek=fake))
    assert len(fake.requests) == 3


async def test_truncated_output_fails_with_actionable_message(make_llm: MakeLLM) -> None:
    fake = FakeOpenAI(make_completion('{"verdict": "o', finish_reason="length"))
    with pytest.raises(StructuredOutputError, match="max_output_tokens"):
        await _ask(make_llm(deepseek=fake))


async def test_rate_limit_is_retried_then_succeeds(make_llm: MakeLLM) -> None:
    fake = FakeOpenAI(api_error(429, headers={"retry-after": "1"}), make_completion(GOOD))
    sleeps: list[float] = []

    async def record_sleep(s: float) -> None:
        sleeps.append(s)

    llm = make_llm(deepseek=fake)
    llm._sleep = record_sleep
    result = await _ask(llm)
    assert result.value.score == 4
    assert [(c.attempt, c.success) for c in result.calls] == [(1, False), (2, True)]
    assert result.calls[0].error_type == "RateLimitError"
    assert sleeps == [1.0]  # Retry-After honoured


async def test_gemini_retry_delay_in_body_is_honoured(make_llm: MakeLLM) -> None:
    body = {"error": {"details": [{"retryDelay": "7s"}]}}
    fake = FakeOpenAI(api_error(429, body=body), make_completion(GOOD))
    sleeps: list[float] = []

    async def record_sleep(s: float) -> None:
        sleeps.append(s)

    llm = make_llm(gemini=fake)
    llm._sleep = record_sleep
    await _ask(llm, model_key="gemini-flash")
    assert sleeps == [7.0]


async def test_daily_quota_fails_fast_without_retry(make_llm: MakeLLM) -> None:
    body = {
        "error": {"details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}
    }
    fake = FakeOpenAI(api_error(429, body=body))
    with pytest.raises(FatalAPIError, match="daily quota"):
        await _ask(make_llm(gemini=fake), model_key="gemini-flash")
    assert len(fake.requests) == 1


async def test_server_errors_exhaust_retries(make_llm: MakeLLM, config: AppConfig) -> None:
    attempts = config.limits.max_retries + 1
    fake = FakeOpenAI(*[api_error(503) for _ in range(attempts)])
    with pytest.raises(RetriesExhaustedError, match=f"after {attempts} attempts"):
        await _ask(make_llm(deepseek=fake))
    assert len(fake.requests) == attempts


async def test_empty_content_is_retried(make_llm: MakeLLM) -> None:
    fake = FakeOpenAI(make_completion(""), make_completion(GOOD))
    result = await _ask(make_llm(deepseek=fake))
    assert result.calls[0].error_type == "EmptyResponseError"


async def test_auth_error_names_the_env_var(make_llm: MakeLLM) -> None:
    fake = FakeOpenAI(api_error(401))
    with pytest.raises(FatalAPIError, match="DEEPSEEK_API_KEY"):
        await _ask(make_llm(deepseek=fake))
    assert len(fake.requests) == 1


async def test_missing_api_key_is_a_configuration_error(
    config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="DEEPSEEK_API_KEY is not set"):
        await _ask(LLMClient(config))


async def test_unexpected_exception_propagates_unwrapped(make_llm: MakeLLM) -> None:
    fake = FakeOpenAI(KeyError("bug"))
    with pytest.raises(KeyError):
        await _ask(make_llm(deepseek=fake))


async def test_semaphore_caps_concurrent_requests(config: AppConfig) -> None:
    limit = config.limits.max_concurrency
    in_flight = peak = 0

    async def slow(_: dict[str, Any]) -> Any:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return make_completion(GOOD)

    class SlowFake(FakeOpenAI):
        def __init__(self) -> None:
            super().__init__()

            async def create(**kwargs: Any) -> Any:
                return await slow(kwargs)

            self.chat.completions.create = create  # type: ignore[attr-defined]

    llm = LLMClient(config, clients={"deepseek": SlowFake()})  # type: ignore[dict-item]
    await asyncio.gather(*[_ask(llm) for _ in range(limit * 3)])
    assert peak == limit


def test_connection_errors_are_transient() -> None:
    import httpx

    from arbiter.llm.client import classify

    err = openai.APIConnectionError(request=httpx.Request("POST", "https://x.test"))
    assert classify(err) == "transient"
    assert classify(api_error(400)) == "fatal"


async def test_gemini_bad_key_400_names_the_env_var(make_llm: MakeLLM) -> None:
    err = openai.BadRequestError(
        "Error code: 400 - Please pass a valid API key",
        response=api_error(400).response,
        body=None,
    )
    fake = FakeOpenAI(err)
    with pytest.raises(FatalAPIError, match="GEMINI_API_KEY"):
        await _ask(make_llm(gemini=fake), model_key="gemini-flash")
