"""Shared fixtures. Every LLM call in the test suite goes through ``FakeOpenAI`` — no network."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletion

from arbiter.config import AppConfig, load_config
from arbiter.llm.client import LLMClient

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_completion(
    content: str | dict[str, Any] | None,
    *,
    finish_reason: str = "stop",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    total_tokens: int | None = None,
    extra_usage: dict[str, Any] | None = None,
) -> ChatCompletion:
    """Build a real ``ChatCompletion`` object, as the SDK would return it."""
    text = json.dumps(content) if isinstance(content, dict) else content
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens or prompt_tokens + completion_tokens,
        **(extra_usage or {}),
    }
    return ChatCompletion.model_validate(
        {
            "id": "cmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": text},
                }
            ],
            "usage": usage,
        }
    )


def api_error(
    status: int, body: dict[str, Any] | None = None, headers: dict[str, str] | None = None
) -> openai.APIStatusError:
    """Build the SDK's typed exception for an HTTP status (429 → RateLimitError, …)."""
    request = httpx.Request("POST", "https://fake.test/v1/chat/completions")
    response = httpx.Response(status, request=request, headers=headers or {}, json=body or {})
    cls: type[openai.APIStatusError] = {
        400: openai.BadRequestError,
        401: openai.AuthenticationError,
        404: openai.NotFoundError,
        429: openai.RateLimitError,
        500: openai.InternalServerError,
        503: openai.InternalServerError,
    }.get(status, openai.APIStatusError)
    return cls(f"HTTP {status}", response=response, body=body)


Responder = ChatCompletion | Exception | Callable[[dict[str, Any]], ChatCompletion]
Router = Callable[[dict[str, Any]], Any]
"""Called for EVERY request (instead of the queue); may raise, return, or be async."""


class _Completions:
    def __init__(self, owner: FakeOpenAI) -> None:
        self._owner = owner

    async def create(self, **kwargs: Any) -> ChatCompletion:
        self._owner.requests.append(kwargs)
        if self._owner.router is not None:
            result = self._owner.router(kwargs)
            return await result if inspect.isawaitable(result) else result
        if not self._owner.queue:
            raise AssertionError("FakeOpenAI: no scripted response left")
        nxt = self._owner.queue.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        if callable(nxt):
            return nxt(kwargs)
        return nxt


class FakeOpenAI:
    """Stands in for ``AsyncOpenAI``: replays scripted responses and records requests."""

    def __init__(self, *responses: Responder, router: Router | None = None) -> None:
        self.router = router
        self.queue: list[Responder] = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.chat = type("Chat", (), {"completions": _Completions(self)})()


async def no_sleep(_: float) -> None:
    """Backoff sleep replacement so retry tests run instantly."""


@pytest.fixture
def config() -> AppConfig:
    return load_config(REPO_ROOT / "config.yaml", env_file=None)


@pytest.fixture
def make_llm(config: AppConfig) -> Callable[..., LLMClient]:
    """Factory: ``make_llm(deepseek=FakeOpenAI(...), gemini=FakeOpenAI(...))``."""

    def _make(**clients: FakeOpenAI) -> LLMClient:
        return LLMClient(config, clients=clients, sleep=no_sleep)  # type: ignore[arg-type]

    return _make
