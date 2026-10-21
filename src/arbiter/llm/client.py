"""Provider-agnostic structured LLM calls over OpenAI-compatible endpoints.

Every provider (DeepSeek, Gemini, Ollama) is reached with the official ``openai`` SDK by
pointing ``base_url`` at the provider. This module adds the three things the SDK alone
does not give us:

1. **Structured output** — ask for JSON, validate it against a Pydantic model, and re-ask
   with the validation error if it does not fit (at most ``max_reasks`` times).
2. **One retry layer** — transient failures (429 per-minute limits, 5xx, timeouts, empty
   responses) are retried with exponential backoff that honours ``Retry-After``; errors that
   retrying cannot fix (bad key, no balance, daily quota, unknown model) fail fast with a
   message that says what to do. The SDK's own retries are disabled to avoid retry-on-retry.
3. **Guard-rails + telemetry** — a process-wide semaphore caps in-flight calls, every call
   is capped at ``max_output_tokens``, and every attempt is logged as a ``CallRecord``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ValidationError

from arbiter.config import AppConfig
from arbiter.llm.telemetry import CallLog, CallRecord, compute_cost, extract_usage

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]


class LLMError(RuntimeError):
    """Base class for every error this module surfaces to callers.

    ``calls`` holds telemetry for the attempts made before failing, so a failed critic's
    spend is still counted.
    """

    calls: list[CallRecord]

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self.calls = []


class ConfigurationError(LLMError):
    """Local setup problem, e.g. a missing API key."""


class FatalAPIError(LLMError):
    """The provider refused in a way retrying cannot fix."""


class RetriesExhaustedError(LLMError):
    """A transient error persisted through every retry."""


class StructuredOutputError(LLMError):
    """The model never produced JSON matching the schema (or was truncated)."""


class EmptyResponseError(Exception):
    """Provider returned no content (a documented DeepSeek JSON-mode quirk). Retryable."""


@dataclass
class StructuredResult[T: BaseModel]:
    """A validated model output plus telemetry for every attempt that produced it."""

    value: T
    calls: list[CallRecord] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        """Total spend across attempts."""
        return sum(c.cost_usd for c in self.calls)


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s")


def strip_code_fences(text: str) -> str:
    """Remove a surrounding ```json fence that some models add despite instructions."""
    match = _FENCE_RE.match(text)
    return match.group(1) if match else text.strip()


def schema_instructions(schema: type[BaseModel]) -> str:
    """Prompt suffix describing the required JSON shape (also satisfies DeepSeek's rule
    that JSON-mode prompts must contain the word 'json')."""
    return (
        "Respond with ONLY a single JSON object — no markdown fences, no commentary — that "
        "conforms to this JSON schema:\n" + json.dumps(schema.model_json_schema())
    )


def _summarize_validation_error(exc: ValidationError, limit: int = 5) -> str:
    parts = [
        f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
        for err in exc.errors()[:limit]
    ]
    return "; ".join(parts)


def _error_blob(exc: openai.APIStatusError) -> str:
    return f"{exc.message} {json.dumps(exc.body, default=str) if exc.body else ''}"


def is_daily_quota(exc: openai.APIStatusError) -> bool:
    """Gemini free-tier daily quota 429s name a per-day quota id; waiting minutes won't help."""
    blob = _error_blob(exc).lower()
    return "perday" in blob or "per day" in blob or "daily" in blob


def retry_after_seconds(exc: BaseException) -> float | None:
    """Server-suggested wait: ``Retry-After`` header, or Gemini's ``retryDelay`` in the body."""
    if isinstance(exc, openai.APIStatusError):
        header = exc.response.headers.get("retry-after")
        if header:
            try:
                return max(0.0, float(header))
            except ValueError:
                pass
        if match := _RETRY_DELAY_RE.search(_error_blob(exc)):
            return float(match.group(1))
    return None


def classify(exc: BaseException) -> Literal["transient", "fatal"]:
    """Decide whether an exception is worth retrying."""
    if isinstance(exc, EmptyResponseError | openai.APITimeoutError | openai.APIConnectionError):
        return "transient"
    if isinstance(exc, openai.RateLimitError):
        return "fatal" if is_daily_quota(exc) else "transient"
    if isinstance(exc, openai.APIStatusError):
        return "transient" if exc.status_code >= 500 or exc.status_code == 408 else "fatal"
    return "fatal"


class LLMClient:
    """Structured, rate-limited, retried, logged LLM calls for every configured provider."""

    def __init__(
        self,
        config: AppConfig,
        *,
        call_log: CallLog | None = None,
        clients: Mapping[str, AsyncOpenAI] | None = None,
        max_reasks: int = 2,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        """
        Args:
            config: Validated app config.
            call_log: Where call records go; defaults to an in-memory log.
            clients: Pre-built clients keyed by provider name (tests inject fakes here).
            max_reasks: Schema-repair attempts after the first invalid JSON reply.
            sleep: Awaitable sleep used for backoff (tests inject a no-op).
        """
        self.config = config
        self.call_log = call_log or CallLog()
        self._clients: dict[str, AsyncOpenAI] = dict(clients or {})
        self._max_reasks = max_reasks
        self._sleep = sleep
        self._semaphore = asyncio.Semaphore(config.limits.max_concurrency)

    # ------------------------------------------------------------------ public API

    async def structured[T: BaseModel](
        self,
        *,
        role: str,
        model_key: str,
        system: str,
        user: str,
        schema: type[T],
        max_output_tokens: int | None = None,
    ) -> StructuredResult[T]:
        """Call ``model_key`` and return its reply parsed as ``schema``.

        Raises:
            ConfigurationError: missing API key.
            FatalAPIError: non-retryable provider error.
            RetriesExhaustedError: transient errors outlasted ``max_retries``.
            StructuredOutputError: invalid JSON after all re-asks, or output truncated.

        Every raised ``LLMError`` carries ``.calls`` (the attempts made before failing).
        """
        calls: list[CallRecord] = []
        try:
            return await self._structured(
                role=role,
                model_key=model_key,
                system=system,
                user=user,
                schema=schema,
                max_tokens=max_output_tokens or self.config.limits.max_output_tokens,
                calls=calls,
            )
        except LLMError as exc:
            exc.calls = calls
            raise

    # ------------------------------------------------------------------ internals

    async def _structured[T: BaseModel](
        self,
        *,
        role: str,
        model_key: str,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int,
        calls: list[CallRecord],
    ) -> StructuredResult[T]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": f"{system}\n\n{schema_instructions(schema)}"},
            {"role": "user", "content": user},
        ]
        last_error = ""
        for reask in range(self._max_reasks + 1):
            completion = await self._complete_with_retries(
                role=role,
                model_key=model_key,
                messages=messages,
                schema=schema,
                max_tokens=max_tokens,
                reask=reask,
                calls=calls,
            )
            choice = completion.choices[0]
            content = choice.message.content or ""
            if choice.finish_reason == "length":
                raise StructuredOutputError(
                    f"{role} ({model_key}): reply was cut off at max_output_tokens={max_tokens}. "
                    "Raise limits.max_output_tokens in config.yaml."
                )
            try:
                value = schema.model_validate_json(strip_code_fences(content))
            except ValidationError as exc:
                last_error = _summarize_validation_error(exc)
                logger.warning("%s: invalid JSON (re-ask %d): %s", role, reask, last_error)
                messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": "Your reply did not match the required JSON schema: "
                        f"{last_error}. Reply again with ONLY the corrected JSON object.",
                    },
                ]
                continue
            return StructuredResult(value=value, calls=calls)
        raise StructuredOutputError(
            f"{role} ({model_key}): no schema-valid JSON after {self._max_reasks} re-asks. "
            f"Last error: {last_error}"
        )

    def _client_for(self, provider_name: str) -> AsyncOpenAI:
        if provider_name in self._clients:
            return self._clients[provider_name]
        provider = self.config.providers[provider_name]
        api_key = provider.api_key()
        if api_key is None:
            if provider.requires_api_key:
                raise ConfigurationError(
                    f"{provider.api_key_env} is not set. Add it to .env (see .env.example)."
                )
            api_key = "not-needed"
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=provider.base_url,
            max_retries=0,  # retries live in this module, not the SDK
            timeout=self.config.limits.request_timeout_s,
        )
        self._clients[provider_name] = client
        return client

    def _backoff_delay(self, attempt: int, exc: BaseException) -> float:
        limits = self.config.limits
        suggested = retry_after_seconds(exc)
        if suggested is not None:
            return min(suggested, limits.backoff_max_s)
        base = min(limits.backoff_max_s, limits.backoff_initial_s * 2.0 ** (attempt - 1))
        return base * random.uniform(0.5, 1.0)  # jitter: parallel callers don't retry in sync

    def _fatal_message(self, exc: BaseException, model_key: str) -> str:
        model = self.config.model(model_key)
        provider = self.config.provider_for(model_key)
        if isinstance(exc, openai.RateLimitError):
            return (
                f"{model.provider} daily quota exhausted for '{model.model_id}'. Retrying today "
                "won't help: wait for the quota reset or point this role at another model "
                "in config.yaml."
            )
        # Gemini reports a bad key as HTTP 400 "Please pass a valid API key", not 401.
        bad_key = isinstance(exc, openai.BadRequestError) and "api key" in exc.message.lower()
        if bad_key or isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
            return f"{model.provider} rejected the API key. Check {provider.api_key_env} in .env."
        if isinstance(exc, openai.APIStatusError) and exc.status_code == 402:
            return f"{model.provider}: insufficient account balance."
        if isinstance(exc, openai.NotFoundError):
            return f"{model.provider}: model '{model.model_id}' not found. Check config.yaml."
        if isinstance(exc, openai.APIStatusError):
            return f"{model.provider} HTTP {exc.status_code}: {exc.message}"
        return f"{type(exc).__name__}: {exc}"

    async def _complete_with_retries(
        self,
        *,
        role: str,
        model_key: str,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        max_tokens: int,
        reask: int,
        calls: list[CallRecord],
    ) -> ChatCompletion:
        max_attempts = self.config.limits.max_retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                return await self._attempt(
                    role=role,
                    model_key=model_key,
                    messages=messages,
                    schema=schema,
                    max_tokens=max_tokens,
                    attempt=attempt,
                    reask=reask,
                    calls=calls,
                )
            except Exception as exc:
                if classify(exc) == "fatal":
                    if isinstance(exc, openai.APIError):
                        raise FatalAPIError(self._fatal_message(exc, model_key)) from exc
                    raise
                if attempt == max_attempts:
                    raise RetriesExhaustedError(
                        f"{role} ({model_key}): still failing after {attempt} attempts "
                        f"({type(exc).__name__}). If this is a free-tier rate limit, lower "
                        "limits.max_concurrency or use --limit."
                    ) from exc
                delay = self._backoff_delay(attempt, exc)
                logger.warning(
                    "%s (%s): %s on attempt %d/%d — retrying in %.1fs",
                    role,
                    model_key,
                    type(exc).__name__,
                    attempt,
                    max_attempts,
                    delay,
                )
                await self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _attempt(
        self,
        *,
        role: str,
        model_key: str,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        max_tokens: int,
        attempt: int,
        reask: int,
        calls: list[CallRecord],
    ) -> ChatCompletion:
        """One HTTP request, holding a concurrency slot only while the request is in flight."""
        model = self.config.model(model_key)
        provider = self.config.provider_for(model_key)
        client = self._client_for(model.provider)

        kwargs: dict[str, Any] = {
            "model": model.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "response_format": self._response_format(provider.structured_mode, schema),
        }
        if model.temperature is not None:
            kwargs["temperature"] = model.temperature
        if model.reasoning_effort is not None:
            kwargs["reasoning_effort"] = model.reasoning_effort
        if model.extra_body:
            kwargs["extra_body"] = dict(model.extra_body)

        record: dict[str, Any] = dict(
            call_id=uuid.uuid4().hex[:12],
            role=role,
            model_key=model_key,
            model_id=model.model_id,
            provider=model.provider,
            attempt=attempt,
            reask=reask,
            started_at=datetime.now(UTC),
        )
        async with self._semaphore:
            # timed INSIDE the semaphore: queueing behind other calls is not provider latency
            start = time.perf_counter()
            try:
                completion: ChatCompletion = await client.chat.completions.create(**kwargs)
                elapsed = time.perf_counter() - start
                choice = completion.choices[0] if completion.choices else None
                if choice is None or (
                    not (choice.message.content or "").strip() and choice.finish_reason != "length"
                ):
                    raise EmptyResponseError("provider returned empty content")
            except Exception as exc:
                self._log(
                    calls,
                    CallRecord(
                        **record,
                        latency_s=time.perf_counter() - start,
                        success=False,
                        error_type=type(exc).__name__,
                        error=str(exc)[:500],
                    ),
                )
                raise
        tokens = extract_usage(completion.usage)
        self._log(
            calls,
            CallRecord(
                **record,
                latency_s=elapsed,
                success=True,
                finish_reason=choice.finish_reason,
                input_tokens=tokens.input_tokens,
                cached_input_tokens=tokens.cached_input_tokens,
                output_tokens=tokens.output_tokens,
                reasoning_tokens=tokens.reasoning_tokens,
                cost_usd=compute_cost(tokens, model.pricing),
            ),
        )
        return completion

    @staticmethod
    def _response_format(mode: str, schema: type[BaseModel]) -> dict[str, Any]:
        if mode == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            }
        return {"type": "json_object"}

    def _log(self, calls: list[CallRecord], record: CallRecord) -> None:
        calls.append(record)
        self.call_log.add(record)
