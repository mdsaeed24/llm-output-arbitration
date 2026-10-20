"""Per-call telemetry: tokens, cost, latency, finish_reason.

One :class:`CallRecord` is produced for every HTTP attempt — including failed attempts
and schema re-asks — so the cost numbers in the eval report are the real spend, not just
the cost of the attempts that happened to succeed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from arbiter.config import Pricing


class CallRecord(BaseModel):
    """Telemetry for a single LLM HTTP attempt."""

    call_id: str
    role: str = Field(description="Who made the call, e.g. 'critic:factual'.")
    model_key: str
    model_id: str
    provider: str
    attempt: int = Field(ge=1, description="1 = first try; >1 = transient-error retry.")
    reask: int = Field(ge=0, description="0 = original prompt; >0 = schema-repair re-ask.")
    started_at: datetime
    latency_s: float
    success: bool
    finish_reason: str | None = None
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = Field(0, description="Billable output, incl. hidden reasoning.")
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    error_type: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class TokenUsage:
    """Provider-neutral token counts extracted from an OpenAI-style ``usage`` object."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


def _get(obj: Any, name: str) -> int:
    value = getattr(obj, name, None) if obj is not None else None
    return int(value) if isinstance(value, int | float) else 0


def extract_usage(usage: Any) -> TokenUsage:
    """Normalise usage across providers.

    * Cached input: OpenAI-style ``prompt_tokens_details.cached_tokens`` or DeepSeek's
      ``prompt_cache_hit_tokens``.
    * Hidden reasoning: if ``total_tokens`` exceeds prompt + completion (Gemini reports
      thinking tokens this way) the difference is billed as output.
    """
    if usage is None:
        return TokenUsage()
    prompt = _get(usage, "prompt_tokens")
    completion = _get(usage, "completion_tokens")
    total = _get(usage, "total_tokens")
    cached = _get(getattr(usage, "prompt_tokens_details", None), "cached_tokens") or _get(
        usage, "prompt_cache_hit_tokens"
    )
    reported_reasoning = _get(getattr(usage, "completion_tokens_details", None), "reasoning_tokens")
    hidden = max(0, total - prompt - completion)
    return TokenUsage(
        input_tokens=prompt,
        cached_input_tokens=min(cached, prompt),
        output_tokens=completion + hidden,
        reasoning_tokens=max(reported_reasoning, hidden),
    )


def compute_cost(tokens: TokenUsage, pricing: Pricing) -> float:
    """USD cost of one call at the configured per-1M-token prices."""
    uncached = tokens.input_tokens - tokens.cached_input_tokens
    return (
        uncached * pricing.input
        + tokens.cached_input_tokens * pricing.cached_input
        + tokens.output_tokens * pricing.output
    ) / 1_000_000


class CallLog:
    """Collects call records in memory and (optionally) appends them to a JSONL file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.records: list[CallRecord] = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, record: CallRecord) -> None:
        """Store one record and persist it if a path is configured."""
        self.records.append(record)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.model_dump(mode="json")) + "\n")

    def total_cost(self) -> float:
        """Sum of cost across all recorded attempts."""
        return sum(r.cost_usd for r in self.records)
