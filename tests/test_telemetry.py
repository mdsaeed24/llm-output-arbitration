"""Token normalisation and cost across provider usage formats."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from arbiter.config import Pricing
from arbiter.llm.telemetry import CallLog, CallRecord, TokenUsage, compute_cost, extract_usage

PRICING = Pricing(input=1.0, cached_input=0.1, output=4.0)


def test_openai_style_cached_tokens() -> None:
    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=200,
        total_tokens=1200,
        prompt_tokens_details=SimpleNamespace(cached_tokens=400),
    )
    t = extract_usage(usage)
    assert t == TokenUsage(input_tokens=1000, cached_input_tokens=400, output_tokens=200)


def test_deepseek_cache_hit_field() -> None:
    usage = SimpleNamespace(
        prompt_tokens=1000, completion_tokens=200, total_tokens=1200, prompt_cache_hit_tokens=600
    )
    assert extract_usage(usage).cached_input_tokens == 600


def test_gemini_hidden_thinking_tokens_billed_as_output() -> None:
    """Gemini reports thoughts only via total_tokens > prompt + completion."""
    usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=200, total_tokens=1500)
    t = extract_usage(usage)
    assert t.output_tokens == 500 and t.reasoning_tokens == 300


def test_missing_usage_is_zero() -> None:
    assert extract_usage(None) == TokenUsage()


def test_cost_formula() -> None:
    t = TokenUsage(input_tokens=1_000_000, cached_input_tokens=500_000, output_tokens=250_000)
    # 0.5M uncached * $1 + 0.5M cached * $0.1 + 0.25M out * $4
    assert compute_cost(t, PRICING) == pytest.approx(0.5 + 0.05 + 1.0)


def test_call_log_writes_jsonl(tmp_path: Path) -> None:
    log = CallLog(tmp_path / "logs" / "calls.jsonl")
    rec = CallRecord(
        call_id="x",
        role="critic:factual",
        model_key="m",
        model_id="m",
        provider="p",
        attempt=1,
        reask=0,
        started_at=datetime.now(UTC),
        latency_s=0.1,
        success=True,
        cost_usd=0.002,
    )
    log.add(rec)
    log.add(rec)
    lines = (tmp_path / "logs" / "calls.jsonl").read_text().splitlines()
    assert len(lines) == 2 and '"role": "critic:factual"' in lines[0]
    assert log.total_cost() == pytest.approx(0.004)
