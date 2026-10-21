"""Typed configuration: loads ``config.yaml`` + ``.env`` into validated Pydantic models.

Validation happens once at startup, so a typo in a role or model name fails immediately
with a clear message instead of halfway through a 40-case evaluation run.
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CONFIG_PATH = Path("config.yaml")


class Dimension(StrEnum):
    """The three evaluation dimensions; each core critic owns exactly one."""

    FACTUAL = "factual"
    LOGIC = "logic"
    COMPLETENESS = "completeness"


class _Strict(BaseModel):
    """Base model that rejects unknown keys (catches config typos)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderConfig(_Strict):
    """One OpenAI-compatible endpoint."""

    base_url: str
    api_key_env: str
    structured_mode: Literal["json", "json_schema"] = "json"
    free_tier: bool = False
    requires_api_key: bool = True

    def api_key(self) -> str | None:
        """Read the API key from the environment (never stored in config)."""
        return os.environ.get(self.api_key_env) or None


class Pricing(_Strict):
    """USD per one million tokens."""

    input: float = Field(ge=0)
    cached_input: float = Field(ge=0)
    output: float = Field(ge=0)


class ModelConfig(_Strict):
    """A concrete model on a provider, plus per-model request tweaks."""

    provider: str
    model_id: str
    pricing: Pricing
    extra_body: dict[str, Any] = Field(default_factory=dict)
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)


class ExtraCriticConfig(_Strict):
    """An additional (experimental) critic, e.g. a local Ollama model on one dimension."""

    dimension: Dimension
    model: str


class RolesConfig(_Strict):
    """Which model plays which role. Values are keys into ``models``."""

    critics: dict[Dimension, str]
    extra_critics: dict[str, ExtraCriticConfig] = Field(default_factory=dict)
    adjudicator: str
    baseline: str
    case_builder: str
    match_judge: str

    @model_validator(mode="after")
    def _all_dimensions_covered(self) -> RolesConfig:
        missing = set(Dimension) - set(self.critics)
        if missing:
            raise ValueError(f"roles.critics is missing dimensions: {sorted(missing)}")
        return self


class LimitsConfig(_Strict):
    """Cost and reliability guard-rails applied to every LLM call."""

    max_concurrency: int = Field(default=4, ge=1, le=64)
    max_output_tokens: int = Field(default=2048, ge=64)
    adjudicator_max_output_tokens: int = Field(default=4096, ge=64)
    max_retries: int = Field(default=4, ge=0, le=10)
    backoff_initial_s: float = Field(default=2.0, gt=0)
    backoff_max_s: float = Field(default=60.0, gt=0)
    request_timeout_s: float = Field(default=120.0, gt=0)
    max_input_chars: int = Field(default=20_000, ge=100)


class DisagreementConfig(_Strict):
    """Thresholds for the disagreement detector."""

    unanimous_pass_min_confidence: float = Field(default=0.8, ge=0, le=1)
    span_overlap_min_ratio: float = Field(default=0.5, gt=0, le=1)
    low_confidence_below: float = Field(default=0.5, ge=0, le=1)


class EvaluationConfig(_Strict):
    """Phase 5 evaluation settings."""

    span_min_overlap: float = Field(default=0.5, gt=0, le=1)
    case_concurrency: int = Field(default=3, ge=1, le=32)


class VerdictConfig(_Strict):
    """How a verdict turns into pass/fail."""

    fail_at_severity: Literal["low", "medium", "high"] = "medium"


class PathsConfig(_Strict):
    """Filesystem locations (relative to the working directory)."""

    db: Path = Path("data/arbiter.db")
    call_log: Path = Path("logs/llm_calls.jsonl")


class AppConfig(_Strict):
    """Root configuration object."""

    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    roles: RolesConfig
    limits: LimitsConfig = LimitsConfig()
    disagreement: DisagreementConfig = DisagreementConfig()
    verdict: VerdictConfig = VerdictConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    paths: PathsConfig = PathsConfig()

    @model_validator(mode="after")
    def _cross_references_resolve(self) -> AppConfig:
        for name, model in self.models.items():
            if model.provider not in self.providers:
                raise ValueError(f"model '{name}' uses unknown provider '{model.provider}'")
        role_refs = {
            **{f"critics.{d}": m for d, m in self.roles.critics.items()},
            **{f"extra_critics.{k}": c.model for k, c in self.roles.extra_critics.items()},
            "adjudicator": self.roles.adjudicator,
            "baseline": self.roles.baseline,
            "case_builder": self.roles.case_builder,
            "match_judge": self.roles.match_judge,
        }
        for role, model_key in role_refs.items():
            if model_key not in self.models:
                raise ValueError(f"roles.{role} references unknown model '{model_key}'")
        return self

    def model(self, key: str) -> ModelConfig:
        """Look up a model entry by its config key."""
        return self.models[key]

    def provider_for(self, model_key: str) -> ProviderConfig:
        """Return the provider that serves ``model_key``."""
        return self.providers[self.models[model_key].provider]


def load_config(
    path: Path | str | None = None, *, env_file: Path | str | None = ".env"
) -> AppConfig:
    """Load and validate configuration.

    Args:
        path: YAML file; defaults to ``$ARBITER_CONFIG`` or ``config.yaml``.
        env_file: dotenv file to load secrets from (``None`` to skip). Existing
            environment variables win over the file.
    """
    if env_file is not None:
        load_dotenv(env_file, override=False)
    resolved = Path(path or os.environ.get("ARBITER_CONFIG") or DEFAULT_CONFIG_PATH)
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    config = AppConfig.model_validate(raw)
    if db_override := os.environ.get("ARBITER_DB_PATH"):
        config = config.model_copy(
            update={"paths": config.paths.model_copy(update={"db": Path(db_override)})}
        )
    return config


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Process-wide cached config for app entry points (API, UI, scripts)."""
    return load_config()
