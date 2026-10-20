"""Tests for typed config loading and cross-reference validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from arbiter.config import AppConfig, Dimension, load_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def _raw() -> dict[str, Any]:
    return yaml.safe_load(REPO_CONFIG.read_text())


def test_repo_config_loads_with_default_assignments() -> None:
    cfg = load_config(REPO_CONFIG, env_file=None)
    assert cfg.roles.critics[Dimension.FACTUAL] == "deepseek-v4-pro"
    assert cfg.roles.critics[Dimension.LOGIC] == "gemini-flash"
    assert cfg.roles.critics[Dimension.COMPLETENESS] == "deepseek-flash"
    assert cfg.roles.adjudicator == "deepseek-v4-pro"
    assert cfg.roles.baseline == "deepseek-v4-pro"


def test_critics_use_at_least_two_providers() -> None:
    """The core claim depends on model diversity, so guard it in config."""
    cfg = load_config(REPO_CONFIG, env_file=None)
    providers = {cfg.provider_for(m).base_url for m in cfg.roles.critics.values()}
    assert len(providers) >= 2


def test_deepseek_thinking_disabled_by_default() -> None:
    cfg = load_config(REPO_CONFIG, env_file=None)
    for key in ("deepseek-v4-pro", "deepseek-flash"):
        assert cfg.model(key).extra_body["thinking"] == {"type": "disabled"}


def test_unknown_model_in_role_is_rejected() -> None:
    raw = _raw()
    raw["roles"]["adjudicator"] = "gpt-nonexistent"
    with pytest.raises(ValidationError, match="unknown model 'gpt-nonexistent'"):
        AppConfig.model_validate(raw)


def test_unknown_provider_is_rejected() -> None:
    raw = _raw()
    raw["models"]["deepseek-flash"]["provider"] = "nope"
    with pytest.raises(ValidationError, match="unknown provider 'nope'"):
        AppConfig.model_validate(raw)


def test_missing_critic_dimension_is_rejected() -> None:
    raw = _raw()
    del raw["roles"]["critics"]["completeness"]
    with pytest.raises(ValidationError, match="missing dimensions"):
        AppConfig.model_validate(raw)


def test_typo_key_is_rejected() -> None:
    raw = _raw()
    raw["limits"]["max_concurency"] = 8  # typo
    with pytest.raises(ValidationError):
        AppConfig.model_validate(raw)


def test_api_key_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(REPO_CONFIG, env_file=None)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    assert cfg.providers["deepseek"].api_key() == "sk-test"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert cfg.providers["gemini"].api_key() is None


def test_db_path_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ARBITER_DB_PATH", str(tmp_path / "x.db"))
    cfg = load_config(REPO_CONFIG, env_file=None)
    assert cfg.paths.db == tmp_path / "x.db"


def test_match_judge_defaults_to_cheap_model() -> None:
    cfg = load_config(REPO_CONFIG, env_file=None)
    assert cfg.roles.match_judge == "deepseek-flash"
