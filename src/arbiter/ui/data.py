"""Data access for the Streamlit UI: read the shared SQLite store, write via the API."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from arbiter.config import load_config
from arbiter.storage.db import ArbitrationStore

REPORTS_DIR = Path(os.environ.get("ARBITER_REPORTS_DIR", "reports"))


def db_path() -> Path:
    """``$ARBITER_DB_PATH`` if set (docker-compose), else ``paths.db`` from config.yaml."""
    env = os.environ.get("ARBITER_DB_PATH")
    return Path(env) if env else load_config(env_file=None).paths.db


def open_store() -> ArbitrationStore:
    """The shared audit-trail store (read side)."""
    return ArbitrationStore(db_path())


def api_url() -> str:
    """Base URL of the FastAPI service."""
    return os.environ.get("ARBITER_API_URL", "http://localhost:8000").rstrip("/")


class ApiError(RuntimeError):
    """The API could not be reached or rejected the request."""


def submit_arbitration(output: str, question: str | None) -> dict[str, Any]:
    """POST /v1/arbitrate. New arbitrations go through the API so it stays the only writer."""
    try:
        r = httpx.post(
            f"{api_url()}/v1/arbitrate",
            json={"output": output, "question": question or None},
            timeout=600,
        )
    except httpx.HTTPError as exc:
        raise ApiError(f"API not reachable at {api_url()} ({type(exc).__name__}).") from exc
    if r.status_code != 200:
        detail = (
            r.json().get("detail", r.text)
            if r.headers.get("content-type", "").startswith("application/json")
            else r.text
        )
        raise ApiError(f"API returned {r.status_code}: {detail}")
    body: dict[str, Any] = r.json()
    return body


@dataclass(frozen=True)
class EvalReport:
    """One evaluation run's outputs on disk."""

    label: str
    metrics_path: Path
    report_path: Path


def eval_reports(root: Path = REPORTS_DIR) -> list[EvalReport]:
    """Available evaluation reports, v1 (pre-registered) first."""
    candidates = [
        EvalReport("v1 — pre-registered", root / "eval_metrics.json", root / "eval_report.md"),
        EvalReport(
            "v2 — tuned on the same cases (optimistic)",
            root / "eval_metrics_v2.json",
            root / "eval_report_v2.md",
        ),
    ]
    return [c for c in candidates if c.metrics_path.exists()]


def load_metrics(report: EvalReport) -> dict[str, Any]:
    """Parsed ``eval_metrics*.json``."""
    data: dict[str, Any] = json.loads(report.metrics_path.read_text(encoding="utf-8"))
    return data
