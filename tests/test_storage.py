"""SQLite audit trail."""

from __future__ import annotations

from pathlib import Path

import pytest

from arbiter.config import AppConfig
from arbiter.pipeline import Arbiter
from arbiter.schemas.verdict import ArbitrationResult
from arbiter.storage.db import ArbitrationStore, DuplicateIdError, excerpt
from tests.helpers import TEXT
from tests.test_eval_runner import CLEAN_TEXT, Scripted, _llm


async def _results(config: AppConfig) -> list[ArbitrationResult]:
    arbiter = Arbiter(config, llm=_llm(config, Scripted()))
    return [
        await arbiter.arbitrate(TEXT, "When did Apollo 11 land?", request_id="a1"),
        await arbiter.arbitrate(CLEAN_TEXT, "Boiling point?", request_id="c1"),
    ]


async def test_round_trip_preserves_the_full_record(config: AppConfig, tmp_path: Path) -> None:
    store = ArbitrationStore(tmp_path / "x.db")
    flawed, clean = await _results(config)
    store.save(flawed)
    store.save(clean, batch_id="b1")
    got = store.get("a1")
    assert got == flawed  # every critique, call record and ruling survives
    assert store.get("nope") is None
    assert store.count() == 2


async def test_summaries_and_batches(config: AppConfig, tmp_path: Path) -> None:
    store = ArbitrationStore(tmp_path / "x.db")
    flawed, clean = await _results(config)
    store.save(flawed)
    store.save(clean, batch_id="b1")
    rows = store.list_summaries()
    assert {r.id for r in rows} == {"a1", "c1"}
    a1 = next(r for r in rows if r.id == "a1")
    assert a1.passed is False and a1.issue_count == len(flawed.verdict.confirmed_issues)
    assert a1.excerpt.startswith("Apollo 11") and a1.cost_usd > 0
    assert [r.id for r in store.list_summaries(batch_id="b1")] == ["c1"]
    assert store.batch_ids() == ["b1"]
    assert len(store.list_summaries(limit=1)) == 1


async def test_duplicate_id_rejected(config: AppConfig, tmp_path: Path) -> None:
    store = ArbitrationStore(tmp_path / "x.db")
    flawed, _ = await _results(config)
    store.save(flawed)
    with pytest.raises(DuplicateIdError):
        store.save(flawed)


def test_store_creates_parent_dirs_and_uses_wal(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "nested" / "dir" / "x.db"
    ArbitrationStore(path)
    mode = sqlite3.connect(path).execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"


def test_excerpt() -> None:
    assert excerpt("a  b\n c") == "a b c"
    assert len(excerpt("x" * 500, limit=50)) == 50
