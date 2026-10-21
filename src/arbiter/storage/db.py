"""SQLite audit trail: every arbitration the API runs is stored here.

One row per arbitration: the full ``ArbitrationResult`` as JSON (the audit record) plus a few
summary columns so listings and the batch table don't need to parse JSON. WAL journal mode
lets the Streamlit UI read while the API writes (they share the file in docker-compose).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from arbiter.schemas.verdict import ArbitrationResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS arbitrations (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    batch_id      TEXT,
    question      TEXT,
    output        TEXT NOT NULL,
    status        TEXT NOT NULL,
    passed        INTEGER,
    overall_score INTEGER,
    confidence    REAL NOT NULL,
    issue_count   INTEGER NOT NULL,
    dismissed_count INTEGER NOT NULL,
    cost_usd      REAL NOT NULL,
    latency_s     REAL NOT NULL,
    adjudicator   TEXT,
    result_json   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_arbitrations_created ON arbitrations (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_arbitrations_batch ON arbitrations (batch_id);
"""


class ArbitrationSummary(BaseModel):
    """A row for listings: enough for the batch table, without the full record."""

    id: str
    created_at: datetime
    batch_id: str | None
    question: str | None
    excerpt: str
    status: str
    passed: bool | None
    overall_score: int | None
    confidence: float
    issue_count: int
    dismissed_count: int
    cost_usd: float
    latency_s: float
    adjudicator: str | None


class DuplicateIdError(ValueError):
    """An arbitration with this id is already stored."""


def excerpt(text: str, limit: int = 120) -> str:
    """First ``limit`` characters of ``text`` on one line."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


class ArbitrationStore:
    """Small repository over the ``arbitrations`` table."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save(self, result: ArbitrationResult, *, batch_id: str | None = None) -> None:
        """Insert one arbitration. Raises :class:`DuplicateIdError` if the id exists."""
        v = result.verdict
        row = (
            result.id,
            result.created_at.isoformat(),
            batch_id,
            result.question,
            result.output,
            v.status,
            None if v.passed is None else int(v.passed),
            v.overall_score,
            v.confidence,
            len(v.confirmed_issues),
            len(v.dismissed_flags),
            result.cost_usd,
            result.latency_s,
            v.adjudicator,
            result.model_dump_json(),
        )
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO arbitrations VALUES (" + ",".join("?" * len(row)) + ")", row
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateIdError(f"arbitration '{result.id}' already exists") from exc

    def get(self, arbitration_id: str) -> ArbitrationResult | None:
        """The full stored record, or ``None``."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result_json FROM arbitrations WHERE id = ?", (arbitration_id,)
            ).fetchone()
        return ArbitrationResult.model_validate_json(row["result_json"]) if row else None

    def list_summaries(
        self, *, limit: int = 50, offset: int = 0, batch_id: str | None = None
    ) -> list[ArbitrationSummary]:
        """Newest first; optionally only one batch."""
        sql = "SELECT * FROM arbitrations"
        params: list[object] = []
        if batch_id is not None:
            sql += " WHERE batch_id = ?"
            params.append(batch_id)
        sql += " ORDER BY created_at DESC, id LIMIT ? OFFSET ?"
        params += [limit, offset]
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._summary(r) for r in rows]

    def batch_ids(self) -> list[str]:
        """Distinct batch ids, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT batch_id, MAX(created_at) AS t FROM arbitrations "
                "WHERE batch_id IS NOT NULL GROUP BY batch_id ORDER BY t DESC"
            ).fetchall()
        return [r["batch_id"] for r in rows]

    def count(self) -> int:
        """Number of stored arbitrations."""
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM arbitrations").fetchone()[0])

    @staticmethod
    def _summary(row: sqlite3.Row) -> ArbitrationSummary:
        return ArbitrationSummary(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            batch_id=row["batch_id"],
            question=row["question"],
            excerpt=excerpt(row["output"]),
            status=row["status"],
            passed=None if row["passed"] is None else bool(row["passed"]),
            overall_score=row["overall_score"],
            confidence=row["confidence"],
            issue_count=row["issue_count"],
            dismissed_count=row["dismissed_count"],
            cost_usd=row["cost_usd"],
            latency_s=row["latency_s"],
            adjudicator=row["adjudicator"],
        )
