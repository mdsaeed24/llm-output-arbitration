"""Demo data: recorded arbitrations replayed into the store so the UI has data immediately.

The fixture (``data/demo/arbitrations.jsonl``) holds real ``ArbitrationResult`` records from an
evaluation run, so seeding needs no API keys and costs nothing. It deliberately includes the
system's failures (a wrong short-circuit, a missed arithmetic error) as well as its successes.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from arbiter.evaluation.runner import load_runs
from arbiter.schemas.verdict import ArbitrationResult
from arbiter.storage.db import ArbitrationStore

FIXTURE = Path("data/demo/arbitrations.jsonl")
DEMO_BATCH = "demo"
DEMO_CASES = (
    "index-fund-mixed",
    "python3-release-date",
    "redesign-sales-unsupported",
    "ww2-end-mixed",
    "clean-apollo-11",
    "everest-boiling-point",
    "mortgage-total-interest",
)


def load_fixture(path: Path = FIXTURE) -> list[ArbitrationResult]:
    """Parse the JSONL fixture."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ArbitrationResult.model_validate_json(line) for line in lines if line.strip()]


def seed(store: ArbitrationStore, results: Iterable[ArbitrationResult]) -> tuple[int, int]:
    """Insert results not already present (idempotent). Returns (inserted, skipped)."""
    inserted = skipped = 0
    for r in results:
        if store.get(r.id) is not None:
            skipped += 1
            continue
        store.save(r, batch_id=DEMO_BATCH)
        inserted += 1
    return inserted, skipped


def export_fixture(run_dir: Path, case_ids: Iterable[str], path: Path = FIXTURE) -> int:
    """Rebuild the fixture from an evaluation run (developer tool). Returns records written."""
    runs = load_runs(run_dir)
    records = [runs[c].system for c in case_ids if c in runs and runs[c].system is not None]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(r.model_dump_json() + "\n" for r in records if r is not None), encoding="utf-8"
    )
    return len(records)
