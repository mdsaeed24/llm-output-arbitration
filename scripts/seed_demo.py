"""Load demo arbitrations into the SQLite store so the UI has data immediately.

Usage:
    python scripts/seed_demo.py                 # insert any demo records not yet present
    python scripts/seed_demo.py --if-empty      # only seed a brand-new database (docker start-up)
    python scripts/seed_demo.py --export-from reports/runs/<run>   # rebuild the fixture

No API keys or LLM calls needed: the fixture holds recorded results from an evaluation run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from arbiter.seed import DEMO_CASES, FIXTURE, export_fixture, load_fixture, seed
from arbiter.storage.db import ArbitrationStore
from arbiter.ui.data import db_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, help="SQLite path (default: $ARBITER_DB_PATH/config).")
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--if-empty", action="store_true", help="Skip if the DB has any rows.")
    parser.add_argument("--export-from", type=Path, help="Eval run dir to rebuild the fixture.")
    args = parser.parse_args()

    if args.export_from:
        n = export_fixture(args.export_from, DEMO_CASES, args.fixture)
        print(f"wrote {n} record(s) to {args.fixture}")
        return

    store = ArbitrationStore(args.db or db_path())
    if args.if_empty and store.count():
        print(f"{store.path}: {store.count()} arbitration(s) already stored — not seeding")
        return
    if not args.fixture.exists():
        sys.exit(f"fixture not found: {args.fixture}")
    inserted, skipped = seed(store, load_fixture(args.fixture))
    print(f"{store.path}: seeded {inserted} demo arbitration(s), {skipped} already present")


if __name__ == "__main__":
    main()
