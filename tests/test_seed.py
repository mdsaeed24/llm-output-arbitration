"""Demo seeding: the committed fixture stays valid, and seeding is idempotent."""

from __future__ import annotations

from pathlib import Path

from arbiter.config import AppConfig
from arbiter.evaluation.findings import CaseRun
from arbiter.pipeline import Arbiter
from arbiter.seed import DEMO_BATCH, DEMO_CASES, export_fixture, load_fixture, seed
from arbiter.storage.db import ArbitrationStore
from tests.conftest import REPO_ROOT
from tests.helpers import TEXT
from tests.test_eval_runner import Scripted, _llm

FIXTURE = REPO_ROOT / "data" / "demo" / "arbitrations.jsonl"


def test_committed_fixture_matches_current_schema() -> None:
    """Fails if a schema change makes the demo data unloadable (re-export it then)."""
    results = load_fixture(FIXTURE)
    assert [r.id for r in results] == list(DEMO_CASES)
    kinds = {r.id: r.verdict for r in results}
    assert kinds["clean-apollo-11"].short_circuited and kinds["clean-apollo-11"].passed
    assert kinds["index-fund-mixed"].passed is False
    assert any(v.dismissed_flags for v in kinds.values())  # UI shows grey highlights
    assert all(r.critiques for r in results)


def test_seed_is_idempotent(tmp_path: Path) -> None:
    store = ArbitrationStore(tmp_path / "s.db")
    results = load_fixture(FIXTURE)
    assert seed(store, results) == (len(DEMO_CASES), 0)
    assert seed(store, results) == (0, len(DEMO_CASES))
    assert {s.batch_id for s in store.list_summaries()} == {DEMO_BATCH}


async def test_export_round_trip(config: AppConfig, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "cases").mkdir(parents=True)
    arbiter = Arbiter(config, llm=_llm(config, Scripted()))
    result = await arbiter.arbitrate(TEXT, request_id="index-fund-mixed")
    case = CaseRun(case_id="index-fund-mixed", system=result)
    (run_dir / "cases" / "index-fund-mixed.json").write_text(case.model_dump_json())
    out = tmp_path / "fixture.jsonl"
    assert export_fixture(run_dir, DEMO_CASES, out) == 1  # missing cases are skipped
    assert load_fixture(out) == [result]
