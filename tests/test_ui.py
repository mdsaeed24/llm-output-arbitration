"""Verdict Explorer: pure rendering logic + headless page runs (Streamlit AppTest)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from arbiter.config import AppConfig, Dimension
from arbiter.pipeline import Arbiter
from arbiter.schemas.verdict import ArbitrationResult
from arbiter.storage.db import ArbitrationStore
from arbiter.ui.render import (
    Mark,
    agreement_rows,
    critic_columns,
    evidence_chain,
    highlight_html,
    marks,
    segments,
)
from tests.conftest import REPO_ROOT
from tests.helpers import TEXT
from tests.test_eval_runner import CLEAN_TEXT, Scripted, _llm

PAGES = REPO_ROOT / "src" / "arbiter" / "ui" / "pages"


async def _flawed(config: AppConfig, scripted: Scripted | None = None) -> ArbitrationResult:
    arbiter = Arbiter(config, llm=_llm(config, scripted or Scripted()))
    return await arbiter.arbitrate(TEXT, "When did Apollo 11 land, and for how long?")


# --------------------------------------------------------------------------- pure logic


async def test_marks_classify_confirmed_and_dismissed(config: AppConfig) -> None:
    r = await _flawed(config)  # scripted adjudicator: g1, g2 confirmed; g3 (logic) dismissed
    ms = marks(r)
    assert [(m.id, m.kind) for m in ms] == [
        ("1", "confirmed"),
        ("2", "confirmed"),
        ("D1", "dismissed"),
    ]
    d1 = ms[-1]
    assert r.output[d1.start : d1.end] == "competition is the only driver of progress"  # type: ignore[misc]


async def test_weak_confirmations_are_amber(config: AppConfig) -> None:
    r = await _flawed(config)
    v = r.verdict
    unsure = v.confirmed_issues[0].model_copy(update={"confidence": 0.4})
    r1 = r.model_copy(
        update={
            "verdict": v.model_copy(update={"confirmed_issues": [unsure, *v.confirmed_issues[1:]]})
        }
    )
    assert [m.kind for m in marks(r1)][:2] == ["low", "confirmed"]
    # confirmed without adjudicator review → every confirmed issue is amber
    r2 = r.model_copy(update={"verdict": v.model_copy(update={"adjudicator": "union-fallback"})})
    assert [m.kind for m in marks(r2) if not m.id.startswith("D")] == ["low", "low"]
    assert "WITHOUT review" in evidence_chain(r2, "1")[-2].title
    # an issue in a low-confidence dimension is amber too
    r3 = r.model_copy(
        update={"verdict": v.model_copy(update={"low_confidence_dimensions": [Dimension.FACTUAL]})}
    )
    assert marks(r3)[0].kind == "low"


def test_segments_split_at_every_boundary_with_priority() -> None:
    text = "aaaa bbbb cccc"
    ms = [
        Mark(id="D1", kind="dismissed", start=0, end=9, title=""),
        Mark(id="1", kind="confirmed", start=5, end=14, title=""),
        Mark(id="D2", kind="dismissed", start=None, end=None, title=""),  # no location
    ]
    segs = segments(text, ms)
    assert [(s.text, s.kind, s.mark_ids) for s in segs] == [
        ("aaaa ", "dismissed", ["D1"]),
        ("bbbb", "confirmed", ["D1", "1"]),  # overlap → confirmed wins
        (" cccc", "confirmed", ["1"]),
    ]
    assert "".join(s.text for s in segs) == text


async def test_evidence_chain_runs_critic_to_final(config: AppConfig) -> None:
    r = await _flawed(config)
    confirmed = evidence_chain(r, "1")
    assert [s.stage for s in confirmed][0] == "critic"
    assert [s.stage for s in confirmed][-2:] == ["ruling", "final"]
    assert "factual critic flagged it" in confirmed[0].title
    assert "verified in text" in confirmed[0].meta
    assert any(s.stage == "disagreement" for s in confirmed)  # uncorroborated
    dismissed = evidence_chain(r, "D1")
    assert dismissed[-2].title == "Adjudicator dismissed it" and dismissed[-1].title.startswith(
        "Final"
    )


async def test_highlight_html_escapes_text_and_embeds_chains(config: AppConfig) -> None:
    r = await _flawed(config)
    r = r.model_copy(update={"output": r.output.replace("Apollo", "<b>Apollo</b>")})
    page = highlight_html(r)
    assert "&lt;b&gt;Apollo&lt;/b&gt;" in page and "<b>Apollo</b>" not in page.split("<script>")[0]
    assert "class='confirmed'" in page and "class='dismissed'" in page
    assert "const CHAINS" in page and "Adjudicator dismissed it" in page
    assert "low-confidence flag" in page  # legend: colour is never the only cue
    assert "class='dark'" in highlight_html(r, dark=True)


async def test_untrusted_text_cannot_break_out_of_the_page(config: AppConfig) -> None:
    """The evaluated output and critic text are LLM-generated: treat them as hostile."""
    r = await _flawed(config)
    payload = "</script><script>alert('pwned')</script><img src=x onerror=alert(1)>"
    issue = r.critiques[0].issues[0].model_copy(update={"problem": payload})
    critique = r.critiques[0].model_copy(update={"issues": [issue, *r.critiques[0].issues[1:]]})
    hostile = r.model_copy(
        update={"output": r.output + " " + payload, "critiques": [critique, *r.critiques[1:]]}
    )
    page = highlight_html(hostile)
    assert page.count("<script>") == 1 and page.count("</script>") == 1  # only ours
    assert "<img" not in page and "alert('pwned')" not in page


async def test_clean_result_has_no_marks(config: AppConfig) -> None:
    arbiter = Arbiter(config, llm=_llm(config, Scripted()))
    clean = await arbiter.arbitrate(CLEAN_TEXT)
    assert marks(clean) == [] and "No issues were flagged" in highlight_html(clean)


async def test_critic_columns_and_agreement_rows(config: AppConfig) -> None:
    r = await _flawed(config)
    cols = critic_columns(r)
    assert [c.critic for c in cols] == ["factual", "logic", "completeness"]
    assert all(c.status == "ok" and c.score is not None for c in cols)
    agreements, disagreements = agreement_rows(r)
    assert agreements == [] and len(disagreements) == 3
    assert {d["kind"] for d in disagreements} == {"uncorroborated"}


# --------------------------------------------------------------------------- pages (AppTest)


@pytest.fixture
async def seeded_db(config: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "ui.db"
    store = ArbitrationStore(db)
    arbiter = Arbiter(config, llm=_llm(config, Scripted()))
    store.save(await arbiter.arbitrate(TEXT, "When did Apollo 11 land?", request_id="flawed"))
    store.save(await arbiter.arbitrate(CLEAN_TEXT, request_id="clean"), batch_id="batch-1")
    monkeypatch.setenv("ARBITER_DB_PATH", str(db))
    reports = tmp_path / "reports"
    reports.mkdir()
    for name in (
        "eval_metrics.json",
        "eval_report.md",
        "eval_metrics_v2.json",
        "eval_report_v2.md",
    ):
        if (REPO_ROOT / "reports" / name).exists():
            shutil.copy(REPO_ROOT / "reports" / name, reports / name)
    monkeypatch.setenv("ARBITER_REPORTS_DIR", str(reports))
    monkeypatch.chdir(REPO_ROOT)
    return db


def _run(page: str) -> AppTest:
    at = AppTest.from_file(str(PAGES / page), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.mark.usefixtures("seeded_db")
def test_verdict_page_renders_stored_arbitration() -> None:
    at = _run("verdict.py")
    assert any("Verdict" in t.value for t in at.title)
    assert any(m.label == "Verdict" for m in at.metric)
    assert any("Critic comparison" in h.value for h in at.subheader)


@pytest.mark.usefixtures("seeded_db")
def test_batch_page_lists_and_filters() -> None:
    at = _run("batch.py")
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) == 2
    at.selectbox[0].select("batch-1").run()
    assert len(at.dataframe[0].value) == 1


@pytest.mark.usefixtures("seeded_db")
def test_analytics_page_renders_metrics() -> None:
    if not (REPO_ROOT / "reports" / "eval_metrics.json").exists():
        pytest.skip("no evaluation metrics committed")
    at = _run("analytics.py")
    assert any(m.label.startswith("Recall") for m in at.metric)


def test_verdict_page_with_empty_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARBITER_DB_PATH", str(tmp_path / "empty.db"))
    monkeypatch.chdir(REPO_ROOT)
    at = _run("verdict.py")
    assert any("No arbitrations stored yet" in i.value for i in at.info)
