"""FastAPI endpoints with a mocked arbiter and a temporary store."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from arbiter.api.main import create_app
from arbiter.config import AppConfig
from arbiter.pipeline import Arbiter
from arbiter.storage.db import ArbitrationStore
from tests.helpers import TEXT
from tests.test_eval_runner import CLEAN_TEXT, Scripted, _llm

QUESTION = "When did Apollo 11 land, who walked on the Moon, and for how long?"


@pytest.fixture
def store(tmp_path: Path) -> ArbitrationStore:
    return ArbitrationStore(tmp_path / "api.db")


@pytest.fixture
def client(config: AppConfig, store: ArbitrationStore) -> Iterator[TestClient]:
    arbiter = Arbiter(config, llm=_llm(config, Scripted()))
    with TestClient(create_app(config=config, arbiter=arbiter, store=store)) as c:
        yield c


def test_arbitrate_returns_and_stores_full_record(
    client: TestClient, store: ArbitrationStore
) -> None:
    r = client.post("/v1/arbitrate", json={"output": TEXT, "question": QUESTION})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"]["passed"] is False
    assert len(body["critiques"]) == 3 and body["llm_calls"] == 4
    assert body["disagreements"]["groups"] and body["adjudication"]["adjudicator"].startswith(
        "llm:"
    )
    assert body["cost_usd"] > 0
    stored = store.get(body["id"])
    assert stored is not None and stored.verdict.model_dump(mode="json") == body["verdict"]


def test_get_returns_identical_record_and_404(client: TestClient) -> None:
    created = client.post("/v1/arbitrate", json={"output": CLEAN_TEXT}).json()
    assert created["verdict"]["short_circuited"] is True and created["adjudication"] is None
    fetched = client.get(f"/v1/arbitrations/{created['id']}")
    assert fetched.status_code == 200 and fetched.json() == created
    missing = client.get("/v1/arbitrations/does-not-exist")
    assert missing.status_code == 404 and "not found" in missing.json()["detail"]


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ({"output": ""}, "at least 1 character"),
        ({"output": "   "}, "empty"),
        ({"output": "x" * 20_001}, "max_input_chars"),
        ({"question": "no output"}, "Field required"),
    ],
)
def test_bad_input_is_422(client: TestClient, payload: dict[str, str], fragment: str) -> None:
    r = client.post("/v1/arbitrate", json=payload)
    assert r.status_code == 422 and fragment in r.text


def test_batch_partial_failure_and_listing(client: TestClient) -> None:
    r = client.post(
        "/v1/arbitrate/batch",
        json={
            "items": [
                {"output": TEXT, "question": QUESTION},
                {"output": "x" * 20_001},
                {"output": CLEAN_TEXT},
            ]
        },
    )
    assert r.status_code == 200
    body = r.json()
    statuses = [i["status"] for i in body["results"]]
    assert statuses == ["ok", "error", "ok"]
    assert "max_input_chars" in body["results"][1]["error"]
    assert body["results"][0]["passed"] is False and body["results"][2]["passed"] is True
    listed = client.get("/v1/arbitrations", params={"batch_id": body["batch_id"]}).json()
    assert {x["id"] for x in listed} == {body["results"][0]["id"], body["results"][2]["id"]}
    assert all(x["batch_id"] == body["batch_id"] for x in listed)


def test_batch_size_is_capped(client: TestClient) -> None:
    r = client.post("/v1/arbitrate/batch", json={"items": [{"output": "a"}] * 21})
    assert r.status_code == 422


def test_list_pagination(client: TestClient) -> None:
    for _ in range(3):
        client.post("/v1/arbitrate", json={"output": CLEAN_TEXT})
    assert len(client.get("/v1/arbitrations", params={"limit": 2}).json()) == 2
    assert len(client.get("/v1/arbitrations", params={"offset": 2}).json()) == 1
    assert client.get("/v1/arbitrations", params={"limit": 0}).status_code == 422


def test_openapi_docs_are_complete(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    assert spec["info"]["title"] == "LLM Output Arbitration API"
    assert set(spec["paths"]) >= {
        "/v1/arbitrate",
        "/v1/arbitrate/batch",
        "/v1/arbitrations/{arbitration_id}",
        "/v1/arbitrations",
    }
    req = spec["components"]["schemas"]["ArbitrateRequest"]["properties"]
    assert req["output"]["examples"] and "description" in req["question"]
    assert client.get("/docs").status_code == 200


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok", "stored_arbitrations": 0}


def test_importing_the_module_has_no_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    import arbiter.api.main as main

    monkeypatch.chdir(tmp_path)  # an empty dir: no config.yaml, no data/
    importlib.reload(main)  # building `app` must not read config, create a DB or call providers
    assert list(tmp_path.iterdir()) == []
