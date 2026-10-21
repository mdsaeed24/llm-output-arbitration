"""FastAPI service: arbitrate LLM outputs and keep every result as an audit trail.

Run:  uvicorn arbiter.api.main:app --reload       (OpenAPI docs at /docs)
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from arbiter.api.schemas import (
    MAX_BATCH,
    ArbitrateRequest,
    ArbitrationRecord,
    BatchItem,
    BatchRequest,
    BatchResponse,
    ErrorResponse,
)
from arbiter.config import AppConfig, get_config
from arbiter.graph.nodes import InputError
from arbiter.pipeline import Arbiter
from arbiter.schemas.verdict import ArbitrationResult
from arbiter.storage.db import ArbitrationStore, ArbitrationSummary

DESCRIPTION = """
Evaluate any LLM-generated output with **three specialised critics on different providers**
(factual accuracy, logical consistency, completeness), resolve their disagreements with an
**adjudicator**, and get back **one confidence-scored verdict** with the full evidence chain.

* Every quoted problem is checked against the text — critics can't invent evidence.
* Critics run in parallel; if one fails, the verdict is returned as `degraded` with that
  dimension marked low-confidence.
* Every arbitration is stored (SQLite) and can be fetched again by id.

Limitations: no live fact-checking (models judge from their own knowledge), and the critics can
share blind spots. See `reports/eval_report.md` for the measured comparison with a single
reviewer.
"""

TAG = "arbitration"


def _store(request: Request) -> ArbitrationStore:
    store: ArbitrationStore = request.app.state.store
    return store


def _arbiter(request: Request) -> Arbiter:
    arbiter: Arbiter = request.app.state.arbiter
    return arbiter


async def _run_and_store(
    request: Request, item: ArbitrateRequest, *, batch_id: str | None = None
) -> ArbitrationResult:
    result = await _arbiter(request).arbitrate(
        item.output, item.question, request_id=uuid.uuid4().hex[:16]
    )
    _store(request).save(result, batch_id=batch_id)
    return result


def create_app(
    *,
    config: AppConfig | None = None,
    arbiter: Arbiter | None = None,
    store: ArbitrationStore | None = None,
) -> FastAPI:
    """App factory. Defaults are built at startup (not import), so importing is side-effect
    free and tests can inject a mocked ``Arbiter`` and a temporary store."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cfg = config or get_config()
        app.state.config = cfg
        app.state.arbiter = arbiter or Arbiter(cfg)
        app.state.store = store or ArbitrationStore(cfg.paths.db)
        yield

    app = FastAPI(
        title="LLM Output Arbitration API",
        version="1.0.0",
        description=DESCRIPTION,
        lifespan=lifespan,
        openapi_tags=[{"name": TAG, "description": "Run and retrieve arbitrations."}],
    )

    @app.exception_handler(InputError)
    async def _input_error(_: Request, exc: InputError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/healthz", include_in_schema=False)
    async def healthz(request: Request) -> dict[str, object]:
        return {"status": "ok", "stored_arbitrations": _store(request).count()}

    @app.post(
        "/v1/arbitrate",
        response_model=ArbitrationRecord,
        tags=[TAG],
        summary="Arbitrate one output",
        responses={422: {"model": ErrorResponse, "description": "Empty or oversized input."}},
    )
    async def arbitrate(request: Request, body: ArbitrateRequest) -> ArbitrationRecord:
        """Run all critics in parallel, detect disagreements, adjudicate, and return the
        verdict with its evidence chain. The result is stored before it is returned."""
        return ArbitrationRecord.from_result(await _run_and_store(request, body))

    @app.post(
        "/v1/arbitrate/batch",
        response_model=BatchResponse,
        tags=[TAG],
        summary=f"Arbitrate up to {MAX_BATCH} outputs",
    )
    async def arbitrate_batch(request: Request, body: BatchRequest) -> BatchResponse:
        """Arbitrate several outputs concurrently (LLM calls stay under the global concurrency
        limit). Returns a summary per item; one failing item does not fail the batch. Fetch
        full records with GET /v1/arbitrations/{id}, or list the batch with
        GET /v1/arbitrations?batch_id=…"""
        batch_id = f"batch-{uuid.uuid4().hex[:10]}"
        outcomes = await asyncio.gather(
            *[_run_and_store(request, item, batch_id=batch_id) for item in body.items],
            return_exceptions=True,
        )
        items, cost = [], 0.0
        for idx, out in enumerate(outcomes):
            if isinstance(out, InputError):
                items.append(BatchItem(index=idx, status="error", error=str(out)))
            elif isinstance(out, BaseException):
                items.append(
                    BatchItem(index=idx, status="error", error=f"{type(out).__name__}: {out}")
                )
            else:
                cost += out.cost_usd
                v = out.verdict
                items.append(
                    BatchItem(
                        index=idx,
                        status="ok",
                        id=out.id,
                        passed=v.passed,
                        overall_score=v.overall_score,
                        confidence=v.confidence,
                        issue_count=len(v.confirmed_issues),
                    )
                )
        return BatchResponse(batch_id=batch_id, results=items, cost_usd=cost)

    @app.get(
        "/v1/arbitrations/{arbitration_id}",
        response_model=ArbitrationRecord,
        tags=[TAG],
        summary="Fetch a stored arbitration",
        responses={404: {"model": ErrorResponse}},
    )
    async def get_arbitration(request: Request, arbitration_id: str) -> ArbitrationRecord:
        """The full audit record exactly as it was produced."""
        result = _store(request).get(arbitration_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"arbitration '{arbitration_id}' not found")
        return ArbitrationRecord.from_result(result)

    @app.get(
        "/v1/arbitrations",
        response_model=list[ArbitrationSummary],
        tags=[TAG],
        summary="List stored arbitrations",
    )
    async def list_arbitrations(
        request: Request,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
        batch_id: Annotated[str | None, Query(description="Only items from this batch.")] = None,
    ) -> list[ArbitrationSummary]:
        """Newest first. Summaries only; fetch a full record by id."""
        return _store(request).list_summaries(limit=limit, offset=offset, batch_id=batch_id)

    return app


app = create_app()
