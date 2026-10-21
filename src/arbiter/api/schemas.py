"""Request/response models for the public API (these define the OpenAPI docs)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from arbiter.schemas.critique import CriticFailure, Critique
from arbiter.schemas.disagreement import DisagreementReport
from arbiter.schemas.verdict import Adjudication, ArbitrationResult, Verdict

MAX_BATCH = 20

_EXAMPLE_QUESTION = (
    "When did Apollo 11 land on the Moon, which astronauts walked on the surface, and how long "
    "did they stay?"
)
_EXAMPLE_OUTPUT = (
    "Apollo 11 landed on the Moon on July 20, 1971. Neil Armstrong and Buzz Aldrin walked on the "
    "surface. Because it happened during the Cold War, it proves that competition is the only "
    "force that drives technological progress."
)


class ArbitrateRequest(BaseModel):
    """One LLM output to evaluate."""

    output: str = Field(
        min_length=1,
        description="The LLM-generated text to evaluate.",
        examples=[_EXAMPLE_OUTPUT],
    )
    question: str | None = Field(
        None,
        description="The original question/prompt. Optional, but completeness can only be "
        "judged properly against it.",
        examples=[_EXAMPLE_QUESTION],
    )


class ArbitrationRecord(BaseModel):
    """The full, stored arbitration: verdict plus the whole evidence chain behind it."""

    id: str = Field(description="Use with GET /v1/arbitrations/{id}.")
    created_at: datetime
    question: str | None
    output: str
    verdict: Verdict = Field(description="The single confidence-scored outcome.")
    critiques: list[Critique] = Field(description="One report per critic that succeeded.")
    failures: list[CriticFailure] = Field(description="Critics that failed after retries.")
    disagreements: DisagreementReport = Field(
        description="Issue groups and the disagreements the adjudicator had to resolve."
    )
    adjudication: Adjudication | None = Field(
        description="Null when the adjudicator was skipped (unanimous confident pass)."
    )
    cost_usd: float = Field(description="Total LLM spend at list prices.")
    latency_s: float = Field(description="Wall-clock time for the arbitration.")
    llm_calls: int = Field(description="HTTP attempts made, including retries and re-asks.")

    @classmethod
    def from_result(cls, r: ArbitrationResult) -> ArbitrationRecord:
        """Build from the pipeline's result object."""
        return cls(
            id=r.id,
            created_at=r.created_at,
            question=r.question,
            output=r.output,
            verdict=r.verdict,
            critiques=r.critiques,
            failures=r.failures,
            disagreements=r.report,
            adjudication=r.adjudication,
            cost_usd=r.cost_usd,
            latency_s=r.latency_s,
            llm_calls=len(r.calls),
        )


class BatchRequest(BaseModel):
    """Several outputs, arbitrated concurrently."""

    items: list[ArbitrateRequest] = Field(min_length=1, max_length=MAX_BATCH)


class BatchItem(BaseModel):
    """Summary for one batch item (fetch the full record by ``id``)."""

    index: int
    status: Literal["ok", "error"]
    id: str | None = None
    error: str | None = None
    passed: bool | None = None
    overall_score: int | None = None
    confidence: float | None = None
    issue_count: int | None = None


class BatchResponse(BaseModel):
    """Batch outcome. One failing item never fails the whole batch."""

    batch_id: str
    results: list[BatchItem]
    cost_usd: float


class ErrorResponse(BaseModel):
    """Error body."""

    detail: str
