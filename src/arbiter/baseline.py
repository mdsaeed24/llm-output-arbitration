"""Single-reviewer baseline: ONE model, ONE prompt, ONE call — what the full system must beat.

Fairness rules (so Phase 5 measures the architecture, nothing else):
* Same model as the adjudicator by default (``roles.baseline``), i.e. the system's strongest
  component — not a weak strawman.
* Same quote rules, severity definitions and per-dimension checklists as the critics
  (imported from ``critics.prompts``).
* Same post-processing: quotes validated against the text, invalid ones dismissed by rule,
  and the same pass/fail threshold.
* Each issue carries a ``type`` so the Phase 5 matching rules (span overlap for
  factual/logic, judge for completeness) apply identically.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

from arbiter.adjudication.prompts import OVERALL_SCORE_RUBRIC
from arbiter.adjudication.synthesis import is_pass
from arbiter.config import Dimension
from arbiter.critics.prompts import CHECKS, DATA_RULE, ISSUE_RULES, question_block
from arbiter.critics.quotes import locate_quote
from arbiter.graph.nodes import InputError
from arbiter.llm.client import LLMClient
from arbiter.llm.telemetry import CallRecord
from arbiter.schemas.critique import Issue, IssueDraft
from arbiter.schemas.verdict import ConfirmedIssue, DismissedFlag, Verdict

CRITIC_NAME = "baseline"

SYSTEM_PROMPT = f"""\
You are an expert reviewer. Review this output for factual errors, logical flaws, and missing
parts.

For every issue, set "type" to "factual", "logic" or "completeness".

{ISSUE_RULES}

{DATA_RULE}

FACTUAL ERRORS — {CHECKS[Dimension.FACTUAL]}

LOGICAL FLAWS — {CHECKS[Dimension.LOGIC]}

MISSING PARTS — {CHECKS[Dimension.COMPLETENESS]}

{OVERALL_SCORE_RUBRIC}
"confidence" (0.0-1.0): how sure you are of your overall assessment.
"summary": one paragraph (3-5 sentences) on what is wrong with the output, or that it is sound.

Write "analysis" first (2-4 sentences), then "issues", then "overall_score", "confidence" and
"summary"."""

USER_TEMPLATE = """\
{question_block}

<evaluated_output>
{output}
</evaluated_output>

Review the text in <evaluated_output> and reply with the JSON object."""


class BaselineIssueDraft(IssueDraft):
    """A baseline issue: like a critic issue, plus the type the reviewer assigns it."""

    type: Literal["factual", "logic", "completeness"]


class BaselineDraft(BaseModel):
    """Raw baseline output."""

    analysis: str
    issues: list[BaselineIssueDraft] = Field(default_factory=list, max_length=30)
    overall_score: int = Field(ge=1, le=10)
    confidence: float = Field(ge=0, le=1)
    summary: str = Field(min_length=1)


class BaselineResult(BaseModel):
    """The baseline's review, in the same Verdict shape as the full system."""

    model_key: str
    model_id: str
    question: str | None
    output: str
    analysis: str
    issues: list[Issue]
    verdict: Verdict
    calls: list[CallRecord]
    latency_s: float

    @property
    def cost_usd(self) -> float:
        """Total spend for this review."""
        return sum(c.cost_usd for c in self.calls)


class SingleReviewer:
    """Runs the baseline review."""

    def __init__(self, llm: LLMClient, model_key: str | None = None) -> None:
        self.llm = llm
        self.model_key = model_key or llm.config.roles.baseline

    async def review(self, output: str, question: str | None = None) -> BaselineResult:
        """Review ``output`` in one call and build a Verdict with the system's rules."""
        output = output.strip()
        question = (question or "").strip() or None
        limit = self.llm.config.limits.max_input_chars
        if not output:
            raise InputError("output to evaluate is empty")
        if len(output) + len(question or "") > limit:
            raise InputError(f"input exceeds limits.max_input_chars={limit}")
        start = time.perf_counter()
        result = await self.llm.structured(
            role="baseline",
            model_key=self.model_key,
            system=SYSTEM_PROMPT,
            user=USER_TEMPLATE.format(question_block=question_block(question), output=output),
            schema=BaselineDraft,
        )
        draft = result.value
        issues = _validate(draft.issues, output)
        confirmed = [
            ConfirmedIssue(
                group_id=i.id,
                issue_ids=[i.id],
                raised_by=[CRITIC_NAME],
                dimensions=[i.dimension],
                severity=i.severity,
                quote=output[i.span_start : i.span_end],
                span_start=i.span_start,
                span_end=i.span_end,
                evidence=i.problem,
                confidence=draft.confidence,
            )
            for i in issues
            if i.valid
        ]
        dismissed = [
            DismissedFlag(
                issue_ids=[i.id],
                raised_by=[CRITIC_NAME],
                quote=i.quote,
                rule="invalid_quote",
                reason=f"Quote does not appear in the evaluated text ({i.invalid_reason}).",
            )
            for i in issues
            if not i.valid
        ]
        verdict = Verdict(
            status="ok",
            passed=is_pass(confirmed, self.llm.config),
            overall_score=draft.overall_score,
            confidence=draft.confidence,
            confirmed_issues=confirmed,
            dismissed_flags=dismissed,
            summary=draft.summary,
            adjudicator=None,
        )
        return BaselineResult(
            model_key=self.model_key,
            model_id=self.llm.config.model(self.model_key).model_id,
            question=question,
            output=output,
            analysis=draft.analysis,
            issues=issues,
            verdict=verdict,
            calls=result.calls,
            latency_s=time.perf_counter() - start,
        )


def _validate(drafts: list[BaselineIssueDraft], output: str) -> list[Issue]:
    """Quote-validate baseline issues exactly like critic issues."""
    issues = []
    for n, draft in enumerate(drafts, start=1):
        match = locate_quote(draft.quote, output)
        issues.append(
            Issue(
                **draft.model_dump(exclude={"type"}),
                id=f"{CRITIC_NAME}-{n}",
                critic=CRITIC_NAME,
                dimension=Dimension(draft.type),
                valid=match.valid,
                span_start=match.start,
                span_end=match.end,
                match_method=match.method,
                invalid_reason=match.reason,
            )
        )
    return issues
