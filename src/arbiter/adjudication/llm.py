"""LLM adjudicator: reasons through every issue group and returns a structured ruling."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from arbiter.adjudication import prompts
from arbiter.adjudication.base import AdjudicationInput
from arbiter.config import Dimension
from arbiter.llm.client import LLMClient
from arbiter.llm.telemetry import CallRecord
from arbiter.schemas.critique import Severity
from arbiter.schemas.verdict import Adjudication, ConfirmedIssue, DismissedFlag


class GroupRuling(BaseModel):
    """The adjudicator's decision on one issue group. ``reasoning`` comes first so the model
    commits to an argument before it commits to a decision."""

    group_id: str
    reasoning: str = Field(min_length=1)
    decision: Literal["confirm", "dismiss"]
    severity: Severity | None = Field(None, description="Required when confirming; else null.")
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _severity_iff_confirmed(self) -> GroupRuling:
        if self.decision == "confirm" and self.severity is None:
            raise ValueError(f"{self.group_id}: a confirmed group needs a severity")
        if self.decision == "dismiss":
            self.severity = None
        return self


class AdjudicationDraft(BaseModel):
    """Raw adjudicator output."""

    model_config = ConfigDict(title="AdjudicationDraft")

    rulings: list[GroupRuling]
    overall_score: int = Field(ge=1, le=10)
    confidence: float = Field(ge=0, le=1)
    summary: str = Field(min_length=1)


def checked_schema(group_ids: list[str]) -> type[AdjudicationDraft]:
    """A schema that also requires exactly one ruling per expected group id.

    Because this is a Pydantic validator, a missing or invented group id becomes a normal
    validation error — which the LLM client turns into a re-ask naming the exact problem.
    """
    expected = set(group_ids)

    class CheckedAdjudication(AdjudicationDraft):
        model_config = ConfigDict(title="AdjudicationDraft")

        @model_validator(mode="after")
        def _one_ruling_per_group(self) -> CheckedAdjudication:
            got = [r.group_id for r in self.rulings]
            problems = []
            if missing := sorted(expected - set(got)):
                problems.append(f"missing rulings for {missing}")
            if unknown := sorted(set(got) - expected):
                problems.append(f"unknown group ids {unknown}")
            if dupes := sorted({g for g in got if got.count(g) > 1}):
                problems.append(f"more than one ruling for {dupes}")
            if problems:
                raise ValueError(
                    "rulings must cover exactly these group ids: "
                    f"{sorted(expected)} ({'; '.join(problems)})"
                )
            return self

    CheckedAdjudication.__name__ = "AdjudicationDraft"
    return CheckedAdjudication


class LLMAdjudicator:
    """Adjudicates with the model configured as ``roles.adjudicator``."""

    def __init__(self, llm: LLMClient, model_key: str | None = None) -> None:
        self.llm = llm
        self.model_key = model_key or llm.config.roles.adjudicator
        self.name = f"llm:{self.model_key}"

    async def adjudicate(self, inp: AdjudicationInput) -> Adjudication:
        """Rule on every group and build the :class:`Adjudication`."""
        covered = {c.dimension for c in inp.critiques}
        result = await self.llm.structured(
            role="adjudicator",
            model_key=self.model_key,
            system=prompts.SYSTEM_PROMPT,
            user=prompts.user_prompt(inp, missing=[d for d in Dimension if d not in covered]),
            schema=checked_schema([g.id for g in inp.report.groups]),
            max_output_tokens=self.llm.config.limits.adjudicator_max_output_tokens,
        )
        return to_adjudication(result.value, inp, adjudicator=self.name, calls=result.calls)


def to_adjudication(
    draft: AdjudicationDraft,
    inp: AdjudicationInput,
    *,
    adjudicator: str,
    calls: list[CallRecord],
) -> Adjudication:
    """Map group rulings back onto spans, critics and disagreements."""
    groups = {g.id: g for g in inp.report.groups}
    disagreement_ids: dict[str, list[str]] = {}
    for d in inp.report.disagreements:
        disagreement_ids.setdefault(d.group_id, []).append(d.id)

    confirmed: list[ConfirmedIssue] = []
    dismissed: list[DismissedFlag] = []
    for ruling in sorted(draft.rulings, key=lambda r: list(groups).index(r.group_id)):
        g = groups[ruling.group_id]
        common = {
            "issue_ids": g.issue_ids,
            "raised_by": g.critics,
            "quote": inp.output[g.span_start : g.span_end],
            "span_start": g.span_start,
            "span_end": g.span_end,
            "disagreement_ids": disagreement_ids.get(g.id, []),
        }
        if ruling.decision == "confirm":
            assert ruling.severity is not None  # enforced by GroupRuling
            confirmed.append(
                ConfirmedIssue(
                    group_id=g.id,
                    dimensions=g.dimensions,
                    severity=ruling.severity,
                    evidence=ruling.reasoning,
                    confidence=ruling.confidence,
                    **common,
                )
            )
        else:
            dismissed.append(DismissedFlag(rule="adjudicator", reason=ruling.reasoning, **common))

    return Adjudication(
        adjudicator=adjudicator,
        overall_score=draft.overall_score,
        confidence=draft.confidence,
        confirmed=confirmed,
        dismissed=dismissed,
        summary=draft.summary,
        calls=calls,
    )
