"""LangGraph state.

The state is a TypedDict that flows through the graph. Each node returns a PARTIAL dict
containing only the keys it updates, and LangGraph merges it in.

Normally a later write to a key replaces the earlier value. ``critiques`` and ``failures``
are different: three critic nodes run in parallel and all write to them in the same step, so
they carry a *reducer* (``Annotated[list, merge_by_critic]``) that tells LangGraph how to
combine concurrent writes — here: concatenate, then sort by critic name so results don't
depend on which provider happened to answer first.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from arbiter.config import Dimension
from arbiter.critics.critic import CriticSpec
from arbiter.schemas.critique import CriticFailure, Critique
from arbiter.schemas.disagreement import DisagreementReport
from arbiter.schemas.verdict import Adjudication, Verdict


def merge_by_critic[T: (Critique, CriticFailure)](left: list[T], right: list[T]) -> list[T]:
    """Reducer: append concurrent writes, deterministic order."""
    return sorted(left + right, key=lambda x: x.critic)


class ArbitrationState(TypedDict, total=False):
    """Shared state of one arbitration run."""

    # input (set by the caller, normalised by parse_input)
    request_id: str
    question: str | None
    output: str
    # fan-in targets — written concurrently by the critic branches
    critiques: Annotated[list[Critique], merge_by_critic]
    failures: Annotated[list[CriticFailure], merge_by_critic]
    # downstream
    missing_dimensions: list[Dimension]
    report: DisagreementReport
    adjudication: Adjudication | None
    verdict: Verdict


class CriticTask(TypedDict):
    """Private input for ONE critic branch (sent via ``Send``; not part of the shared state)."""

    spec: CriticSpec
    question: str | None
    output: str
