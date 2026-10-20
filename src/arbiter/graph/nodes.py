"""Graph nodes and routing functions.

Nodes are methods on :class:`ArbitrationNodes` so their dependencies (config, LLM client,
adjudicator) are injected once, instead of being stuffed into the graph state — state should
hold data, not live clients.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from typing import Any, Literal

from langgraph.types import Send

from arbiter.adjudication.base import AdjudicationInput, Adjudicator
from arbiter.adjudication.synthesis import synthesize_verdict
from arbiter.config import AppConfig, Dimension
from arbiter.critics.critic import CriticSpec, run_critic
from arbiter.graph.disagreement import detect_disagreements
from arbiter.graph.state import ArbitrationState, CriticTask
from arbiter.llm.client import LLMClient, LLMError
from arbiter.schemas.critique import CriticFailure

logger = logging.getLogger(__name__)

CRITIC_NODE = "run_critic"


class InputError(ValueError):
    """The request itself is unusable (empty or too long)."""


class ArbitrationNodes:
    """All node callables for the arbitration graph."""

    def __init__(
        self,
        config: AppConfig,
        llm: LLMClient,
        adjudicator: Adjudicator,
        specs: Sequence[CriticSpec],
    ) -> None:
        self.config = config
        self.llm = llm
        self.adjudicator = adjudicator
        self.specs = list(specs)

    # ---------------------------------------------------------------- nodes

    async def parse_input(self, state: ArbitrationState) -> dict[str, Any]:
        """Normalise and guard the input before any money is spent."""
        output = (state.get("output") or "").strip()
        question = (state.get("question") or "").strip() or None
        if not output:
            raise InputError("output to evaluate is empty")
        limit = self.config.limits.max_input_chars
        if len(output) + len(question or "") > limit:
            raise InputError(f"input exceeds limits.max_input_chars={limit}")
        return {
            "request_id": state.get("request_id") or uuid.uuid4().hex[:12],
            "output": output,
            "question": question,
        }

    async def run_critic(self, task: CriticTask) -> dict[str, Any]:
        """ONE critic branch. A failure is recorded, not raised, so siblings keep going."""
        spec = task["spec"]
        try:
            critique = await run_critic(
                self.llm, spec, output=task["output"], question=task["question"]
            )
        except LLMError as exc:
            logger.warning("critic %s failed: %s", spec.name, exc)
            failure = CriticFailure(
                critic=spec.name,
                dimension=spec.dimension,
                model_key=spec.model_key,
                error_type=type(exc).__name__,
                message=str(exc),
                calls=exc.calls,
            )
            return {"failures": [failure]}
        return {"critiques": [critique]}

    async def collect_critiques(self, state: ArbitrationState) -> dict[str, Any]:
        """Fan-in barrier: runs once, after EVERY critic branch has finished."""
        covered = {c.dimension for c in state.get("critiques", [])}
        missing = [d for d in Dimension if d not in covered]
        if missing:
            logger.warning("no successful critic for: %s", ", ".join(missing))
        return {"missing_dimensions": missing}

    async def detect_disagreements(self, state: ArbitrationState) -> dict[str, Any]:
        """Deterministic comparison of critic reports."""
        report = detect_disagreements(
            state.get("critiques", []), state.get("failures", []), self.config.disagreement
        )
        return {"report": report}

    async def adjudicate(self, state: ArbitrationState) -> dict[str, Any]:
        """Resolve disagreements (Phase 2: union rule; Phase 3: LLM adjudicator)."""
        adjudication = await self.adjudicator.adjudicate(
            AdjudicationInput(
                question=state.get("question"),
                output=state["output"],
                critiques=state.get("critiques", []),
                report=state["report"],
            )
        )
        return {"adjudication": adjudication}

    async def synthesize_verdict(self, state: ArbitrationState) -> dict[str, Any]:
        """Combine everything into the final verdict."""
        verdict = synthesize_verdict(
            config=self.config,
            critiques=state.get("critiques", []),
            failures=state.get("failures", []),
            report=state["report"],
            adjudication=state.get("adjudication"),
        )
        return {"verdict": verdict}

    # ---------------------------------------------------------------- routing

    def dispatch_critics(self, state: ArbitrationState) -> list[Send]:
        """FAN-OUT: one ``Send`` per critic → that many parallel ``run_critic`` branches.

        ``Send(node, payload)`` runs ``node`` with ``payload`` as its input instead of the
        shared state, so each branch knows which critic it is.
        """
        return [
            Send(
                CRITIC_NODE,
                CriticTask(spec=spec, question=state.get("question"), output=state["output"]),
            )
            for spec in self.specs
        ]

    @staticmethod
    def route_after_detection(
        state: ArbitrationState,
    ) -> Literal["adjudicate", "synthesize_verdict"]:
        """CONDITIONAL EDGE: skip the adjudicator when there is nothing to adjudicate.

        * every critic failed → nothing to adjudicate (verdict = failed)
        * unanimous high-confidence pass → short-circuit (saves an expensive LLM call)
        """
        if not state.get("critiques") or state["report"].unanimous_pass:
            return "synthesize_verdict"
        return "adjudicate"
