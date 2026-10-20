"""Public entry point: ``await Arbiter(config).arbitrate(output, question)``.

API, scripts and evaluation all go through this class, so they share one graph, one LLM
client and therefore one concurrency limit.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from arbiter.adjudication.base import Adjudicator
from arbiter.adjudication.union import UnionAdjudicator
from arbiter.config import AppConfig
from arbiter.critics.critic import CriticSpec, critic_specs
from arbiter.graph.build import build_graph
from arbiter.graph.nodes import ArbitrationNodes
from arbiter.llm.client import LLMClient
from arbiter.llm.telemetry import CallLog
from arbiter.schemas.verdict import ArbitrationResult


class Arbiter:
    """Runs the full critic → disagreement → adjudication → verdict graph."""

    def __init__(
        self,
        config: AppConfig,
        *,
        llm: LLMClient | None = None,
        adjudicator: Adjudicator | None = None,
        specs: Sequence[CriticSpec] | None = None,
    ) -> None:
        self.config = config
        self.llm = llm or LLMClient(config, call_log=CallLog(config.paths.call_log))
        self.nodes = ArbitrationNodes(
            config,
            self.llm,
            adjudicator or UnionAdjudicator(),
            specs if specs is not None else critic_specs(config),
        )
        self.graph: CompiledStateGraph[Any] = build_graph(self.nodes)

    async def arbitrate(
        self, output: str, question: str | None = None, *, request_id: str | None = None
    ) -> ArbitrationResult:
        """Evaluate ``output`` (optionally against ``question``) and return the full record."""
        start = time.perf_counter()
        state = await self.graph.ainvoke(
            {"output": output, "question": question, "request_id": request_id or ""}
        )
        return ArbitrationResult(
            id=state["request_id"],
            question=state["question"],
            output=state["output"],
            critiques=state.get("critiques", []),
            failures=state.get("failures", []),
            report=state["report"],
            adjudication=state.get("adjudication"),
            verdict=state["verdict"],
            latency_s=time.perf_counter() - start,
        )

    def mermaid(self) -> str:
        """Mermaid source for the compiled graph (used in the README)."""
        return self.graph.get_graph().draw_mermaid()
