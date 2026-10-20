"""Graph wiring.

START → parse_input ─(Send × N)→ run_critic ×N → collect_critiques → detect_disagreements
      ─(conditional)→ adjudicate → synthesize_verdict → END
                   └──────────────→ synthesize_verdict → END   (short-circuit / all failed)
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from arbiter.graph.nodes import CRITIC_NODE, ArbitrationNodes
from arbiter.graph.state import ArbitrationState


def build_graph(nodes: ArbitrationNodes) -> CompiledStateGraph[Any]:
    """Assemble and compile the arbitration graph."""
    g: StateGraph[Any] = StateGraph(ArbitrationState)

    g.add_node("parse_input", nodes.parse_input)
    # input is a Send payload (CriticTask), not the shared state — LangGraph allows this,
    # its type stubs do not express it
    g.add_node(CRITIC_NODE, nodes.run_critic)  # type: ignore[arg-type]
    g.add_node("collect_critiques", nodes.collect_critiques)
    g.add_node("detect_disagreements", nodes.detect_disagreements)
    g.add_node("adjudicate", nodes.adjudicate)
    g.add_node("synthesize_verdict", nodes.synthesize_verdict)

    g.add_edge(START, "parse_input")
    # fan-out: the routing function returns a list of Send objects → parallel branches
    g.add_conditional_edges("parse_input", nodes.dispatch_critics, [CRITIC_NODE])
    # fan-in: collect_critiques waits until every branch of the step has completed
    g.add_edge(CRITIC_NODE, "collect_critiques")
    g.add_edge("collect_critiques", "detect_disagreements")
    g.add_conditional_edges(
        "detect_disagreements",
        nodes.route_after_detection,
        ["adjudicate", "synthesize_verdict"],
    )
    g.add_edge("adjudicate", "synthesize_verdict")
    g.add_edge("synthesize_verdict", END)
    return g.compile()
