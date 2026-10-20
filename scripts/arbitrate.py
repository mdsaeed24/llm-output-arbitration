"""Live end-to-end arbitration through the LangGraph pipeline.

Usage:
    python scripts/arbitrate.py                     # built-in flawed sample
    python scripts/arbitrate.py --clean             # built-in clean sample (short-circuit demo)
    python scripts/arbitrate.py --output-file a.txt --question "..."
    python scripts/arbitrate.py --mermaid           # print the graph diagram and exit
    python scripts/arbitrate.py --json out.json     # also save the full ArbitrationResult

Phase 2 uses the union adjudicator (no LLM); Phase 3 swaps in the LLM adjudicator.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from arbiter.config import get_config
from arbiter.pipeline import Arbiter
from arbiter.schemas.verdict import ArbitrationResult

FLAWED_QUESTION = (
    "When did Apollo 11 land on the Moon, which astronauts walked on the surface, "
    "and how long did they stay on the lunar surface?"
)
FLAWED_OUTPUT = (
    "Apollo 11 landed on the Moon on July 20, 1971, in the Sea of Tranquility. Neil Armstrong "
    "and Buzz Aldrin walked on the surface, while Michael Collins remained in lunar orbit aboard "
    "the command module Columbia. Because the landing happened during the Cold War, it proves "
    "that competition is the only force that drives technological progress."
)
CLEAN_QUESTION = "What is the boiling point of water at sea level, in Celsius and Fahrenheit?"
CLEAN_OUTPUT = (
    "At standard sea-level atmospheric pressure (1 atm), pure water boils at 100 °C, "
    "which is 212 °F."
)


def _print(result: ArbitrationResult) -> None:
    print("\n── CRITICS (ran in parallel)")
    for c in result.critiques:
        print(
            f"  {c.critic:<13} {c.provider}/{c.model_id:<17} score {c.score}/5 "
            f"conf {c.confidence:.2f} · {len(c.valid_issues)} valid / "
            f"{len(c.invalid_issues)} invalid issues · {len(c.verified_claims)} verified claims "
            f"· {c.latency_s:.1f}s"
        )
    for f in result.failures:
        print(f"  {f.critic:<13} FAILED {f.error_type}: {f.message}")

    r = result.report
    print(f"\n── DISAGREEMENT DETECTION · {len(r.groups)} issue groups")
    for g in r.groups:
        tag = "AGREED by " + "+".join(g.critics) if g.corroborated else "only " + g.critics[0]
        quote = result.output[g.span_start : g.span_end]
        print(f'  {g.id} [{g.max_severity.value:>6}] {tag}: "{quote}"')
    for d in r.disagreements:
        print(f"  {d.id} {d.kind.value:<18} {d.group_id}: {d.detail}")
    print(f"  unanimous pass: {r.unanimous_pass}")

    v = result.verdict
    print("\n── VERDICT")
    print(
        f"  status={v.status} passed={v.passed} score={v.overall_score}/10 "
        f"confidence={v.confidence:.2f} short_circuited={v.short_circuited} "
        f"adjudicator={v.adjudicator}"
    )
    if v.low_confidence_dimensions:
        print(f"  low-confidence dimensions: {', '.join(v.low_confidence_dimensions)}")
    for ci in v.confirmed_issues:
        print(f'  ✗ [{ci.severity.value}] ({"+".join(ci.raised_by)}) "{ci.quote}"')
    for df in v.dismissed_flags:
        print(f'  ○ dismissed ({df.rule}) "{df.quote}": {df.reason}')
    print(f"  {v.summary}")
    print(
        f"\n── COST {len(result.calls)} calls · ${result.cost_usd:.5f} · "
        f"wall {result.latency_s:.1f}s (critics run in parallel, so wall ≈ the slowest "
        "critic incl. retry waits, not the sum)"
    )


async def _main(args: argparse.Namespace) -> None:
    arbiter = Arbiter(get_config())
    if args.mermaid:
        print(arbiter.mermaid())
        return
    if args.output_file:
        output, question = Path(args.output_file).read_text(), args.question
    elif args.clean:
        output, question = CLEAN_OUTPUT, CLEAN_QUESTION
    else:
        output, question = FLAWED_OUTPUT, FLAWED_QUESTION
    print(f"QUESTION: {question or '(none)'}\nOUTPUT:   {output}")
    result = await arbiter.arbitrate(output, question)
    _print(result)
    if args.json:
        Path(args.json).write_text(result.model_dump_json(indent=2))
        print(f"saved → {args.json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clean", action="store_true", help="Use the clean built-in sample.")
    parser.add_argument("--output-file")
    parser.add_argument("--question")
    parser.add_argument("--mermaid", action="store_true")
    parser.add_argument("--json", help="Write the full result as JSON to this path.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING if args.verbose else logging.ERROR)
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
