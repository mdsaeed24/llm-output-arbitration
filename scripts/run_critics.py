"""Live smoke test: run the critics on one output and print their critiques.

Usage:
    python scripts/run_critics.py                          # built-in sample with planted errors
    python scripts/run_critics.py --dimension factual      # one critic only
    python scripts/run_critics.py --output-file a.txt --question "..."

This makes real API calls (a few cents at most). Handy for debugging one critic; the full
pipeline (LangGraph fan-out/fan-in) is scripts/arbitrate.py.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from arbiter.config import Dimension, get_config
from arbiter.critics.critic import critic_specs, run_critic
from arbiter.llm.client import LLMClient, LLMError
from arbiter.llm.telemetry import CallLog
from arbiter.schemas.critique import Critique

SAMPLE_QUESTION = (
    "When did Apollo 11 land on the Moon, which astronauts walked on the surface, "
    "and how long did they stay on the lunar surface?"
)
# Planted: wrong year (factual), unsupported sweeping conclusion (logic),
# the "how long" part is never answered (completeness).
SAMPLE_OUTPUT = (
    "Apollo 11 landed on the Moon on July 20, 1971, in the Sea of Tranquility. Neil Armstrong "
    "and Buzz Aldrin walked on the surface, while Michael Collins remained in lunar orbit aboard "
    "the command module Columbia. Because the landing happened during the Cold War, it proves "
    "that competition is the only force that drives technological progress."
)


def _print_critique(c: Critique) -> None:
    calls = c.calls
    last = calls[-1] if calls else None
    print(f"\n=== {c.critic.upper()} critic · {c.provider}/{c.model_id}")
    print(f"score {c.score}/5 · confidence {c.confidence:.2f}")
    print(f"analysis: {c.analysis}")
    for issue in c.issues:
        mark = "✓ valid" if issue.valid else f"✗ INVALID ({issue.invalid_reason})"
        print(f"  [{issue.severity.value:>6}] {mark} [{issue.match_method}]")
        print(f'           quote:   "{issue.quote}"')
        print(f"           problem: {issue.problem}")
    if not c.issues:
        print("  (no issues)")
    if last is not None:
        print(
            f"telemetry: {len(calls)} call(s) · in {sum(x.input_tokens for x in calls)} tok · "
            f"out {sum(x.output_tokens for x in calls)} tok "
            f"(reasoning {sum(x.reasoning_tokens for x in calls)}) · "
            f"${c.cost_usd:.5f} · {c.latency_s:.1f}s · finish_reason={last.finish_reason}"
        )


async def _main(args: argparse.Namespace) -> int:
    config = get_config()
    output = Path(args.output_file).read_text() if args.output_file else SAMPLE_OUTPUT
    question = args.question if args.output_file else (args.question or SAMPLE_QUESTION)
    specs = [s for s in critic_specs(config) if args.dimension in (None, s.dimension.value)]

    llm = LLMClient(config, call_log=CallLog(config.paths.call_log))
    print(f"QUESTION: {question or '(none)'}\nOUTPUT:   {output}")
    results = await asyncio.gather(
        *[run_critic(llm, s, output=output, question=question) for s in specs],
        return_exceptions=True,
    )
    failures = 0
    for spec, result in zip(specs, results, strict=True):
        if isinstance(result, Critique):
            _print_critique(result)
        elif isinstance(result, LLMError):
            failures += 1
            print(f"\n=== {spec.name.upper()} critic FAILED ({type(result).__name__}): {result}")
        else:
            raise result
    print(
        f"\nTOTAL: {len(llm.call_log.records)} calls · ${llm.call_log.total_cost():.5f} "
        f"· log: {config.paths.call_log}"
    )
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dimension", choices=[d.value for d in Dimension])
    parser.add_argument("--output-file", help="Text file containing the output to evaluate.")
    parser.add_argument("--question", help="Original question (optional).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show retry warnings.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING if args.verbose else logging.ERROR)
    sys.exit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
