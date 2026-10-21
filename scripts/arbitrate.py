"""Live end-to-end arbitration through the LangGraph pipeline.

Usage:
    python scripts/arbitrate.py                       # built-in flawed sample
    python scripts/arbitrate.py --sample clean        # clean sample (short-circuit demo)
    python scripts/arbitrate.py --sample tricky       # true-but-surprising facts (false-flag bait)
    python scripts/arbitrate.py --baseline            # also run the single-reviewer baseline
    python scripts/arbitrate.py --union               # skip the LLM adjudicator (ablation)
    python scripts/arbitrate.py --output-file a.txt --question "..."
    python scripts/arbitrate.py --mermaid             # print the graph diagram and exit
    python scripts/arbitrate.py --json out.json       # also save the full ArbitrationResult
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from arbiter.adjudication.union import UnionAdjudicator
from arbiter.baseline import BaselineResult, SingleReviewer
from arbiter.config import get_config
from arbiter.pipeline import Arbiter
from arbiter.schemas.verdict import ArbitrationResult, Verdict

SAMPLES: dict[str, tuple[str, str]] = {
    # planted: wrong year (factual), unsupported conclusion (logic), duration missing
    "flawed": (
        "When did Apollo 11 land on the Moon, which astronauts walked on the surface, "
        "and how long did they stay on the lunar surface?",
        "Apollo 11 landed on the Moon on July 20, 1971, in the Sea of Tranquility. Neil "
        "Armstrong and Buzz Aldrin walked on the surface, while Michael Collins remained in lunar "
        "orbit aboard the command module Columbia. Because the landing happened during the Cold "
        "War, it proves that competition is the only force that drives technological progress.",
    ),
    "clean": (
        "What is the boiling point of water at sea level, in Celsius and Fahrenheit?",
        "At standard sea-level atmospheric pressure (1 atm), pure water boils at 100 °C, "
        "which is 212 °F.",
    ),
    # every claim is TRUE but sounds wrong — tempts critics into false flags
    "tricky": (
        "Give two surprising but true facts about historical timelines, and briefly explain why "
        "each one is true.",
        "First, the University of Oxford is older than the Aztec Empire: teaching was taking "
        "place at Oxford by 1096, while the Aztec capital Tenochtitlan was founded around 1325. "
        "Second, Cleopatra lived closer in time to the Moon landing than to the construction of "
        "the Great Pyramid of Giza. The pyramid was built around 2560 BC, Cleopatra died in 30 "
        "BC, and Apollo 11 landed in 1969 — a gap of roughly 2,000 years versus about 2,500.",
    ),
}


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

    _print_verdict(result.verdict, "VERDICT (multi-critic system)")
    print(
        f"\n── COST {len(result.calls)} calls · ${result.cost_usd:.5f} · "
        f"wall {result.latency_s:.1f}s (critics run in parallel, so wall ≈ the slowest "
        "critic incl. retry waits, not the sum)"
    )


def _print_verdict(v: Verdict, title: str) -> None:
    print(f"\n── {title}")
    print(
        f"  status={v.status} passed={v.passed} score={v.overall_score}/10 "
        f"confidence={v.confidence:.2f} short_circuited={v.short_circuited} "
        f"adjudicator={v.adjudicator}"
    )
    if v.low_confidence_dimensions:
        print(f"  low-confidence dimensions: {', '.join(v.low_confidence_dimensions)}")
    for ci in v.confirmed_issues:
        print(f'  ✗ CONFIRMED [{ci.severity.value}] ({"+".join(ci.raised_by)}) "{ci.quote}"')
        print(f"      why: {ci.evidence}")
    for df in v.dismissed_flags:
        print(f'  ○ DISMISSED ({df.rule}; raised by {"+".join(df.raised_by)}) "{df.quote}"')
        print(f"      why: {df.reason}")
    print(f"  summary: {v.summary}")


def _print_baseline(b: BaselineResult) -> None:
    _print_verdict(b.verdict, f"BASELINE (single reviewer: {b.model_id})")
    print(f"  types: {', '.join(i.dimension.value for i in b.issues if i.valid) or '-'}")
    print(f"── BASELINE COST {len(b.calls)} call(s) · ${b.cost_usd:.5f} · {b.latency_s:.1f}s")


async def _main(args: argparse.Namespace) -> None:
    config = get_config()
    arbiter = Arbiter(config, adjudicator=UnionAdjudicator() if args.union else None)
    if args.mermaid:
        print(arbiter.mermaid())
        return
    if args.output_file:
        output, question = Path(args.output_file).read_text(), args.question
    else:
        question, output = SAMPLES[args.sample]
    print(f"QUESTION: {question or '(none)'}\nOUTPUT:   {output}")
    if args.baseline:
        # same LLM client → same concurrency limit and call log
        result, baseline = await asyncio.gather(
            arbiter.arbitrate(output, question),
            SingleReviewer(arbiter.llm).review(output, question),
        )
    else:
        result, baseline = await arbiter.arbitrate(output, question), None
    _print(result)
    if baseline is not None:
        _print_baseline(baseline)
    if args.json:
        Path(args.json).write_text(result.model_dump_json(indent=2))
        print(f"saved → {args.json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", choices=list(SAMPLES), default="flawed")
    parser.add_argument("--baseline", action="store_true", help="Also run the baseline.")
    parser.add_argument("--union", action="store_true", help="Union instead of LLM adjudicator.")
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
