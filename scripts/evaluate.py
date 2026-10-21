"""Phase 5: does the multi-critic system beat a single reviewer?

Runs the full system AND the single-reviewer baseline on every READY case in
data/testset.yaml, matches their findings to the hand-planted answer keys, computes the
metrics and ablations, and writes reports/eval_report.md.

Usage:
    python scripts/evaluate.py --limit 3                 # quick trial on the first 3 cases
    python scripts/evaluate.py                           # all ready cases
    python scripts/evaluate.py --cases id-a,id-b         # specific cases
    python scripts/evaluate.py --resume reports/runs/X   # finish an interrupted run
    python scripts/evaluate.py --from-run reports/runs/X # re-score saved results, no new calls
                                                         # (judge decisions are cached too)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from arbiter.config import get_config
from arbiter.evaluation.matching import OmissionJudge
from arbiter.evaluation.report import render
from arbiter.evaluation.runner import (
    load_runs,
    new_run_dir,
    run_cases,
    score_runs,
    write_metrics_json,
)
from arbiter.llm.client import LLMClient
from arbiter.llm.telemetry import CallLog
from arbiter.schemas.testcase import load_testset


async def _main(args: argparse.Namespace) -> int:
    config = get_config()
    cases = load_testset(args.testset).ready
    if args.cases:
        wanted = [c.strip() for c in args.cases.split(",") if c.strip()]
        unknown = set(wanted) - {c.id for c in cases}
        if unknown:
            sys.exit(f"not ready / not found: {sorted(unknown)}")
        cases = [c for c in cases if c.id in wanted]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        sys.exit("no ready cases to evaluate (mark cases `status: ready` in the test set)")

    run_dir = (
        Path(args.from_run or args.resume) if (args.from_run or args.resume) else new_run_dir()
    )
    llm = LLMClient(config, call_log=CallLog(run_dir / "llm_calls.jsonl"))

    if args.from_run:
        saved = load_runs(run_dir)
        runs = {c.id: saved[c.id] for c in cases if c.id in saved}
        print(f"re-scoring {len(runs)} saved case(s) from {run_dir}")
    else:
        print(f"evaluating {len(cases)} case(s) → {run_dir}")
        runs = await run_cases(config, cases, run_dir, llm=llm)

    judge = OmissionJudge(llm, cache_path=run_dir / "judge_cache.json")
    results = await score_runs(config, cases, runs, judge, run_dir=run_dir)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(results, note=args.note), encoding="utf-8")
    write_metrics_json(results, run_dir / "metrics.json")
    # eval_report.md → eval_metrics.json; eval_report_v2.md → eval_metrics_v2.json
    write_metrics_json(results, out.with_name(out.stem.replace("report", "metrics") + ".json"))

    s, b = results.scores["system"], results.scores["baseline"]
    print(
        f"\nsystem   recall {s.recall:.0%} ({s.caught}/{s.planted_total}) · false flags "
        f"{s.false_flags} · verdict acc {s.accuracy:.0%} · ${s.mean_cost:.4f}/case"
    )
    print(
        f"baseline recall {b.recall:.0%} ({b.caught}/{b.planted_total}) · false flags "
        f"{b.false_flags} · verdict acc {b.accuracy:.0%} · ${b.mean_cost:.4f}/case"
    )
    print(f"\nreport → {out}\nraw results → {run_dir}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--testset", default="data/testset.yaml")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N ready cases.")
    parser.add_argument("--cases", help="Comma-separated case ids.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", help="Run dir of an interrupted run to continue.")
    group.add_argument("--from-run", help="Run dir to re-score without new system calls.")
    parser.add_argument("--out", default="reports/eval_report.md")
    parser.add_argument(
        "--note", help="Paragraph shown under the report title (e.g. what changed)."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show retry warnings.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING if args.verbose else logging.ERROR)
    sys.exit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
