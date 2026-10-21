"""Test-set helper: draft CLEAN answers, then check coverage of your hand-written cases.

Usage:
    # 1. Draft a clean answer to your question (appended to data/testset.yaml as a draft)
    python scripts/case_builder.py draft --question "Why do we have leap years, and ..."
    python scripts/case_builder.py draft --question "..." --length long --tags domain-history
    python scripts/case_builder.py draft --questions-file my_questions.txt   # one per line

    # 2. Edit the output in data/testset.yaml to plant errors, record each in `planted`,
    #    then set `status: ready`.

    # 3. Check your work
    python scripts/case_builder.py check                 # coverage vs the checklist
    python scripts/case_builder.py check --show my-id    # one case with planted spans marked

The model only ever drafts a correct answer; it is never asked to choose or plant errors.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from arbiter.casebuilder import (
    LENGTHS,
    add_draft,
    coverage,
    draft_answer,
    format_case,
    format_coverage,
)
from arbiter.config import get_config
from arbiter.llm.client import LLMClient, LLMError
from arbiter.llm.telemetry import CallLog
from arbiter.schemas.testcase import load_testset


async def _draft(args: argparse.Namespace) -> int:
    path = Path(args.testset)
    if args.questions_file:
        lines = Path(args.questions_file).read_text(encoding="utf-8").splitlines()
        questions = [q.strip() for q in lines if q.strip() and not q.lstrip().startswith("#")]
    else:
        questions = [args.question]
    if args.limit:
        questions = questions[: args.limit]
    if args.id and len(questions) > 1:
        sys.exit("--id only works with a single --question")

    config = get_config()
    llm = LLMClient(config, call_log=CallLog(config.paths.call_log))
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
    results = await asyncio.gather(
        *[draft_answer(llm, q, length=args.length) for q in questions], return_exceptions=True
    )
    failed = 0
    for question, result in zip(questions, results, strict=True):
        if isinstance(result, LLMError):
            failed += 1
            print(f"✗ FAILED for {question[:60]!r}: {result}")
            continue
        if isinstance(result, BaseException):
            raise result
        print(f"\nQ: {question}\nparts: {'; '.join(result.question_parts)}\n\n{result.answer}\n")
        if args.dry_run:
            continue
        case = add_draft(path, result, case_id=args.id, tags=tags)  # appended in input order
        print(f"→ appended as '{case.id}' (status: draft) · ${result.cost_usd:.5f}")
    print(f"\ntotal ${llm.call_log.total_cost():.5f} · {len(questions) - failed} drafted")
    return 1 if failed else 0


def _check(args: argparse.Namespace) -> int:
    testset = load_testset(args.testset)  # raises with the offending case if invalid
    if args.show:
        case = next((c for c in testset.cases if c.id == args.show), None)
        if case is None:
            sys.exit(f"no case with id '{args.show}'")
        print(format_case(case))
        return 0
    print(f"{args.testset}: valid ✓\n")
    print(format_coverage(coverage(testset)))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--testset", default="data/testset.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("draft", help="Draft a CLEAN answer and append it as a draft case.")
    src = d.add_mutually_exclusive_group(required=True)
    src.add_argument("--question")
    src.add_argument("--questions-file", help="Text file, one question per line (# = comment).")
    d.add_argument("--length", choices=list(LENGTHS), default="medium")
    d.add_argument("--tags", help="Comma-separated, e.g. domain-history,wrong-date")
    d.add_argument("--id", help="Case id (default: derived from the question).")
    d.add_argument("--limit", type=int, help="Draft at most N questions from the file.")
    d.add_argument("--dry-run", action="store_true", help="Print the draft, don't save it.")

    c = sub.add_parser("check", help="Validate the test set and report checklist coverage.")
    c.add_argument("--show", metavar="ID", help="Print one case with its planted spans marked.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    if args.command == "draft":
        sys.exit(asyncio.run(_draft(args)))
    sys.exit(_check(args))


if __name__ == "__main__":
    main()
