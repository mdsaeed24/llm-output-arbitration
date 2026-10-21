"""Helpers for building the test set by hand.

The LLM's ONLY job here is to draft a CLEAN, high-quality answer to a question you provide.
It is never asked to suggest, plant or describe errors — you edit the draft yourself and
record each planted problem in the answer key. That keeps the ground truth human-authored:
if the model chose the errors, it would also be choosing what is "easy to detect".

Drafts are appended to ``data/testset.yaml`` as plain text (so your comments survive) with
``status: draft``. A case only enters the evaluation after you set ``status: ready``.
"""

from __future__ import annotations

import re
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from arbiter.llm.client import LLMClient
from arbiter.llm.telemetry import CallRecord
from arbiter.schemas.testcase import EvalCase, EvalSet, PlantedSpanIssue, load_testset

Length = Literal["short", "medium", "long"]
LENGTHS: dict[Length, str] = {
    "short": "about 60-100 words",
    "medium": "about 120-200 words",
    "long": "about 250-400 words",
}

RECOMMENDED_TAGS: dict[str, list[str]] = {
    "factual": ["wrong-date", "wrong-number", "wrong-name"],
    "logic": ["contradiction", "unsupported-conclusion", "circular"],
    "completeness": ["skipped-part", "missed-point", "ignored-constraint"],
    "modifiers": ["subtle", "mixed"],
    "clean": ["clean"],
}

SYSTEM_PROMPT = """\
You write reference answers for an evaluation test set. Write the best possible answer to the
user's question: factually accurate, logically sound, and complete — address every part of the
question and respect any constraint it states (length, format, examples).

- Plain prose paragraphs: no markdown headings, bullet lists or bold text.
- Length: {length}, unless the question itself asks for a different length.
- State only facts you are confident are correct. Prefer well-established facts over obscure
  or disputed ones; never invent statistics, quotations or sources.
- Be direct: answer what was asked, without filler or excessive hedging.

First list "question_parts": every distinct thing the question asks for. Then write "answer"."""


class DraftAnswer(BaseModel):
    """What the case builder asks the model for."""

    question_parts: list[str] = Field(
        description="Each distinct thing the question asks for (helps you plant omissions)."
    )
    answer: str = Field(min_length=1)


@dataclass
class Draft:
    """A drafted answer plus telemetry."""

    question: str
    answer: str
    question_parts: list[str]
    model_id: str
    calls: list[CallRecord] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        """Spend for this draft."""
        return sum(c.cost_usd for c in self.calls)


async def draft_answer(llm: LLMClient, question: str, *, length: Length = "medium") -> Draft:
    """Ask the ``case_builder`` model for a clean answer to ``question``."""
    model_key = llm.config.roles.case_builder
    result = await llm.structured(
        role="case_builder",
        model_key=model_key,
        system=SYSTEM_PROMPT.format(length=LENGTHS[length]),
        user=f"Question:\n{question.strip()}",
        schema=DraftAnswer,
    )
    return Draft(
        question=question.strip(),
        answer=result.value.answer.strip(),
        question_parts=result.value.question_parts,
        model_id=llm.config.model(model_key).model_id,
        calls=result.calls,
    )


# --------------------------------------------------------------------------- YAML writing


def slugify(text: str, max_words: int = 6) -> str:
    """'When did Apollo 11 land?' → 'when-did-apollo-11-land'."""
    words = re.findall(r"[a-z0-9]+", text.lower())[:max_words]
    return "-".join(words) or "case"


def unique_id(base: str, existing: set[str]) -> str:
    """``base``, or ``base-2``, ``base-3`` … if taken."""
    if base not in existing:
        return base
    n = 2
    while f"{base}-{n}" in existing:
        n += 1
    return f"{base}-{n}"


def _yaml_text_block(text: str, indent: str) -> str:
    """Render ``text`` as a YAML block scalar.

    Folded (``>``) when the text is prose paragraphs — wrapped for readability, re-joined with
    spaces on load. Literal (``|``) when it contains single line breaks (lists, code) that
    folding would destroy.
    """
    text = text.strip().replace("\t", "    ")
    paragraphs = re.split(r"\n\s*\n", text)
    if any("\n" in p for p in paragraphs):
        body = "\n".join(f"{indent}{line}" if line.strip() else "" for line in text.splitlines())
        return f"|\n{body}"
    wrapped = [
        textwrap.fill(
            " ".join(p.split()),
            width=88,
            initial_indent=indent,
            subsequent_indent=indent,
            break_long_words=False,
            break_on_hyphens=False,
        )
        for p in paragraphs
    ]
    return ">\n" + "\n\n".join(wrapped)


def render_case(
    *,
    case_id: str,
    question: str,
    output: str,
    question_parts: list[str],
    tags: list[str],
    note: str,
) -> str:
    """YAML text for one draft case, ready to append under ``cases:``."""
    indent = "      "
    parts = "\n".join(f"    #   - {p}" for p in question_parts) or "    #   (none listed)"
    tag_list = ", ".join(tags)
    return f"""
  - id: {case_id}
    status: draft   # set to `ready` after you've planted/recorded errors (or verified it clean)
    tags: [{tag_list}]
    question: {_yaml_text_block(question, indent)}
    output: {_yaml_text_block(output, indent)}
    # Question parts the drafting model identified (useful for planting omissions):
{parts}
    planted: []     # TODO: edit `output` to plant errors, record each one here
    notes: {_yaml_quote(note)}
"""


def _yaml_quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def append_case(path: Path, case_yaml: str) -> EvalCase:
    """Append ``case_yaml`` to the test set, re-validate the whole file, roll back on error."""
    original = path.read_text(encoding="utf-8")
    updated = original.rstrip("\n") + "\n" + case_yaml
    path.write_text(updated, encoding="utf-8")
    try:
        testset = load_testset(path)
    except Exception:
        path.write_text(original, encoding="utf-8")
        raise
    return testset.cases[-1]


def add_draft(
    path: Path, draft: Draft, *, case_id: str | None = None, tags: list[str] | None = None
) -> EvalCase:
    """Render and append a drafted case with a unique id."""
    existing = {c.id for c in load_testset(path).cases}
    cid = case_id or unique_id(slugify(draft.question), existing)
    if cid in existing:
        raise ValueError(f"case id '{cid}' already exists")
    yaml_text = render_case(
        case_id=cid,
        question=draft.question,
        output=draft.answer,
        question_parts=draft.question_parts,
        tags=tags or [],
        note=f"Drafted clean by case_builder ({draft.model_id}) on {date.today().isoformat()}.",
    )
    return append_case(path, yaml_text)


# --------------------------------------------------------------------------- coverage check


@dataclass
class Coverage:
    """How the test set is shaping up against the checklist."""

    total: int
    ready: int
    drafts: int
    clean_ready: int
    planted_by_type: Counter[str]
    tag_counts: Counter[str]
    domains: Counter[str]
    warnings: list[str]

    @property
    def clean_share(self) -> float:
        """Fraction of ready cases that are clean."""
        return self.clean_ready / self.ready if self.ready else 0.0


def coverage(testset: EvalSet) -> Coverage:
    """Summarise READY cases (drafts are only counted) and flag inconsistencies."""
    ready = testset.ready
    tags: Counter[str] = Counter(t for c in ready for t in c.tags)
    warnings = []
    for c in testset.cases:
        if "clean" in c.tags and not c.is_clean:
            warnings.append(f"{c.id}: tagged 'clean' but has planted issues")
        if c.is_clean and c.status == "ready" and "clean" not in c.tags:
            warnings.append(f"{c.id}: no planted issues but not tagged 'clean' (intended?)")
        types = {p.type for p in c.planted}
        if "mixed" in c.tags and len(types) < 2:
            warnings.append(f"{c.id}: tagged 'mixed' but has {len(types)} issue type(s)")
        if c.question is None:
            warnings.append(f"{c.id}: no question — completeness can't really be judged")
    return Coverage(
        total=len(testset.cases),
        ready=len(ready),
        drafts=len(testset.drafts),
        clean_ready=sum(c.is_clean for c in ready),
        planted_by_type=Counter(p.type for c in ready for p in c.planted),
        tag_counts=tags,
        domains=Counter(t for t in tags.elements() if t.startswith("domain-")),
        warnings=warnings,
    )


def format_coverage(cov: Coverage, *, target: int = 40, clean_target: float = 0.25) -> str:
    """Human-readable coverage report for ``case_builder.py check``."""
    lines = [
        f"cases: {cov.total} total · {cov.ready} ready · {cov.drafts} draft "
        f"(target ~{target} ready)",
        f"clean: {cov.clean_ready}/{cov.ready} ready = {cov.clean_share:.0%} "
        f"(target ~{clean_target:.0%} ≈ {round(target * clean_target)})",
        "planted issues (ready cases): "
        + (", ".join(f"{t} {n}" for t, n in sorted(cov.planted_by_type.items())) or "none"),
        "",
        "checklist tags (ready cases):",
    ]
    for group, names in RECOMMENDED_TAGS.items():
        cells = [f"{'✓' if cov.tag_counts[n] else '✗'} {n} {cov.tag_counts[n]}" for n in names]
        lines.append(f"  {group:<13} " + "   ".join(cells))
    lines.append(
        "  domains       "
        + (", ".join(f"{d} {n}" for d, n in sorted(cov.domains.items())) or "none yet")
    )
    if cov.warnings:
        lines += ["", "warnings:", *[f"  ! {w}" for w in cov.warnings]]
    return "\n".join(lines)


def format_case(case: EvalCase) -> str:
    """Show a case with planted spans bracketed — for proof-reading your answer key."""
    marks: list[tuple[int, int, str]] = []
    for n, issue in enumerate(case.planted, start=1):
        loc = case.locate(issue)
        if loc is not None:
            label = f"{issue.type[0].upper()}{n}"
            kind = "" if isinstance(issue, PlantedSpanIssue) else " anchor"
            marks.append((loc[0], loc[1], f"{label}{kind}"))
    text = case.output
    for start, end, label in sorted(marks, reverse=True):
        text = f"{text[:start]}⟦{label}: {text[start:end]}⟧{text[end:]}"
    lines = [
        f"== {case.id} [{case.status}] tags: {', '.join(case.tags) or '-'}",
        f"Q: {case.question or '(none)'}",
        "",
        text,
        "",
    ]
    if case.is_clean:
        lines.append("answer key: CLEAN (any flag = false flag)")
    for n, issue in enumerate(case.planted, start=1):
        detail = issue.description if isinstance(issue, PlantedSpanIssue) else issue.missing
        lines.append(f"  {issue.type[0].upper()}{n} {issue.type}: {detail}")
    return "\n".join(lines)
