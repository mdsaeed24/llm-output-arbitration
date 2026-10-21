"""Evaluation cases and their hand-written answer keys (``data/testset.yaml``).

Two kinds of planted issue, because they are matched differently in Phase 5:

* ``PlantedSpanIssue`` (factual, logic): the error lives in specific text, so ``span`` is an
  exact substring of the output and found issues are matched by span overlap.
* ``PlantedOmission`` (completeness): the problem is something *absent*, which has no span.
  ``missing`` describes what is absent; found completeness issues are matched to it by an LLM
  judge. ``anchor`` is optional and used only for highlighting in the UI.

The loader is strict on purpose: a span that is not in the output, or appears more than once,
is an error in the answer key and would silently corrupt every metric.

(Classes are named ``Eval*`` rather than ``Test*`` so pytest doesn't try to collect them.)
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from arbiter.config import Dimension


def find_all(text: str, needle: str) -> list[tuple[int, int]]:
    """Exact-character matches of ``needle`` in ``text``, where any run of whitespace in the
    needle matches any run of whitespace in the text.

    So a span copied across a wrapped YAML line ("scatters the most" vs "scatters\nthe most")
    still matches. Everything else — letters, digits, punctuation, case — must be identical.
    """
    words = needle.split()
    if not words:
        return []
    pattern = re.compile(r"\s+".join(re.escape(w) for w in words))
    return [m.span() for m in pattern.finditer(text)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlantedSpanIssue(_Strict):
    """A factual or logic error located at an exact span of the output."""

    type: Literal["factual", "logic"]
    span: str = Field(min_length=1, description="Exact substring of the output (unique).")
    description: str = Field(min_length=1, description="What is wrong (for you and the report).")

    @property
    def dimension(self) -> Dimension:
        """The issue type as a :class:`Dimension`."""
        return Dimension(self.type)


class PlantedOmission(_Strict):
    """Something the output fails to provide (a skipped part, a missed point)."""

    type: Literal["completeness"]
    missing: str = Field(min_length=1, description="WHAT is missing, e.g. 'course duration'.")
    anchor: str | None = Field(
        None, description="Optional sentence near the gap; UI highlighting only, never matching."
    )

    @property
    def dimension(self) -> Dimension:
        """Always :attr:`Dimension.COMPLETENESS`."""
        return Dimension.COMPLETENESS


PlantedIssue = Annotated[PlantedSpanIssue | PlantedOmission, Field(discriminator="type")]


class EvalCase(_Strict):
    """One test case: question, output to evaluate, and its answer key."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    question: str | None = None
    output: str = Field(min_length=1)
    planted: list[PlantedIssue] = Field(
        default_factory=list, description="Answer key. Empty list = clean case."
    )
    status: Literal["draft", "ready"] = Field(
        "draft",
        description="Only 'ready' cases are evaluated. Mark a case ready once YOU have checked "
        "the output and its answer key — an unreviewed draft might hide a real error that "
        "would be scored as a false flag.",
    )
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("question", "output")
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        # YAML block scalars add a trailing newline; offsets are computed on stripped text.
        return v.strip() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _spans_exist_in_output(self) -> EvalCase:
        for n, issue in enumerate(self.planted, start=1):
            if isinstance(issue, PlantedSpanIssue):
                count = len(find_all(self.output, issue.span))
                if count != 1:
                    problem = "not found in" if count == 0 else f"appears {count} times in"
                    hint = "copy it exactly" if count == 0 else "make it longer so it is unique"
                    raise ValueError(
                        f"case '{self.id}' planted #{n}: span {issue.span!r} {problem} "
                        f"the output ({hint})"
                    )
            elif issue.anchor is not None and not find_all(self.output, issue.anchor):
                raise ValueError(
                    f"case '{self.id}' planted #{n}: anchor {issue.anchor!r} not found in output"
                )
        return self

    @property
    def is_clean(self) -> bool:
        """True when the answer key lists no planted issues."""
        return not self.planted

    def locate(self, issue: PlantedIssue) -> tuple[int, int] | None:
        """Character span of a planted span issue (or an omission's anchor) in the output."""
        text = issue.span if isinstance(issue, PlantedSpanIssue) else issue.anchor
        if text is None:
            return None
        return find_all(self.output, text)[0]


class EvalSet(_Strict):
    """The whole test set."""

    version: Literal[1] = 1
    cases: list[EvalCase] = Field(default_factory=list)

    @property
    def ready(self) -> list[EvalCase]:
        """Cases a human has marked ready — the only ones the evaluation runs on."""
        return [c for c in self.cases if c.status == "ready"]

    @property
    def drafts(self) -> list[EvalCase]:
        """Cases still awaiting review."""
        return [c for c in self.cases if c.status == "draft"]

    @model_validator(mode="after")
    def _unique_ids(self) -> EvalSet:
        counts = Counter(c.id for c in self.cases)
        dupes = [case_id for case_id, n in counts.items() if n > 1]
        if dupes:
            raise ValueError(f"duplicate case ids: {sorted(dupes)}")
        return self


def load_testset(path: Path | str = Path("data/testset.yaml")) -> EvalSet:
    """Load and validate the test set (raises ``ValidationError`` with the offending case)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return EvalSet.model_validate(raw)
