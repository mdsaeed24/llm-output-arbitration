"""Case builder: clean drafting only, YAML append round-trips, coverage report."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from arbiter.casebuilder import (
    SYSTEM_PROMPT,
    Draft,
    add_draft,
    append_case,
    coverage,
    draft_answer,
    format_case,
    format_coverage,
    render_case,
    slugify,
    unique_id,
)
from arbiter.llm.client import LLMClient
from arbiter.schemas.testcase import EvalSet, find_all, load_testset
from tests.conftest import REPO_ROOT, FakeOpenAI, make_completion

MakeLLM = Callable[..., LLMClient]


@pytest.fixture
def testset_path(tmp_path: Path) -> Path:
    """A private copy of the real data/testset.yaml (comments, examples and all)."""
    path = tmp_path / "testset.yaml"
    shutil.copy(REPO_ROOT / "data" / "testset.yaml", path)
    return path


def _draft(answer: str, question: str = "Why is the sky blue?") -> Draft:
    return Draft(
        question=question, answer=answer, question_parts=["why blue"], model_id="deepseek-v4-pro"
    )


def test_prompt_only_asks_for_a_correct_answer() -> None:
    lowered = SYSTEM_PROMPT.lower()
    assert "accurate" in lowered and "every part of the" in lowered
    for word in ("plant", "error", "mistake", "flaw", "wrong"):
        assert word not in lowered, f"drafting prompt must not mention {word!r}"


async def test_draft_answer_uses_case_builder_model(make_llm: MakeLLM) -> None:
    fake = FakeOpenAI(make_completion({"question_parts": ["a", "b"], "answer": "  The answer.  "}))
    draft = await draft_answer(make_llm(deepseek=fake), "  Q?  ", length="long")
    assert draft.answer == "The answer." and draft.question == "Q?"
    assert draft.question_parts == ["a", "b"] and draft.model_id == "deepseek-v4-pro"
    req = fake.requests[0]
    assert req["model"] == "deepseek-v4-pro"
    assert "250-400 words" in req["messages"][0]["content"]


@pytest.mark.parametrize(
    "answer",
    [
        "One paragraph: with a colon, \"quotes\", 'single', # hash, 100 °C, and — a dash.",
        "First paragraph is here.\n\nSecond paragraph: - starts with text.\n\nThird.",
        "Line one of a list:\n1. alpha\n2. beta: gamma",  # single newlines → literal block
        "- starts with a dash and has a very long sentence " * 6,
    ],
)
def test_rendered_case_round_trips_exactly(testset_path: Path, answer: str) -> None:
    case = add_draft(testset_path, _draft(answer))
    expected = answer.strip()
    if "\n" not in expected.replace("\n\n", ""):  # prose: whitespace runs normalised
        expected = "\n".join(" ".join(p.split()) for p in expected.split("\n\n"))
    assert case.output == expected
    assert case.status == "draft" and case.planted == []


def test_append_keeps_comments_and_existing_cases(testset_path: Path) -> None:
    before = testset_path.read_text()
    add_draft(testset_path, _draft("A clean answer."), tags=["domain-science"])
    after = testset_path.read_text()
    assert after.startswith(before.rstrip("\n"))
    assert "# COVERAGE CHECKLIST" in after
    ts = load_testset(testset_path)
    assert [c.id for c in ts.cases][-1] == "why-is-the-sky-blue"
    assert ts.cases[-1].tags == ["domain-science"]
    assert "why blue" in after  # question parts written as a comment


def test_ids_are_slugged_and_deduplicated(testset_path: Path) -> None:
    assert slugify("When did Apollo 11 land on the Moon, exactly?") == "when-did-apollo-11-land-on"
    assert unique_id("x", {"x", "x-2"}) == "x-3"
    a = add_draft(testset_path, _draft("A."))
    b = add_draft(testset_path, _draft("B."))
    assert (a.id, b.id) == ("why-is-the-sky-blue", "why-is-the-sky-blue-2")
    with pytest.raises(ValueError, match="already exists"):
        add_draft(testset_path, _draft("C."), case_id=a.id)


def test_append_rolls_back_if_result_is_invalid(testset_path: Path) -> None:
    before = testset_path.read_text()
    bad = render_case(
        case_id="example-sky-colour",  # duplicate id → whole file invalid
        question="Q",
        output="O",
        question_parts=[],
        tags=[],
        note="n",
    )
    with pytest.raises(ValidationError, match="duplicate case ids"):
        append_case(testset_path, bad)
    assert testset_path.read_text() == before


def test_find_all_ignores_whitespace_differences_only() -> None:
    text = "Because red light scatters\nthe most, sunsets"
    [(start, end)] = find_all(text, "red light scatters the most")
    assert text[start:end] == "red light scatters\nthe most"
    assert find_all(text, "red light scatters the MOST") == []
    assert find_all(text, "   ") == []


def _cases(*specs: tuple[str, str, list[str], list[dict[str, str]]]) -> EvalSet:
    return EvalSet.model_validate(
        {
            "cases": [
                {
                    "id": cid,
                    "status": status,
                    "tags": tags,
                    "question": "Q?",
                    "output": "The year 1971 was long. It proves everything.",
                    "planted": planted,
                }
                for cid, status, tags, planted in specs
            ]
        }
    )


FACT = {"type": "factual", "span": "1971", "description": "x"}
LOGIC = {"type": "logic", "span": "It proves everything.", "description": "x"}
OMIT = {"type": "completeness", "missing": "x"}


def test_coverage_counts_ready_cases_only() -> None:
    cov = coverage(
        _cases(
            ("a", "ready", ["wrong-date", "domain-history"], [FACT]),
            ("b", "ready", ["mixed", "contradiction"], [FACT, LOGIC, OMIT]),
            ("c", "ready", ["clean"], []),
            ("d", "draft", ["wrong-name"], [FACT]),
        )
    )
    assert (cov.total, cov.ready, cov.drafts, cov.clean_ready) == (4, 3, 1, 1)
    assert cov.clean_share == pytest.approx(1 / 3)
    assert cov.planted_by_type == {"factual": 2, "logic": 1, "completeness": 1}
    assert cov.tag_counts["wrong-name"] == 0  # draft not counted
    assert cov.domains == {"domain-history": 1}
    assert cov.warnings == []
    report = format_coverage(cov)
    assert "✓ wrong-date 1" in report and "✗ circular 0" in report
    assert "clean: 1/3 ready = 33%" in report


def test_coverage_warns_about_inconsistent_tags() -> None:
    cov = coverage(
        _cases(
            ("a", "ready", ["clean"], [FACT]),
            ("b", "ready", [], []),
            ("c", "ready", ["mixed"], [FACT]),
        )
    )
    assert len(cov.warnings) == 3


def test_format_case_marks_spans_and_anchors() -> None:
    ts = load_testset(REPO_ROOT / "data" / "testset.yaml")
    sky = next(c for c in ts.cases if c.id == "example-sky-colour")
    shown = format_case(sky)
    assert "⟦F1: at around 650 nanometers⟧" in shown
    assert "⟦L2: Because red light scatters the most⟧" in shown
    roth = next(c for c in ts.cases if c.id == "example-roth-vs-traditional")
    assert "⟦C1 anchor: The right account" in format_case(roth)
    clean = next(c for c in ts.cases if c.id == "example-list-vs-tuple")
    assert "CLEAN" in format_case(clean)


def test_repo_examples_are_drafts_awaiting_review() -> None:
    ts = load_testset(REPO_ROOT / "data" / "testset.yaml")
    assert {c.id for c in ts.drafts} >= {
        "example-sky-colour",
        "example-roth-vs-traditional",
        "example-list-vs-tuple",
    }
    tuple_case = next(c for c in ts.cases if c.id == "example-list-vs-tuple")
    assert "\nChoose a tuple" in tuple_case.output  # folded scalar kept the paragraph break
