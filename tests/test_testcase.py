"""Answer-key format: span issues vs omissions, strict validation on load."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from arbiter.config import Dimension
from arbiter.schemas.testcase import (
    EvalCase,
    EvalSet,
    PlantedOmission,
    PlantedSpanIssue,
    load_testset,
)

REPO_TESTSET = Path(__file__).resolve().parents[1] / "data" / "testset.yaml"

OUTPUT = (
    "Apollo 11 landed on the Moon on July 20, 1971. Neil Armstrong and Buzz Aldrin walked "
    "on the surface. Because it happened in the Cold War, competition is the only driver of "
    "progress."
)


def _case(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "apollo-01",
        "question": "When did Apollo 11 land, who walked on the Moon, and for how long?",
        "output": OUTPUT + "\n",
        "planted": [
            {"type": "factual", "span": "July 20, 1971", "description": "1969, not 1971."},
            {
                "type": "logic",
                "span": "competition is the only driver of progress",
                "description": "Unsupported generalisation.",
            },
            {
                "type": "completeness",
                "missing": "How long they stayed on the lunar surface.",
                "anchor": "Neil Armstrong and Buzz Aldrin walked on the surface.",
            },
        ],
    }
    base.update(overrides)
    return base


def test_repo_testset_loads() -> None:
    assert isinstance(load_testset(REPO_TESTSET), EvalSet)


def test_discriminated_union_builds_the_right_issue_kinds() -> None:
    case = EvalCase.model_validate(_case())
    factual, logic, omission = case.planted
    assert isinstance(factual, PlantedSpanIssue) and factual.dimension is Dimension.FACTUAL
    assert isinstance(logic, PlantedSpanIssue) and logic.dimension is Dimension.LOGIC
    assert isinstance(omission, PlantedOmission)
    assert omission.dimension is Dimension.COMPLETENESS
    assert not case.is_clean


def test_output_is_stripped_and_spans_are_located() -> None:
    case = EvalCase.model_validate(_case())
    assert case.output == OUTPUT
    start, end = case.locate(case.planted[0])  # type: ignore[misc]
    assert case.output[start:end] == "July 20, 1971"
    assert case.locate(case.planted[2]) is not None  # anchor


def test_omission_without_anchor_has_no_location() -> None:
    case = EvalCase.model_validate(
        _case(planted=[{"type": "completeness", "missing": "Duration of the stay."}])
    )
    assert case.locate(case.planted[0]) is None


def test_clean_case() -> None:
    assert EvalCase.model_validate(_case(planted=[])).is_clean


def test_span_not_in_output_is_rejected() -> None:
    bad = [{"type": "factual", "span": "July 21, 1971", "description": "x"}]
    with pytest.raises(ValidationError, match="not found in the output"):
        EvalCase.model_validate(_case(planted=bad))


def test_ambiguous_span_is_rejected() -> None:
    bad = [{"type": "factual", "span": "the", "description": "x"}]
    with pytest.raises(ValidationError, match="appears .* times"):
        EvalCase.model_validate(_case(planted=bad))


def test_missing_anchor_text_is_rejected() -> None:
    bad = [{"type": "completeness", "missing": "x", "anchor": "not in the text"}]
    with pytest.raises(ValidationError, match="anchor"):
        EvalCase.model_validate(_case(planted=bad))


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "completeness", "span": "July 20, 1971", "missing": "x"},  # span on omission
        {"type": "factual", "missing": "x", "description": "x"},  # missing on span issue
        {"type": "style", "span": "July 20, 1971", "description": "x"},  # unknown type
    ],
)
def test_wrong_shape_for_type_is_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        EvalCase.model_validate(_case(planted=[bad]))


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate case ids"):
        EvalSet.model_validate({"cases": [_case(), _case()]})


def test_yaml_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "cases": [_case()]}))
    assert load_testset(path).cases[0].id == "apollo-01"
