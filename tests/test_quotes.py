"""Quote validation: real quotes are located precisely; invented ones are rejected."""

from __future__ import annotations

import pytest

from arbiter.config import Dimension
from arbiter.critics.quotes import locate_quote, validate_issues
from arbiter.schemas.critique import IssueDraft, Severity

TEXT = (
    "Apollo 11 landed on the Moon on July 20, 1969. Neil Armstrong said “That’s one "
    "small step for man.”  The mission—launched from Florida—ended the space race."
)


def _span(text: str, quote: str) -> str:
    m = locate_quote(quote, text)
    assert m.valid and m.start is not None and m.end is not None, m
    return text[m.start : m.end]


def test_exact_match_returns_exact_span() -> None:
    m = locate_quote("landed on the Moon on July 20, 1969", TEXT)
    assert m.valid and m.method == "exact"
    assert TEXT[m.start : m.end] == "landed on the Moon on July 20, 1969"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        # straight quotes/apostrophes where the text uses curly ones
        # (edge punctuation is trimmed before matching, so the span ends before the ".")
        ('"That\'s one small step for man."', "That’s one small step for man"),
        # hyphens instead of em dashes, collapsed whitespace, different case
        ("the mission-launched from florida-ended", "The mission—launched from Florida—ended"),
        # trailing punctuation added by the critic
        ("ended the space race!", "ended the space race"),
    ],
)
def test_normalized_match_maps_back_to_original_text(quote: str, expected: str) -> None:
    assert locate_quote(quote, TEXT).method == "normalized"
    assert _span(TEXT, quote) == expected


def test_whitespace_runs_in_text_are_collapsed() -> None:
    assert _span(TEXT, "for man.” The mission") == "for man.”  The mission"


def test_ellipsis_fragments_in_order_are_valid() -> None:
    m = locate_quote("Apollo 11 landed ... ended the space race", TEXT)
    assert m.valid and m.method == "fragments"
    assert TEXT[m.start : m.end].startswith("Apollo 11 landed")  # type: ignore[misc]
    assert TEXT[m.start : m.end].endswith("ended the space race")  # type: ignore[misc]


def test_ellipsis_fragments_out_of_order_are_invalid() -> None:
    m = locate_quote("ended the space race ... Apollo 11 landed", TEXT)
    assert not m.valid and "not found in order" in (m.reason or "")


def test_paraphrased_quote_is_invalid() -> None:
    m = locate_quote("Apollo 11 touched down on the lunar surface in 1969", TEXT)
    assert not m.valid and m.method == "none"


def test_hallucinated_number_is_invalid() -> None:
    """A critic 'quoting' a wrong date that the text never contains must not validate."""
    assert not locate_quote("July 20, 1971", TEXT).valid


@pytest.mark.parametrize("quote", ["", "  ", "a", "...", "11"])
def test_too_short_quotes_are_invalid(quote: str) -> None:
    assert not locate_quote(quote, TEXT).valid


def test_validate_issues_keeps_invalid_issues_flagged() -> None:
    drafts = [
        IssueDraft(quote="July 20, 1969", problem="fine", severity=Severity.LOW),
        IssueDraft(quote="landed on Mars", problem="invented", severity=Severity.HIGH),
    ]
    issues = validate_issues(drafts, TEXT, critic="factual", dimension=Dimension.FACTUAL)
    assert [i.valid for i in issues] == [True, False]
    assert [i.id for i in issues] == ["factual-1", "factual-2"]
    assert issues[1].invalid_reason == "quote not found in evaluated text"
    assert issues[0].dimension is Dimension.FACTUAL


def test_severity_rank_orders_levels() -> None:
    assert Severity.HIGH.rank - Severity.LOW.rank == 2
