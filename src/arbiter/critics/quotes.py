"""Quote validation: does an issue's quote actually appear in the evaluated text?

This is the main defence against critics *hallucinating problems*. Matching is deliberately
strict — exact, then a normalised match that forgives only cosmetic differences LLMs commonly
introduce (curly vs straight quotes, dashes, whitespace, case, trailing punctuation), then
ellipsis fragments in order. There is no fuzzy matching: a paraphrased "quote" is exactly
the kind of invented evidence we want to reject.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from arbiter.config import Dimension
from arbiter.schemas.critique import (
    Issue,
    IssueDraft,
    MatchMethod,
    VerifiedClaim,
    VerifiedClaimDraft,
)

MIN_QUOTE_CHARS = 3
"""Quotes with fewer alphanumeric chars would match almost anything ("the", "a")."""

_CHAR_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "′": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "″": '"',
        "–": "-",
        "—": "-",
        "−": "-",
        "‐": "-",
        "‑": "-",
        " ": " ",
        " ": " ",
        " ": " ",
        "…": "...",
    }
)
_WRAPPING_QUOTES = "\"'`“”‘’"
_EDGE_PUNCT = " .,;:!?"
_ELLIPSIS_RE = re.compile(r"\s*(?:\.\.\.|\[\.\.\.\])\s*")


@dataclass(frozen=True)
class QuoteMatch:
    """Where (and how) a quote was found."""

    valid: bool
    start: int | None = None
    end: int | None = None
    method: MatchMethod = "none"
    reason: str | None = None


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalise ``text`` and return, for every normalised char, its index in the original."""
    out: list[str] = []
    index: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch).translate(_CHAR_MAP).lower():
            if c.isspace():
                if prev_space:
                    continue
                c, prev_space = " ", True
            else:
                prev_space = False
            out.append(c)
            index.append(i)
    return "".join(out), index


def _normalize(text: str) -> str:
    return _normalize_with_map(text)[0].strip()


def _alnum_len(text: str) -> int:
    return sum(ch.isalnum() for ch in text)


def _span(index: list[int], start: int, length: int) -> tuple[int, int]:
    return index[start], index[start + length - 1] + 1


def locate_quote(quote: str, text: str) -> QuoteMatch:
    """Find ``quote`` in ``text`` and return its character span in the ORIGINAL text."""
    raw = quote.strip()
    if _alnum_len(raw) < MIN_QUOTE_CHARS:
        return QuoteMatch(valid=False, reason="quote too short to locate reliably")

    if (pos := text.find(raw)) >= 0:
        return QuoteMatch(valid=True, start=pos, end=pos + len(raw), method="exact")

    norm_text, index = _normalize_with_map(text)
    norm_quote = _normalize(raw.strip(_WRAPPING_QUOTES)).strip(_EDGE_PUNCT + _WRAPPING_QUOTES)

    if norm_quote and (pos := norm_text.find(norm_quote)) >= 0:
        start, end = _span(index, pos, len(norm_quote))
        return QuoteMatch(valid=True, start=start, end=end, method="normalized")

    fragments = [f.strip(_EDGE_PUNCT) for f in _ELLIPSIS_RE.split(norm_quote)]
    fragments = [f for f in fragments if f]
    if len(fragments) > 1:
        if any(_alnum_len(f) < MIN_QUOTE_CHARS for f in fragments):
            return QuoteMatch(valid=False, reason="ellipsis fragment too short to locate")
        cursor, first_start, last_end = 0, -1, -1
        for frag in fragments:
            pos = norm_text.find(frag, cursor)
            if pos < 0:
                return QuoteMatch(valid=False, reason=f"fragment not found in order: {frag!r}")
            start, end = _span(index, pos, len(frag))
            first_start = start if first_start < 0 else first_start
            last_end = end
            cursor = pos + len(frag)
        return QuoteMatch(valid=True, start=first_start, end=last_end, method="fragments")

    return QuoteMatch(valid=False, reason="quote not found in evaluated text")


def validate_issues(
    drafts: Sequence[IssueDraft], text: str, *, critic: str, dimension: Dimension
) -> list[Issue]:
    """Turn raw critic issues into validated :class:`Issue` objects (invalid ones are kept,
    flagged, so hallucination rates can be measured)."""
    issues: list[Issue] = []
    for n, draft in enumerate(drafts, start=1):
        match = locate_quote(draft.quote, text)
        issues.append(
            Issue(
                **draft.model_dump(),
                id=f"{critic}-{n}",
                critic=critic,
                dimension=dimension,
                valid=match.valid,
                span_start=match.start,
                span_end=match.end,
                match_method=match.method,
                invalid_reason=match.reason,
            )
        )
    return issues


def validate_claims(
    drafts: Sequence[VerifiedClaimDraft], text: str, *, critic: str, dimension: Dimension
) -> list[VerifiedClaim]:
    """Locate verified-claim quotes the same way as issue quotes."""
    claims: list[VerifiedClaim] = []
    for draft in drafts:
        match = locate_quote(draft.quote, text)
        claims.append(
            VerifiedClaim(
                **draft.model_dump(),
                critic=critic,
                dimension=dimension,
                valid=match.valid,
                span_start=match.start,
                span_end=match.end,
            )
        )
    return claims
