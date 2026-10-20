"""Character-span arithmetic shared by disagreement detection and evaluation matching."""

from __future__ import annotations

Span = tuple[int, int]


def overlap_ratio(a: Span, b: Span) -> float:
    """Overlap length divided by the length of the SHORTER span (0.0 – 1.0).

    Using the shorter span means a tight quote ("July 20, 1971") fully inside a longer quote
    of the same sentence scores 1.0, while two long quotes that merely touch score low.
    """
    shorter = min(a[1] - a[0], b[1] - b[0])
    if shorter <= 0:
        return 0.0
    inter = min(a[1], b[1]) - max(a[0], b[0])
    return max(0, inter) / shorter


def spans_overlap(a: Span, b: Span, min_ratio: float) -> bool:
    """True when ``a`` and ``b`` refer to the same text under ``min_ratio``."""
    return overlap_ratio(a, b) >= min_ratio


def union(spans: list[Span]) -> Span:
    """Smallest span covering all ``spans``."""
    return min(s[0] for s in spans), max(s[1] for s in spans)
