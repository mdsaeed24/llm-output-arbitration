"""Builders for hand-made critiques (no LLM involved)."""

from __future__ import annotations

from arbiter.config import Dimension
from arbiter.critics.quotes import validate_claims, validate_issues
from arbiter.schemas.critique import (
    CriticFailure,
    Critique,
    IssueDraft,
    Severity,
    VerifiedClaimDraft,
)

TEXT = (
    "Apollo 11 landed on the Moon on July 20, 1971. Neil Armstrong and Buzz Aldrin walked "
    "on the surface. Because it happened in the Cold War, competition is the only driver of "
    "progress."
)


def critique(
    critic: str,
    dimension: Dimension | None = None,
    *,
    issues: list[tuple[str, str]] = (),  # type: ignore[assignment]
    verified: list[str] = (),  # type: ignore[assignment]
    confidence: float = 0.9,
    text: str = TEXT,
) -> Critique:
    """``issues`` = [(quote, severity), ...]; ``verified`` = [quote, ...]."""
    dim = dimension or Dimension(critic)
    drafts = [
        IssueDraft(quote=q, problem=f"problem with {q!r}", severity=Severity(s)) for q, s in issues
    ]
    claims = [VerifiedClaimDraft(quote=q, note="checked") for q in verified]
    return Critique(
        critic=critic,
        dimension=dim,
        model_key="m",
        model_id="m",
        provider="p",
        analysis="a",
        score=5 if not issues else 3,
        confidence=confidence,
        issues=validate_issues(drafts, text, critic=critic, dimension=dim),
        verified_claims=validate_claims(claims, text, critic=critic, dimension=dim),
    )


def failure(critic: str) -> CriticFailure:
    return CriticFailure(
        critic=critic,
        dimension=Dimension(critic),
        model_key="m",
        error_type="FatalAPIError",
        message="boom",
    )
