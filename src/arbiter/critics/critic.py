"""Run one critic: prompt → structured LLM call → quote validation → :class:`Critique`."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from arbiter.config import AppConfig, Dimension
from arbiter.critics.prompts import system_prompt, user_prompt
from arbiter.critics.quotes import validate_claims, validate_issues
from arbiter.llm.client import LLMClient
from arbiter.schemas.critique import Critique, CritiqueDraft


class CriticSpec(BaseModel):
    """Which critic runs which dimension on which model."""

    model_config = ConfigDict(frozen=True)

    name: str
    dimension: Dimension
    model_key: str


def critic_specs(config: AppConfig, *, include_extra: bool = True) -> list[CriticSpec]:
    """The three core critics (named after their dimension) plus any configured extras."""
    specs = [
        CriticSpec(name=dim.value, dimension=dim, model_key=model_key)
        for dim, model_key in config.roles.critics.items()
    ]
    if include_extra:
        specs += [
            CriticSpec(name=name, dimension=extra.dimension, model_key=extra.model)
            for name, extra in config.roles.extra_critics.items()
        ]
    return specs


async def run_critic(
    llm: LLMClient, spec: CriticSpec, *, output: str, question: str | None = None
) -> Critique:
    """Evaluate ``output`` on ``spec.dimension`` and validate every quoted span.

    Raises the ``LLMError`` subclasses from :mod:`arbiter.llm.client` on failure; the graph
    (Phase 2) catches these and records a ``CriticFailure`` instead of aborting.
    """
    result = await llm.structured(
        role=f"critic:{spec.name}",
        model_key=spec.model_key,
        system=system_prompt(spec.dimension),
        user=user_prompt(spec.dimension, output=output, question=question),
        schema=CritiqueDraft,
    )
    draft = result.value
    model = llm.config.model(spec.model_key)
    return Critique(
        critic=spec.name,
        dimension=spec.dimension,
        model_key=spec.model_key,
        model_id=model.model_id,
        provider=model.provider,
        analysis=draft.analysis,
        score=draft.score,
        confidence=draft.confidence,
        issues=validate_issues(draft.issues, output, critic=spec.name, dimension=spec.dimension),
        verified_claims=validate_claims(
            draft.verified_claims, output, critic=spec.name, dimension=spec.dimension
        ),
        calls=result.calls,
    )
