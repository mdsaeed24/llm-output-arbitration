"""Disagreement detector: grouping, the three disagreement kinds, unanimous pass."""

from __future__ import annotations

from arbiter.config import Dimension, DisagreementConfig
from arbiter.graph.disagreement import detect_disagreements
from arbiter.schemas.disagreement import DisagreementKind as K
from arbiter.spans import overlap_ratio, spans_overlap
from tests.helpers import critique, failure

CFG = DisagreementConfig()
DATE = "July 20, 1971"
DATE_SENTENCE = "Apollo 11 landed on the Moon on July 20, 1971."
CONCLUSION = "competition is the only driver of progress"
WALKERS = "Neil Armstrong and Buzz Aldrin walked on the surface"


def _kinds(report) -> list[K]:  # type: ignore[no-untyped-def]
    return [d.kind for d in report.disagreements]


def test_overlap_ratio_uses_shorter_span() -> None:
    assert overlap_ratio((0, 10), (2, 5)) == 1.0  # nested
    assert overlap_ratio((0, 10), (8, 20)) == 0.2
    assert overlap_ratio((0, 5), (5, 9)) == 0.0  # touching, not overlapping
    assert not spans_overlap((0, 10), (8, 20), 0.5)


def test_overlapping_issues_from_two_critics_are_an_agreement() -> None:
    report = detect_disagreements(
        [
            critique("factual", issues=[(DATE, "high")]),
            critique("logic", issues=[(DATE_SENTENCE, "high")]),
            critique("completeness"),
        ],
        [],
        CFG,
    )
    [group] = report.groups
    assert group.critics == ["factual", "logic"] and group.corroborated
    assert report.agreements == [group]
    assert report.disagreements == []


def test_single_source_issue_is_uncorroborated_and_names_silent_critics() -> None:
    report = detect_disagreements(
        [critique("factual", issues=[(DATE, "high")]), critique("logic"), critique("completeness")],
        [],
        CFG,
    )
    [d] = report.disagreements
    assert d.kind is K.UNCORROBORATED
    assert d.raised_by == ["factual"] and d.other_critics == ["logic", "completeness"]


def test_severity_conflict_needs_two_levels() -> None:
    def run(sev_a: str, sev_b: str) -> list[K]:
        return _kinds(
            detect_disagreements(
                [
                    critique("factual", issues=[(DATE, sev_a)]),
                    critique("logic", issues=[(DATE_SENTENCE, sev_b)]),
                ],
                [],
                CFG,
            )
        )

    assert run("low", "high") == [K.SEVERITY_CONFLICT]
    assert run("medium", "high") == []


def test_flag_on_a_span_another_critic_verified_is_opposite_judgment() -> None:
    report = detect_disagreements(
        [
            critique("factual", verified=[WALKERS]),
            critique("logic", issues=[(WALKERS, "medium")]),
        ],
        [],
        CFG,
    )
    [d] = report.disagreements  # opposite REPLACES uncorroborated, not in addition
    assert d.kind is K.OPPOSITE_JUDGMENT
    assert d.raised_by == ["logic"] and d.other_critics == ["factual"]


def test_critic_verifying_its_own_flag_is_not_opposite() -> None:
    report = detect_disagreements(
        [critique("factual", issues=[(DATE, "high")], verified=[DATE_SENTENCE])], [], CFG
    )
    assert _kinds(report) == [K.UNCORROBORATED]


def test_completeness_issue_never_groups_by_span() -> None:
    """Completeness quotes are anchors; overlapping a logic span is NOT agreement."""
    report = detect_disagreements(
        [
            critique("logic", issues=[(CONCLUSION, "medium")]),
            critique("completeness", issues=[(CONCLUSION, "high")]),
        ],
        [],
        CFG,
    )
    assert len(report.groups) == 2
    assert not any(g.corroborated for g in report.groups)
    assert _kinds(report) == [K.UNCORROBORATED, K.UNCORROBORATED]


def test_completeness_issue_on_verified_span_is_not_opposite() -> None:
    report = detect_disagreements(
        [
            critique("factual", verified=[WALKERS]),
            critique("completeness", issues=[(WALKERS, "high")]),
        ],
        [],
        CFG,
    )
    assert _kinds(report) == [K.UNCORROBORATED]


def test_invalid_quotes_are_excluded_but_listed() -> None:
    report = detect_disagreements(
        [critique("factual", issues=[("landed on Mars", "high")])], [], CFG
    )
    assert report.groups == [] and report.disagreements == []
    assert report.invalid_issue_ids == ["factual-1"]


def test_unanimous_pass_requires_all_clean_confident_and_no_failures() -> None:
    clean = [critique(d.value) for d in Dimension]
    assert detect_disagreements(clean, [], CFG).unanimous_pass

    unsure = [*clean[:2], critique("completeness", confidence=0.6)]
    assert not detect_disagreements(unsure, [], CFG).unanimous_pass

    assert not detect_disagreements(clean[:2], [failure("completeness")], CFG).unanimous_pass
    assert not detect_disagreements([], [], CFG).unanimous_pass

    flagged = [*clean[:2], critique("completeness", issues=[(WALKERS, "low")])]
    assert not detect_disagreements(flagged, [], CFG).unanimous_pass


def test_hallucinated_only_issues_do_not_block_unanimous_pass() -> None:
    critiques = [
        critique("factual", issues=[("landed on Mars", "high")]),
        critique("logic"),
        critique("completeness"),
    ]
    assert detect_disagreements(critiques, [], CFG).unanimous_pass
