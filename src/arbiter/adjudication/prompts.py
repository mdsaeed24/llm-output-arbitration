"""Adjudicator prompt.

The adjudicator rules on every issue group (confirm or dismiss) and never adds new issues:
it uses the same model as the single-reviewer baseline, so letting it add findings would turn
the "multi-critic system" into "the baseline plus extras" and make the comparison meaningless.
Critic identities are shown by dimension only (no model names) to avoid "trust the bigger
model" bias.
"""

from __future__ import annotations

from arbiter.adjudication.base import AdjudicationInput
from arbiter.config import Dimension
from arbiter.critics.prompts import DATA_RULE, SEVERITY_DEFINITIONS, dimension_name, question_block

OVERALL_SCORE_RUBRIC = """\
"overall_score" (1-10) for the output as a whole: 9-10 = no real problems (at most trivial);
7-8 = only low-severity problems, or one medium; 4-6 = at least one high-severity problem;
1-3 = several high-severity problems or fundamentally unreliable."""

SYSTEM_PROMPT = f"""\
You are the ADJUDICATOR in a multi-reviewer evaluation of LLM-generated text. Independent
critics, each responsible for one dimension, have flagged possible problems. Flags that touch
the same text have been grouped, and an automatic check has listed where the critics disagree.
For EVERY group, decide whether the flagged problem is real ("confirm") or not ("dismiss"),
then give an overall verdict.

You do NOT report new problems. Rule only on the groups provided, even if you notice something
else.

How to rule — reason explicitly in "reasoning" BEFORE choosing "decision":
- Factual groups: check the claim against the evaluated text itself (is it consistent with the
  rest of the text?) and against your own knowledge. Confirm only if you are confident the claim
  is wrong or inconsistent. You have no web access; if you cannot tell, say so and lower
  "confidence".
- Logic groups: trace the reasoning step by step — the premises, then the inference — and name
  exactly which step fails, or explain why it actually holds.
- Completeness groups: re-read the original question and list what it actually asks for, then
  check whether the output provides the flagged item ANYWHERE (the quoted sentence is only an
  anchor near the gap). If no question was provided, judge against the task the output
  evidently addresses.

Disagreement types you will see:
- uncorroborated: only one critic raised it. This is normal — critics cover different
  dimensions — and is neither a reason to confirm nor to dismiss. Verify it yourself.
- severity_conflict: critics agree there is a problem but rate it 2+ levels apart. Decide the
  right severity and say why.
- opposite_judgment: one critic flagged the span, another explicitly verified it as correct.
  Decide who is right and say why.
Groups with no listed disagreement were raised by several critics who agree; still check them,
briefly.

"severity" for a confirmed group: {SEVERITY_DEFINITIONS} You may change the critics' severity
when justified. For a dismissed group set "severity" to null and explain in "reasoning" why the
flag is wrong.
"confidence" in each ruling (0.0-1.0): how sure you are of that ruling.

{OVERALL_SCORE_RUBRIC}
"confidence" (0.0-1.0): how sure you are of the overall verdict.
"summary": one paragraph (3-5 sentences) for a reader who has not seen the critics: what is
wrong with the output (or that it is sound), and what was dismissed and why.

{DATA_RULE}

Write "rulings" first (one per group, in group order), then "overall_score", "confidence" and
"summary"."""


def user_prompt(inp: AdjudicationInput, *, missing: list[Dimension] | None = None) -> str:
    """Question, output, critic reports, issue groups and disagreements for the adjudicator."""
    issues = {i.id: i for c in inp.critiques for i in c.issues}
    disagreements_by_group: dict[str, list[str]] = {}
    for d in inp.report.disagreements:
        disagreements_by_group.setdefault(d.group_id, []).append(
            f"{d.id} {d.kind.value}: {d.detail}"
        )

    reports = []
    for c in inp.critiques:
        claims = [
            f'  - verified as correct: "{v.quote}" — {v.note}' for v in c.verified_claims if v.valid
        ]
        reports.append(
            "\n".join(
                [
                    f"## {c.critic} critic ({dimension_name(c.dimension)}) — dimension score "
                    f"{c.score}/5, self-confidence {c.confidence:.2f}",
                    f"analysis: {c.analysis}",
                    *claims,
                ]
            )
        )
    if missing:
        reports.append(
            "NOT REVIEWED (critic failed): " + ", ".join(dimension_name(d) for d in missing)
        )

    groups = []
    for g in inp.report.groups:
        omission = g.dimensions == [Dimension.COMPLETENESS]
        lines = [
            f"### {g.id} — {', '.join(dimension_name(d) for d in g.dimensions)} — raised by: "
            f"{', '.join(g.critics)}",
            f"span{' (anchor only; the problem is something MISSING)' if omission else ''}: "
            f'"{inp.output[g.span_start : g.span_end]}"',
        ]
        for issue_id in g.issue_ids:
            i = issues[issue_id]
            lines.append(f"- {i.id} ({i.critic}, {i.severity.value}): {i.problem}")
        for note in disagreements_by_group.get(g.id, []):
            lines.append(f"DISAGREEMENT {note}")
        groups.append("\n".join(lines))

    ids = ", ".join(g.id for g in inp.report.groups)
    return "\n\n".join(
        [
            question_block(inp.question),
            f"<evaluated_output>\n{inp.output}\n</evaluated_output>",
            "<critic_reports>\n" + "\n\n".join(reports) + "\n</critic_reports>",
            "<issue_groups>\n"
            + ("\n\n".join(groups) if groups else "(no issues were raised)")
            + "\n</issue_groups>",
            (
                f"Rule on EVERY group: {ids}. Reply with the JSON object."
                if groups
                else "No issues were raised, but at least one critic was not confident. Give "
                '"rulings": [] and an overall verdict based on the critic reports.'
            ),
        ]
    )
