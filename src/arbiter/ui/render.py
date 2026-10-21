"""Pure rendering logic for the Verdict Explorer (no Streamlit imports — unit-testable).

Highlight categories (each mark also carries a text tag, so meaning is never colour-only):
* confirmed      (red)    – the adjudicator (or union rule) accepted the issue with confidence
* low-confidence (amber)  – confirmed, but weakly: low ruling confidence, a low-confidence
                            dimension, or confirmed without review (adjudicator fallback)
* dismissed      (grey)   – flagged by a critic, rejected by the adjudicator
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from typing import Literal

from arbiter.schemas.critique import Critique, Issue
from arbiter.schemas.verdict import ArbitrationResult, ConfirmedIssue, DismissedFlag

Kind = Literal["confirmed", "low", "dismissed"]
PRIORITY: dict[Kind, int] = {"confirmed": 3, "low": 2, "dismissed": 1}
LOW_CONFIDENCE_BELOW = 0.6


@dataclass(frozen=True)
class Mark:
    """One highlightable flag."""

    id: str  # "1", "2" … for confirmed/low, "D1" … for dismissed
    kind: Kind
    start: int | None
    end: int | None
    title: str


@dataclass
class Segment:
    """A run of text with the same set of marks over it."""

    text: str
    mark_ids: list[str] = field(default_factory=list)
    kind: Kind | None = None


@dataclass
class EvidenceStep:
    """One link in the evidence chain."""

    stage: str
    title: str
    body: str
    meta: str = ""


def _is_low(ci: ConfirmedIssue, result: ArbitrationResult) -> bool:
    v = result.verdict
    return (
        ci.confidence < LOW_CONFIDENCE_BELOW
        or any(d in v.low_confidence_dimensions for d in ci.dimensions)
        or v.adjudicator == "union-fallback"
    )


def marks(result: ArbitrationResult) -> list[Mark]:
    """Every confirmed issue and dismissed flag as a mark (dismissed ones may lack a span)."""
    out: list[Mark] = []
    for n, ci in enumerate(result.verdict.confirmed_issues, start=1):
        kind: Kind = "low" if _is_low(ci, result) else "confirmed"
        out.append(
            Mark(
                id=str(n),
                kind=kind,
                start=ci.span_start,
                end=ci.span_end,
                title=f"{ci.severity.value} · {', '.join(d.value for d in ci.dimensions)}",
            )
        )
    for n, df in enumerate(result.verdict.dismissed_flags, start=1):
        out.append(
            Mark(
                id=f"D{n}",
                kind="dismissed",
                start=df.span_start,
                end=df.span_end,
                title=f"dismissed ({df.rule.replace('_', ' ')})",
            )
        )
    return out


def segments(text: str, all_marks: list[Mark]) -> list[Segment]:
    """Split ``text`` at every mark boundary; each segment shows its highest-priority kind."""
    located = [(m.start, m.end, m) for m in all_marks if m.start is not None and m.end is not None]
    cuts = sorted({0, len(text), *[s for s, _, _ in located], *[e for _, e, _ in located]})
    out: list[Segment] = []
    for a, b in zip(cuts, cuts[1:], strict=False):
        if a >= b:
            continue
        covering = [m for s, e, m in located if s <= a and b <= e]
        kind = max((m.kind for m in covering), key=PRIORITY.__getitem__, default=None)
        out.append(Segment(text=text[a:b], mark_ids=[m.id for m in covering], kind=kind))
    return out


def _issue_lookup(result: ArbitrationResult) -> dict[str, tuple[Issue, Critique]]:
    return {i.id: (i, c) for c in result.critiques for i in c.issues}


def evidence_chain(result: ArbitrationResult, mark_id: str) -> list[EvidenceStep]:
    """Critic flag(s) → disagreement(s) → adjudicator ruling → final status."""
    issues = _issue_lookup(result)
    disagreements = {d.id: d for d in result.report.disagreements}
    item: ConfirmedIssue | DismissedFlag
    if mark_id.startswith("D"):
        item = result.verdict.dismissed_flags[int(mark_id[1:]) - 1]
    else:
        item = result.verdict.confirmed_issues[int(mark_id) - 1]

    steps: list[EvidenceStep] = []
    for iid in item.issue_ids:
        if iid not in issues:
            continue
        issue, critique = issues[iid]
        steps.append(
            EvidenceStep(
                stage="critic",
                title=f"{critique.critic} critic flagged it ({issue.severity.value})",
                body=issue.problem,
                meta=f"{critique.provider}/{critique.model_id} · self-confidence "
                f"{critique.confidence:.2f} · quote "
                + ("verified in text" if issue.valid else "NOT found in text"),
            )
        )
    for did in item.disagreement_ids:
        d = disagreements.get(did)
        if d is not None:
            steps.append(
                EvidenceStep(
                    stage="disagreement",
                    title=f"Disagreement: {d.kind.value.replace('_', ' ')}",
                    body=d.detail,
                )
            )
    if isinstance(item, DismissedFlag):
        stage_title = (
            "Dismissed automatically: quote not in the text"
            if item.rule == "invalid_quote"
            else "Adjudicator dismissed it"
        )
        steps.append(EvidenceStep(stage="ruling", title=stage_title, body=item.reason))
        steps.append(EvidenceStep(stage="final", title="Final: not counted", body=""))
    else:
        who = result.verdict.adjudicator or "rule"
        reviewed = who.startswith("llm:")
        steps.append(
            EvidenceStep(
                stage="ruling",
                title=(
                    f"Adjudicator confirmed it ({item.severity.value})"
                    if reviewed
                    else "Confirmed WITHOUT review (adjudicator unavailable — union of critics)"
                ),
                body=item.evidence,
                meta=f"{who} · ruling confidence {item.confidence:.2f}",
            )
        )
        steps.append(
            EvidenceStep(
                stage="final",
                title=f"Final: confirmed {item.severity.value} issue",
                body="Counts towards the verdict"
                + (" (fails it)" if result.verdict.passed is False else ""),
            )
        )
    return steps


# --------------------------------------------------------------------------- HTML

_CSS = """
:root { --ink:#0b0b0b; --muted:#52514e; --surface:#fcfcfb; --panel:#f3f2ef; --line:#dedcd6;
  --red-bg:rgba(208,59,59,.16); --red:#d03b3b; --amber-bg:rgba(250,178,25,.30); --amber:#b07a00;
  --grey-bg:rgba(120,120,116,.16); --grey:#8a8984; }
body.dark { --ink:#f0efec; --muted:#c3c2b7; --surface:#1a1a19; --panel:#242422; --line:#3a3a37;
  --red-bg:rgba(230,103,103,.30); --red:#e66767; --amber-bg:rgba(250,178,25,.28); --amber:#fab219;
  --grey-bg:rgba(170,170,165,.22); --grey:#a8a7a0; }
body { margin:0; font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  color:var(--ink); background:var(--surface); }
.wrap { display:flex; gap:16px; flex-wrap:wrap; }
.text { flex:3 1 380px; white-space:pre-wrap; padding:4px 2px; }
.panel { flex:2 1 260px; background:var(--panel); border:1px solid var(--line); border-radius:8px;
  padding:12px 14px; font-size:13.5px; align-self:flex-start; }
mark { color:inherit; border-radius:3px; padding:1px 0; cursor:pointer; }
mark.confirmed { background:var(--red-bg); border-bottom:2px solid var(--red); }
mark.low { background:var(--amber-bg); border-bottom:2px solid var(--amber); }
mark.dismissed { background:var(--grey-bg); border-bottom:2px dashed var(--grey); }
mark.active { outline:2px solid var(--ink); outline-offset:1px; }
sup.tag { font-size:10px; font-weight:700; margin-left:1px; color:var(--muted); }
.legend { display:flex; gap:14px; flex-wrap:wrap; font-size:12.5px; color:var(--muted);
  margin-bottom:10px; }
.legend span mark { cursor:default; padding:0 6px; }
.step { border-left:3px solid var(--line); padding:2px 0 8px 10px; margin-left:4px; }
.step.critic { border-color:#2a78d6; } .step.disagreement { border-color:#b07a00; }
.step.ruling { border-color:var(--ink); } .step.final { border-color:var(--muted); }
.step b { display:block; } .meta { color:var(--muted); font-size:12px; }
.hint { color:var(--muted); }
"""

_JS = """
function show(id) {
  document.querySelectorAll('mark.active').forEach(m => m.classList.remove('active'));
  document.querySelectorAll('mark[data-ids~="' + id + '"]').forEach(m => m.classList.add('active'));
  document.getElementById('panel').innerHTML = CHAINS[id] || '';
}
"""


def _chain_html(steps: list[EvidenceStep], mark: Mark) -> str:
    parts = [
        f"<div><b>Flag {html.escape(mark.id)}</b> "
        f"<span class='meta'>{html.escape(mark.title)}</span></div>"
    ]
    for s in steps:
        parts.append(
            f"<div class='step {s.stage}'><b>{html.escape(s.title)}</b>"
            + (f"<div>{html.escape(s.body)}</div>" if s.body else "")
            + (f"<div class='meta'>{html.escape(s.meta)}</div>" if s.meta else "")
            + "</div>"
        )
    return "".join(parts)


def _script_safe_json(data: object) -> str:
    """JSON safe to inline in <script>: no "</script>", "<!--" or HTML-significant characters.

    Defence in depth — every user/LLM-derived string is already html-escaped, but the evaluated
    output and the critics' explanations are untrusted LLM text, so the script block must not be
    breakable even if an escaping step is ever missed.
    """
    return json.dumps(data).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def highlight_html(result: ArbitrationResult, *, dark: bool = False) -> str:
    """Self-contained HTML: highlighted text + an evidence panel updated on click.

    All text from the evaluated output, critics and adjudicator is UNTRUSTED (LLM-generated) and
    is html-escaped; the inline JSON is additionally script-safe (see ``_script_safe_json``).
    """
    all_marks = marks(result)
    chains = {m.id: _chain_html(evidence_chain(result, m.id), m) for m in all_marks}
    first_tag: set[str] = set()
    body: list[str] = []
    for seg in segments(result.output, all_marks):
        text = html.escape(seg.text)
        if seg.kind is None:
            body.append(text)
            continue
        new = [i for i in seg.mark_ids if i not in first_tag]
        first_tag.update(new)
        tags = "".join(f"<sup class='tag'>{html.escape(i)}</sup>" for i in new)
        target = sorted(seg.mark_ids, key=lambda i: (i.startswith("D"), i))[0]
        body.append(
            f"<mark class='{seg.kind}' data-ids='{' '.join(seg.mark_ids)}' "
            f"onclick=\"show('{target}')\">{text}</mark>{tags}"
        )
    legend = (
        "<div class='legend'>"
        "<span><mark class='confirmed'>red</mark> confirmed issue</span>"
        "<span><mark class='low'>amber</mark> low-confidence flag</span>"
        "<span><mark class='dismissed'>grey</mark> dismissed by adjudicator</span>"
        "</div>"
    )
    hint = (
        "<span class='hint'>Click a highlight to see its evidence chain.</span>"
        if all_marks
        else "<span class='hint'>No issues were flagged.</span>"
    )
    return (
        f"<html><head><style>{_CSS}</style></head><body class='{'dark' if dark else ''}'>"
        f"{legend}<div class='wrap'><div class='text'>{''.join(body)}</div>"
        f"<div class='panel' id='panel'>{hint}</div></div>"
        f"<script>const CHAINS = {_script_safe_json(chains)};{_JS}</script></body></html>"
    )


# --------------------------------------------------------------------------- comparison


@dataclass
class CriticColumn:
    """Data for one critic's column in the comparison panel."""

    critic: str
    model: str
    status: str
    score: int | None
    confidence: float | None
    analysis: str
    valid_issues: list[Issue]
    invalid_issues: list[Issue]
    verified: list[str]


def critic_columns(result: ArbitrationResult) -> list[CriticColumn]:
    """One column per critic, including failed ones."""
    cols = [
        CriticColumn(
            critic=c.critic,
            model=f"{c.provider}/{c.model_id}",
            status="ok",
            score=c.score,
            confidence=c.confidence,
            analysis=c.analysis,
            valid_issues=c.valid_issues,
            invalid_issues=c.invalid_issues,
            verified=[v.quote for v in c.verified_claims if v.valid],
        )
        for c in result.critiques
    ]
    cols += [
        CriticColumn(
            critic=f.critic,
            model=f.model_key,
            status=f"failed: {f.error_type}",
            score=None,
            confidence=None,
            analysis=f.message,
            valid_issues=[],
            invalid_issues=[],
            verified=[],
        )
        for f in result.failures
    ]
    order = {"factual": 0, "logic": 1, "completeness": 2}
    return sorted(cols, key=lambda c: (order.get(c.critic, 9), c.critic))


def agreement_rows(result: ArbitrationResult) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """(agreements, disagreements) as table rows."""
    text = result.output
    agreements = [
        {
            "group": g.id,
            "raised by": " + ".join(g.critics),
            "severity": g.max_severity.value
            + ("" if g.max_severity == g.min_severity else f" (min {g.min_severity.value})"),
            "span": text[g.span_start : g.span_end],
        }
        for g in result.report.agreements
    ]
    disagreements = [
        {
            "id": d.id,
            "kind": d.kind.value.replace("_", " "),
            "group": d.group_id,
            "detail": d.detail,
        }
        for d in result.report.disagreements
    ]
    return agreements, disagreements
