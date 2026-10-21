"""Render ``reports/eval_report.md`` from :class:`EvalResults`.

The headline is generated from the numbers — including when the full system does NOT beat
the baseline — so the report can't quietly spin a result.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime

from arbiter.config import Dimension
from arbiter.evaluation.findings import BASELINE, SYSTEM, UNION, critic_variant
from arbiter.evaluation.metrics import PLANTED_TYPES, PairedComparison, ScoredCase, wilson
from arbiter.evaluation.runner import EvalResults
from arbiter.schemas.testcase import PlantedOmission, PlantedSpanIssue

ALPHA = 0.05


def _pct(k: int, n: int) -> str:
    return f"{k / n:.0%}" if n else "–"


def _ci(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{_pct(k, n)} ({k}/{n}) [{lo:.0%}–{hi:.0%}]" if n else "–"


def _cell(text: str, limit: int = 140) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _labels(results: EvalResults) -> dict[str, str]:
    cfg = results.config
    model = {k: m.model_id for k, m in cfg.models.items()}
    labels = {
        SYSTEM: "**Full system** (3 critics + adjudicator)",
        BASELINE: f"**Single reviewer** ({model[cfg.roles.baseline]})",
        UNION: "Ablation: union of critics, no adjudicator",
    }
    for name in results.critics:
        core = {d.value: m for d, m in cfg.roles.critics.items()}
        mk = core.get(name) or cfg.roles.extra_critics[name].model
        labels[critic_variant(name)] = f"Ablation: {name} critic alone ({model[mk]})"
    return labels


def _headline(results: EvalResults) -> list[str]:
    s, b = results.scores[SYSTEM], results.scores[BASELINE]
    cmp_ = results.comparisons[0]
    if s.recall > b.recall:
        recall_word = "beat"
    elif s.recall < b.recall:
        recall_word = "did **not** beat"
    else:
        recall_word = "tied"
    significant = cmp_.recall_p < ALPHA
    ff_word = (
        "fewer"
        if s.false_flags < b.false_flags
        else "more"
        if s.false_flags > b.false_flags
        else "the same number of"
    )
    lines = [
        f"**The multi-critic system caught {_pct(s.caught, s.planted_total)} "
        f"({s.caught}/{s.planted_total}) of planted errors vs "
        f"{_pct(b.caught, b.planted_total)} ({b.caught}/{b.planted_total}) for a single "
        f"reviewer, with {s.false_flags} false flags vs {b.false_flags} "
        f"({s.false_flags / max(s.n_cases, 1):.2f} vs {b.false_flags / max(b.n_cases, 1):.2f} "
        f"per case).**",
        "",
        f"- **Recall:** the full system {recall_word} the single reviewer "
        f"(paired exact McNemar p = {_p(cmp_.recall_p)}; system-only catches "
        f"{cmp_.recall_a_only}, baseline-only catches {cmp_.recall_b_only}). "
        + (
            "The difference is statistically significant at α = 0.05."
            if significant
            else f"With n = {s.planted_total} planted issues this difference is **not** "
            "statistically significant at α = 0.05."
        ),
        f"- **False flags:** the system raised {ff_word} false flags"
        f" ({s.false_flags} vs {b.false_flags}; ≥ medium severity: "
        f"{s.false_flags_ge_medium} vs {b.false_flags_ge_medium}). Precision "
        f"{s.precision:.0%} vs {b.precision:.0%}.",
        f"- **Sensitivity:** {s.restatements} of the system's false flags are omission-style "
        "complaints anchored on a planted error's span (mostly the completeness critic "
        "restating a planted error). Counting those as duplicates instead: "
        f"{s.false_flags_excl_restatements} vs {b.false_flags_excl_restatements} false flags, "
        f"precision {s.precision_excl_restatements:.0%} vs {b.precision_excl_restatements:.0%} — "
        + (
            "the false-flag comparison does not change direction."
            if (s.false_flags_excl_restatements > b.false_flags_excl_restatements)
            == (s.false_flags > b.false_flags)
            else "which REVERSES the false-flag comparison."
        ),
        f"- **Verdicts:** pass/fail accuracy {_pct(s.verdict_correct, s.n_cases)} vs "
        f"{_pct(b.verdict_correct, b.n_cases)} (paired p = {_p(cmp_.verdict_p)}). "
        f"Clean outputs passed: {s.clean_passed}/{s.clean_cases} vs {b.clean_passed}/"
        f"{b.clean_cases}; flawed outputs failed: {s.flawed_failed}/{s.flawed_cases} vs "
        f"{b.flawed_failed}/{b.flawed_cases}.",
        f"- **Cost & latency:** ${s.mean_cost:.4f} vs ${b.mean_cost:.4f} per case "
        f"({s.mean_cost / b.mean_cost:.1f}× the cost) and {s.mean_latency:.1f}s vs "
        f"{b.mean_latency:.1f}s critical-path LLM time per case."
        if b.mean_cost
        else "",
    ]
    return lines


def _results_table(results: EvalResults) -> list[str]:
    labels = _labels(results)
    head = (
        "| Variant | Recall [95% CI] | Factual | Logic | Completeness | Precision | "
        "False flags (≥ med) | Clean cases flagged | Verdict accuracy | $ / case | s / case |"
    )
    rows = [head, "|" + "---|" * 11]
    for v in results.variants:
        s = results.scores[v]
        by = [
            _pct(*s.by_type[t]) + f" ({s.by_type[t][0]}/{s.by_type[t][1]})" for t in PLANTED_TYPES
        ]
        rows.append(
            f"| {labels[v]} | {_ci(s.caught, s.planted_total)} | {' | '.join(by)} | "
            f"{_pct(s.findings_matched, s.findings_total)} ({s.findings_matched}/"
            f"{s.findings_total}) | {s.false_flags} ({s.false_flags_ge_medium}) | "
            f"{s.clean_flagged}/{s.clean_cases} | {_ci(s.verdict_correct, s.n_cases)} | "
            f"{s.mean_cost:.4f} | {s.mean_latency:.1f} |"
        )
    return rows


def _sensitivity(results: EvalResults) -> list[str]:
    rows = [
        "The strict rule (agreed before the run) never lets a completeness finding match a "
        "factual/logic planted issue. A completeness critic that restates a planted error as an "
        'omission ("the summary reverses the core distinction…") is therefore a false flag '
        "under the strict rule, although it is really a duplicate. This table shows the same "
        "results with those restatements — omission findings whose quote overlaps a planted "
        "factual/logic span — set aside. It is a mechanical proxy (an anchored complaint could "
        "occasionally be about something else — check the false-flag lists) and is NOT the "
        "headline.",
        "",
        "| Variant | False flags (strict) | Restatements of planted errors | False flags excl. "
        "restatements | Precision (strict) | Precision excl. restatements |",
        "|---|---|---|---|---|---|",
    ]
    for v in results.variants:
        s = results.scores[v]
        rows.append(
            f"| {v} | {s.false_flags} | {s.restatements} | {s.false_flags_excl_restatements} | "
            f"{s.precision:.0%} | {s.precision_excl_restatements:.0%} |"
        )
    return rows


def _comparison(c: PairedComparison, labels: dict[str, str]) -> list[str]:
    return [
        f"**{labels[c.a].replace('**', '')} vs {labels[c.b].replace('**', '')}**",
        "",
        "| | count |",
        "|---|---|",
        f"| planted issues caught by both | {c.recall_both} |",
        f"| caught only by {c.a} | {c.recall_a_only} |",
        f"| caught only by {c.b} | {c.recall_b_only} |",
        f"| caught by neither | {c.recall_neither} |",
        f"| exact McNemar p (recall) | {_p(c.recall_p)} |",
        f"| cases where only {c.a} had the right verdict | {c.verdict_a_only} |",
        f"| cases where only {c.b} had the right verdict | {c.verdict_b_only} |",
        f"| exact McNemar p (verdicts) | {_p(c.verdict_p)} |",
        "",
    ]


def _planted_text(results: EvalResults, case_id: str, idx: int) -> str:
    case = next(c for c in results.cases if c.id == case_id)
    p = case.planted[idx]
    if isinstance(p, PlantedSpanIssue):
        return f'{p.type}: "{_cell(p.span, 70)}" — {_cell(p.description, 90)}'
    assert isinstance(p, PlantedOmission)
    return f"completeness: missing {_cell(p.missing, 120)}"


def _misses(results: EvalResults) -> list[str]:
    sys_s, base_s = results.scored[SYSTEM], results.scored[BASELINE]
    both, sys_only, base_only = [], [], []
    for cid in sorted(set(sys_s) & set(base_s)):
        for idx, (cs, cb) in enumerate(zip(sys_s[cid].caught, base_s[cid].caught, strict=True)):
            line = f"- `{cid}` — {_planted_text(results, cid, idx)}"
            if not cs and not cb:
                both.append(line)
            elif not cs:
                sys_only.append(line)
            elif not cb:
                base_only.append(line)
    out = ["### Missed by both (hardest cases)", "", *(both or ["- none"]), ""]
    out += [
        "### Missed by the full system but caught by the baseline",
        "",
        *(sys_only or ["- none"]),
        "",
    ]
    out += [
        "### Missed by the baseline but caught by the full system",
        "",
        *(base_only or ["- none"]),
        "",
    ]
    return out


def _false_flags_on_clean(results: EvalResults) -> list[str]:
    rows = ["| Case | Variant | Severity | Quote | Complaint |", "|---|---|---|---|---|"]
    for v in (SYSTEM, BASELINE):
        for cid, sc in sorted(results.scored[v].items()):
            if not sc.is_clean:
                continue
            case = next(c for c in results.cases if c.id == cid)
            for f in sc.findings:
                quote = case.output[f.span_start : f.span_end] if f.span else ""
                rows.append(
                    f"| `{cid}` | {v} | {f.severity.value} | {_cell(quote, 60)} | "
                    f"{_cell(f.description, 120)} |"
                )
    return rows if len(rows) > 2 else ["None — neither raised any issue on a clean case."]


def _critics(results: EvalResults) -> list[str]:
    rows = [
        "| Critic | Model | Valid issues | Hallucinated quotes | True positives (precision) "
        "| Planted caught alone | Unique catches | Overruled by adjudicator "
        "| …of which really false |",
        "|" + "---|" * 9,
    ]
    for s in results.critic_stats:
        rows.append(
            f"| {s.critic} | {s.model_id} | {s.valid_issues} | {s.invalid_quotes} "
            f"({s.hallucination_rate:.0%}) | {s.true_positives} ({s.precision:.0%}) | "
            f"{s.planted_caught} | {s.unique_catches} | {s.overruled}/{s.adjudicated} "
            f"({s.overrule_rate:.0%}) | {s.overruled_false_flags}/{s.overruled} |"
        )
    stats = [s for s in results.critic_stats if s.valid_issues]
    if stats:
        most_tp = max(stats, key=lambda s: s.true_positives)
        most_over = max(stats, key=lambda s: s.overrule_rate)
        rows += [
            "",
            f"- **Most valid (true-positive) issues:** {most_tp.critic} "
            f"({most_tp.true_positives}).",
            f"- **Overruled most often:** {most_over.critic} ({most_over.overruled}/"
            f"{most_over.adjudicated} = {most_over.overrule_rate:.0%} of its adjudicated issues).",
        ]
    ag = results.agreement
    rows += [
        "",
        "**Agreement between critics** (pass/fail verdict each critic would give alone):",
        "",
    ]
    rows += ["| Pair | Agree |", "|---|---|"]
    for (a, b), (k, n) in ag.pairwise_verdict.items():
        rows.append(f"| {a} – {b} | {_pct(k, n)} ({k}/{n}) |")
    k, n = ag.all_agree
    rows.append(f"| all critics | {_pct(k, n)} ({k}/{n}) |")
    k, n = ag.corroborated_groups
    rows += [
        "",
        f"Issue groups raised by 2+ critics: {_pct(k, n)} ({k}/{n}). Low corroboration is "
        "expected by design: each critic owns one dimension, so critics rarely flag the same "
        "span — single-source issues are what the adjudicator exists to check.",
    ]
    return rows


def _adjudicator(results: EvalResults) -> list[str]:
    a = results.adjudicator
    total_dismissed = a.dismissed_false_flags + a.dismissed_true_catches
    return [
        f"- Cases adjudicated: {a.cases_adjudicated}; short-circuited (unanimous confident "
        f"pass): {a.cases_short_circuited}, of which really clean: {a.short_circuit_correct}; "
        f"fell back to union after an adjudicator error: {a.cases_fallback}.",
        f"- Issue groups ruled on: {a.groups_confirmed + a.groups_dismissed} "
        f"({a.groups_confirmed} confirmed, {a.groups_dismissed} dismissed).",
        f"- Dismissals that removed a false flag: {a.dismissed_false_flags}/{total_dismissed}; "
        f"dismissals that threw away a real catch: {a.dismissed_true_catches}/{total_dismissed}.",
        f"- Pass/fail verdicts the adjudicator fixed vs the raw union: {a.verdicts_fixed}; "
        f"broke: {a.verdicts_broken}.",
    ]


def _judge(results: EvalResults) -> list[str]:
    labels_short = {SYSTEM: "system", BASELINE: "baseline", UNION: "union"}
    cfg = results.config
    judge_model = cfg.model(cfg.roles.match_judge).model_id
    rows = [
        f"Completeness matches come from the LLM judge ({judge_model}); factual/logic matches "
        f"from span overlap (≥ {cfg.evaluation.span_min_overlap:.0%} of the shorter span)."
        " The judge saw descriptions only, never which variant raised them.",
        "",
        "| Variant | Planted caught via span | via judge |",
        "|---|---|---|",
    ]
    for v in results.variants:
        s = results.scores[v]
        rows.append(f"| {v} | {s.caught_span} | {s.caught_judge} |")
    cost = sum(c.cost_usd for c in results.judge_calls)
    rows += [
        "",
        f"Judge decisions: {len(results.judge_matches)} complaint(s) matched, "
        f"{results.judge_rejections} rejected · judge cost ${cost:.4f} "
        "(not charged to any variant).",
        "",
        "### Judge matches — spot-check these",
        "",
    ]
    if not results.judge_matches:
        return rows + ["None."]
    rows += [
        "| Case | Planted omission | Complaint | Judge's reasoning | Raised by |",
        "|---|---|---|---|---|",
    ]
    for m in sorted(results.judge_matches, key=lambda m: m.case_id):
        by = ", ".join(labels_short.get(v, v.replace("critic:", "")) for v in m.variants)
        rows.append(
            f"| `{m.case_id}` | {_cell(m.missing, 90)} | {_cell(m.complaint, 120)} | "
            f"{_cell(m.reasoning, 120)} | {by} |"
        )
    return rows


def _per_case(results: EvalResults) -> list[str]:
    rows = [
        "| Case | Tags | Planted | System caught | Baseline caught | System FF | Baseline FF | "
        "System verdict | Baseline verdict |",
        "|" + "---|" * 9,
    ]
    sys_s, base_s = results.scored[SYSTEM], results.scored[BASELINE]

    def verdict(sc: ScoredCase | None) -> str:
        if sc is None:
            return "error"
        word = "pass" if sc.passed else "fail" if sc.passed is False else "none"
        return f"{word} {'✓' if sc.verdict_correct else '✗'}"

    for case in results.cases:
        s, b = sys_s.get(case.id), base_s.get(case.id)
        n = len(case.planted)
        rows.append(
            f"| `{case.id}` | {', '.join(t for t in case.tags if not t.startswith('domain-'))} "
            f"| {n} | {sum(s.caught) if s else '–'}/{n} | {sum(b.caught) if b else '–'}/{n} | "
            f"{len(s.false_flags) if s else '–'} | {len(b.false_flags) if b else '–'} | "
            f"{verdict(s)} | {verdict(b)} |"
        )
    return rows


def _health(results: EvalResults) -> list[str]:
    errors = [
        f"- `{r.case_id}`: system error {r.system_error}"
        for r in results.runs.values()
        if r.system_error
    ] + [
        f"- `{r.case_id}`: baseline error {r.baseline_error}"
        for r in results.runs.values()
        if r.baseline_error
    ]
    degraded = [
        r.case_id
        for r in results.runs.values()
        if r.system is not None and r.system.verdict.status != "ok"
    ]
    failures = sum(len(r.system.failures) for r in results.runs.values() if r.system)
    retries = sum(
        not c.success
        for r in results.runs.values()
        for c in ([*r.system.calls] if r.system else []) + (r.baseline.calls if r.baseline else [])
    )
    return [
        f"- Cases with a crashed run: {len(errors)}",
        *errors,
        f"- Degraded verdicts (a critic or the adjudicator failed): {len(degraded)}"
        + (f" — {', '.join(f'`{d}`' for d in degraded)}" if degraded else ""),
        f"- Critic failures after retries: {failures}",
        f"- Failed HTTP attempts that were retried: {retries}",
    ]


def _code_version() -> str:
    """Short git commit of the code that produced the report (+ "-dirty" if uncommitted)."""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{rev}-dirty" if dirty else rev


def render(results: EvalResults, *, note: str | None = None) -> str:
    """The full markdown report."""
    labels = _labels(results)
    s = results.scores[SYSTEM]
    clean = sum(c.is_clean for c in results.cases if c.id in results.runs)
    spend = (
        sum(r.system.cost_usd for r in results.runs.values() if r.system)
        + sum(r.baseline.cost_usd for r in results.runs.values() if r.baseline)
        + sum(c.cost_usd for c in results.judge_calls)
    )
    lines = [
        "# Evaluation: multi-critic arbitration vs a single reviewer",
        "",
        f"> Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC"
        + (f" from run `{results.run_dir}`" if results.run_dir else "")
        + f" · {len(results.runs)} cases ({clean} clean, {len(results.runs) - clean} with "
        f"{s.planted_total} hand-planted issues) · total spend ${spend:.3f} · code "
        f"`{_code_version()}` · logic critic "
        f"`{results.config.model(results.config.roles.critics[Dimension.LOGIC]).model_id}`",
        "",
        *([f"> **Note:** {note}", ""] if note else []),
        "## Headline",
        "",
        *_headline(results),
        "",
        "## Results",
        "",
        "Every row except the single reviewer is computed from the SAME system run (ablations "
        "re-use the critics' stored outputs), so they see identical critic findings.",
        "",
        *_results_table(results),
        "",
        "*Recall per type uses the PLANTED type. False flags = findings that match no planted "
        "issue (every finding on a clean case is one). s / case = critical-path LLM time "
        "(slowest critic + adjudicator), excluding queueing and retry waits.*",
        "",
        "## Sensitivity: completeness restatements of planted errors",
        "",
        *_sensitivity(results),
        "",
        "## Paired comparisons",
        "",
        *[line for c in results.comparisons for line in _comparison(c, labels)],
        "## How matches were made",
        "",
        *_judge(results),
        "",
        "## Critic analytics",
        "",
        *_critics(results),
        "",
        "## What the adjudicator changed",
        "",
        *_adjudicator(results),
        "",
        "## Misses",
        "",
        *_misses(results),
        "## False flags on clean cases",
        "",
        *_false_flags_on_clean(results),
        "",
        "## Per-case results",
        "",
        *_per_case(results),
        "",
        "## Run health",
        "",
        *_health(results),
        "",
        "## Caveats",
        "",
        f"- **Small sample.** {s.planted_total} planted issues over {len(results.runs)} cases; "
        "the 95% intervals above are wide, and differences of a few catches are within noise.",
        "- **One author.** All answer keys were written by one person; what counts as an error "
        "reflects their judgement.",
        "- **LLM judge for omissions.** Completeness matches depend on the judge; every one is "
        "listed above for manual checking.",
        "- **No live fact-checking.** Critics, adjudicator and baseline all judge facts from "
        "model knowledge only.",
        "- **Prices** are list prices (DeepSeek peak rates; Gemini paid-tier rates although the "
        "free tier was used).",
        "",
    ]
    return "\n".join(lines)
