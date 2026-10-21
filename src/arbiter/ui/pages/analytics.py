"""Evaluation analytics: the Phase 5 comparison, ablations and critic analytics."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from arbiter.ui.data import eval_reports, load_metrics

# reference palette, categorical slots 1-2 (validated adjacent pair); text stays in ink tokens
SYSTEM_COLOR, BASELINE_COLOR = "#2a78d6", "#eb6834"
NAMES = {
    "system": "Full system",
    "baseline": "Single reviewer",
    "union": "Union (no adjudicator)",
    "critic:factual": "Factual critic alone",
    "critic:logic": "Logic critic alone",
    "critic:completeness": "Completeness critic alone",
}

st.title("Evaluation analytics")
reports = eval_reports()
if not reports:
    st.info("No evaluation results yet. Run `python scripts/evaluate.py`.")
    st.stop()

report = st.radio("Evaluation run", reports, format_func=lambda r: r.label, horizontal=True)
m = load_metrics(report)
if "optimistic" in report.label:
    st.warning(
        "v2 prompts were changed after seeing v1 results on these same 40 cases, so these "
        "numbers are optimistic. v1 is the pre-registered result.",
        icon=":material/warning:",
    )
var = m["variants"]
s, b = var["system"], var["baseline"]
cmp_ = m["comparisons"][0]

# ---------------------------------------------------------------- headline tiles
st.caption(f"{m['n_cases']} cases · {s['planted_total']} hand-planted issues")
c = st.columns(5)
c[0].metric(
    "Recall: system vs single",
    f"{s['recall']:.0%} vs {b['recall']:.0%}",
    help=f"Paired exact McNemar p = {cmp_['recall_p']:.3f}",
)
c[1].metric("False flags", f"{s['false_flags']} vs {b['false_flags']}")
c[2].metric("Precision", f"{s['precision']:.0%} vs {b['precision']:.0%}")
c[3].metric("Verdict accuracy", f"{s['accuracy']:.0%} vs {b['accuracy']:.0%}")
c[4].metric(
    "Cost per case",
    f"${s['mean_cost_usd']:.4f}",
    f"{s['mean_cost_usd'] / b['mean_cost_usd']:.1f}× single reviewer"
    if b["mean_cost_usd"]
    else None,
    delta_color="inverse",
)

# ---------------------------------------------------------------- comparison table
st.subheader("All variants")
table = pd.DataFrame(
    [
        {
            "variant": NAMES.get(k, k),
            "recall": v["recall"],
            "factual": v["by_type"]["factual"]["caught"] / max(v["by_type"]["factual"]["total"], 1),
            "logic": v["by_type"]["logic"]["caught"] / max(v["by_type"]["logic"]["total"], 1),
            "completeness": v["by_type"]["completeness"]["caught"]
            / max(v["by_type"]["completeness"]["total"], 1),
            "precision": v["precision"],
            "false flags": v["false_flags"],
            "verdict acc.": v["accuracy"],
            "$ / case": v["mean_cost_usd"],
            "s / case": v["mean_latency_s"],
        }
        for k, v in var.items()
    ]
)
pct = st.column_config.NumberColumn(format="percent")
st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    column_config={
        "recall": pct,
        "factual": pct,
        "logic": pct,
        "completeness": pct,
        "precision": pct,
        "verdict acc.": pct,
        "$ / case": st.column_config.NumberColumn(format="%.4f"),
        "s / case": st.column_config.NumberColumn(format="%.1f"),
    },
)
st.caption(
    "Ablations reuse the full system's stored critic outputs, so every row saw identical "
    "critic findings. s / case = critical-path LLM time."
)


# ---------------------------------------------------------------- charts (2 series each)
def _bars(df: pd.DataFrame, y: str, y_title: str, fmt: str) -> alt.Chart:
    chart: alt.Chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, size=22)
        .encode(
            x=alt.X("group:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("who:N", sort=["Full system", "Single reviewer"]),
            y=alt.Y(f"{y}:Q", title=y_title, axis=alt.Axis(format=fmt, grid=True)),
            color=alt.Color(
                "who:N",
                title=None,
                scale=alt.Scale(
                    domain=["Full system", "Single reviewer"],
                    range=[SYSTEM_COLOR, BASELINE_COLOR],
                ),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("who:N", title="Variant"),
                alt.Tooltip("group:N", title="Group"),
                alt.Tooltip(f"{y}:Q", title=y_title, format=fmt),
                alt.Tooltip("detail:N", title="Count"),
            ],
        )
        .properties(height=260)
    )
    return chart


left, right = st.columns(2)
with left:
    st.markdown("**Recall by planted issue type**")
    rows = []
    for key, who in (("system", "Full system"), ("baseline", "Single reviewer")):
        for t in ("factual", "logic", "completeness"):
            bt = var[key]["by_type"][t]
            rows.append(
                {
                    "who": who,
                    "group": t,
                    "recall": bt["caught"] / max(bt["total"], 1),
                    "detail": f"{bt['caught']}/{bt['total']}",
                }
            )
    st.altair_chart(_bars(pd.DataFrame(rows), "recall", "Recall", ".0%"), width="stretch")
with right:
    st.markdown("**False flags**")
    rows = []
    for key, who in (("system", "Full system"), ("baseline", "Single reviewer")):
        v = var[key]
        rows += [
            {
                "who": who,
                "group": "strict rule",
                "flags": v["false_flags"],
                "detail": str(v["false_flags"]),
            },
            {
                "who": who,
                "group": "excl. restatements",
                "flags": v.get("false_flags_excl_restatements", v["false_flags"]),
                "detail": str(v.get("false_flags_excl_restatements", v["false_flags"])),
            },
            {
                "who": who,
                "group": "≥ medium only",
                "flags": v["false_flags_ge_medium"],
                "detail": str(v["false_flags_ge_medium"]),
            },
        ]
    st.altair_chart(_bars(pd.DataFrame(rows), "flags", "False flags", "d"), width="stretch")

# ---------------------------------------------------------------- critic & adjudicator
st.subheader("Critic analytics")
crit = pd.DataFrame(
    [
        {
            "critic": x["critic"],
            "model": x["model_id"],
            "valid issues": x["valid_issues"],
            "true positives": x["true_positives"],
            "precision": x["precision"],
            "unique catches": x["unique_catches"],
            "overruled": f"{x['overruled']}/{x['adjudicated']}",
            "overruled & really false": x["overruled_false_flags"],
            "hallucinated quotes": x["hallucination_rate"],
        }
        for x in m["critics"]
    ]
)
st.dataframe(
    crit,
    hide_index=True,
    width="stretch",
    column_config={"precision": pct, "hallucinated quotes": pct},
)
adj = m["adjudicator"]
c = st.columns(4)
c[0].metric(
    "Short-circuited",
    adj["cases_short_circuited"],
    help=f"{adj['short_circuit_correct']} of them were really clean",
)
c[1].metric(
    "Groups confirmed / dismissed", f"{adj['groups_confirmed']} / {adj['groups_dismissed']}"
)
c[2].metric("Dismissals that removed a false flag", adj["dismissed_false_flags"])
c[3].metric("Dismissals that lost a real catch", adj["dismissed_true_catches"])

with st.expander("Full markdown report"):
    if report.report_path.exists():
        st.markdown(report.report_path.read_text(encoding="utf-8"))
