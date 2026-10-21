"""Batch results: every stored arbitration (or one batch) in a sortable table."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from arbiter.ui.data import open_store

store = open_store()
st.title("Batch results")

batches = store.batch_ids()
choice = st.selectbox(
    "Show", ["All arbitrations", *batches], help="Batches come from POST /v1/arbitrate/batch."
)
rows = store.list_summaries(limit=1000, batch_id=None if choice == "All arbitrations" else choice)
if not rows:
    st.info("Nothing stored yet.")
    st.stop()

VERDICT = {True: "✅ pass", False: "❌ fail", None: "⚠️ none"}
df = pd.DataFrame(
    [
        {
            "created": r.created_at,
            "excerpt": r.excerpt,
            "verdict": VERDICT[r.passed],
            "score": r.overall_score,
            "issues": r.issue_count,
            "dismissed": r.dismissed_count,
            "confidence": r.confidence,
            "status": r.status,
            "cost $": r.cost_usd,
            "seconds": r.latency_s,
            "batch": r.batch_id or "",
            "id": r.id,
        }
        for r in rows
    ]
)
st.caption(f"{len(df)} arbitration(s). Click a column header to sort; select a row to open it.")
event = st.dataframe(
    df,
    hide_index=True,
    width="stretch",
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "created": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
        "excerpt": st.column_config.TextColumn(width="large"),
        "score": st.column_config.NumberColumn(format="%d/10"),
        "confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
        "cost $": st.column_config.NumberColumn(format="%.4f"),
        "seconds": st.column_config.NumberColumn(format="%.1f"),
    },
)
selected = event.selection.rows if event and event.selection else []
if selected:
    st.session_state.selected_id = df.iloc[selected[0]]["id"]
    st.switch_page("pages/verdict.py")

c = st.columns(4)
c[0].metric("Pass rate", f"{(df['verdict'] == '✅ pass').mean():.0%}")
c[1].metric("Mean score", f"{df['score'].mean():.1f}/10" if df["score"].notna().any() else "–")
c[2].metric("Mean confidence", f"{df['confidence'].mean():.0%}")
c[3].metric("Total cost", f"${df['cost $'].sum():.3f}")
