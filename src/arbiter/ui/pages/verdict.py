"""Verdict view: highlighted text with clickable evidence chains + critic comparison."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from arbiter.schemas.verdict import ArbitrationResult
from arbiter.storage.db import ArbitrationSummary
from arbiter.ui.data import ApiError, open_store, submit_arbitration
from arbiter.ui.render import agreement_rows, critic_columns, evidence_chain, highlight_html, marks

store = open_store()


def _is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except AttributeError:
        return False


def _label(s: ArbitrationSummary) -> str:
    verdict = {True: "PASS", False: "FAIL", None: "no verdict"}[s.passed]
    score = f"{s.overall_score}/10" if s.overall_score is not None else "–"
    return f"{s.created_at:%m-%d %H:%M} · {verdict} {score} · {s.excerpt[:60]}"


# ---------------------------------------------------------------- sidebar: pick or create
with st.sidebar:
    st.header("Arbitrations")
    with st.form("new", clear_on_submit=False):
        st.caption("New arbitration (sent to the API)")
        question = st.text_area("Original question (optional)", height=80)
        output = st.text_area("LLM output to evaluate", height=160)
        go = st.form_submit_button("Arbitrate", type="primary", width="stretch")
    if go:
        if not output.strip():
            st.error("Paste an output to evaluate.")
        else:
            with st.spinner("Critics running in parallel…"):
                try:
                    created = submit_arbitration(output, question)
                    st.session_state.selected_id = created["id"]
                except ApiError as exc:
                    st.error(str(exc))

    recent = store.list_summaries(limit=200)
    ids = [s.id for s in recent]
    wanted = st.query_params.get("id") or st.session_state.get("selected_id")
    if wanted and wanted not in ids and store.get(wanted) is not None:
        ids.insert(0, wanted)
    labels = {s.id: _label(s) for s in recent}
    chosen = (
        st.selectbox(
            "Stored arbitrations",
            ids,
            index=ids.index(wanted) if wanted in ids else 0,
            format_func=lambda i: labels.get(i, i),
        )
        if ids
        else None
    )

if chosen is None:
    st.title("Verdict Explorer")
    st.info(
        "No arbitrations stored yet. Submit one from the sidebar, call `POST /v1/arbitrate`, "
        "or run `python scripts/seed_demo.py`."
    )
    st.stop()

st.session_state.selected_id = chosen
st.query_params["id"] = chosen
result: ArbitrationResult | None = store.get(chosen)
if result is None:
    st.error(f"Arbitration {chosen} not found.")
    st.stop()
v = result.verdict

# ---------------------------------------------------------------- verdict header
st.title("Verdict")
VERDICT_WORDS: dict[bool | None, str] = {True: "✅ PASS", False: "❌ FAIL", None: "⚠️ NO VERDICT"}
c = st.columns(6)
c[0].metric("Verdict", VERDICT_WORDS[v.passed], help="Fails if any confirmed issue is ≥ medium.")
c[1].metric("Score", f"{v.overall_score}/10" if v.overall_score is not None else "–")
c[2].metric("Confidence", f"{v.confidence:.0%}")
c[3].metric("Confirmed / dismissed", f"{len(v.confirmed_issues)} / {len(v.dismissed_flags)}")
c[4].metric("Cost", f"${result.cost_usd:.4f}")
c[5].metric("Time", f"{result.latency_s:.1f}s")
st.caption(
    f"id `{result.id}` · status **{v.status}** · adjudicator "
    f"{'skipped (unanimous confident pass)' if v.short_circuited else v.adjudicator}"
)
if v.low_confidence_dimensions:
    st.warning(
        "Low-confidence dimension(s): "
        + ", ".join(d.value for d in v.low_confidence_dimensions)
        + " — a critic failed or was unsure, so this part of the verdict is weak.",
        icon=":material/warning:",
    )
st.markdown(f"**Summary.** {v.summary}")
if result.question:
    with st.expander("Original question"):
        st.write(result.question)

# ---------------------------------------------------------------- highlighted text
st.subheader("Evaluated text")
height = min(900, max(280, 140 + len(result.output) // 3))
# st.iframe runs this HTML with JavaScript enabled: highlight_html escapes every untrusted string
st.iframe(highlight_html(result, dark=_is_dark()), height=height)

unlocated = [m for m in marks(result) if m.start is None]
with st.expander(f"All flags as a list ({len(marks(result))}) — accessible view"):
    for m in marks(result):
        st.markdown(
            f"**{m.id}** · {m.kind} · {m.title}" + (" · *no location*" if m.start is None else "")
        )
        for step in evidence_chain(result, m.id):
            st.markdown(f"- *{step.title}* — {step.body} {f'({step.meta})' if step.meta else ''}")
if unlocated:
    st.caption(
        f"{len(unlocated)} flag(s) have no location in the text (e.g. a critic quoted words that "
        "don't appear) — see the list above."
    )

# ---------------------------------------------------------------- critic comparison
st.subheader("Critic comparison")
cols = critic_columns(result)
for col, data in zip(st.columns(len(cols)), cols, strict=True):
    with col:
        st.markdown(f"#### {data.critic}")
        st.caption(data.model)
        if data.status != "ok":
            st.error(data.status)
            st.write(data.analysis)
            continue
        st.markdown(f"Score **{data.score}/5** · self-confidence **{data.confidence:.0%}**")
        st.write(data.analysis)
        st.markdown(f"**Issues ({len(data.valid_issues)})**")
        for i in data.valid_issues:
            st.markdown(f"- `{i.severity.value}` “{i.quote[:90]}” — {i.problem}")
        for i in data.invalid_issues:
            st.markdown(f"- ~~“{i.quote[:90]}”~~ *quote not in text*")
        if data.verified:
            st.markdown("**Verified correct**")
            for q in data.verified:
                st.markdown(f"- “{q[:90]}”")

agreements, disagreements = agreement_rows(result)
a, d = st.columns(2)
with a:
    st.markdown(f"**Agreements** — raised by 2+ critics ({len(agreements)})")
    if agreements:
        st.dataframe(pd.DataFrame(agreements), hide_index=True, width="stretch")
    else:
        st.caption("None. Critics each own one dimension, so overlap is rare by design.")
with d:
    st.markdown(f"**Disagreements** — resolved by the adjudicator ({len(disagreements)})")
    if disagreements:
        st.dataframe(pd.DataFrame(disagreements), hide_index=True, width="stretch")
    else:
        st.caption("None.")
