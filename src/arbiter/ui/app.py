"""Verdict Explorer — Streamlit entry point.

Run:  streamlit run src/arbiter/ui/app.py
Env:  ARBITER_API_URL (default http://localhost:8000), ARBITER_DB_PATH (shared SQLite file)
"""

import streamlit as st

st.set_page_config(page_title="Verdict Explorer", page_icon=":material/gavel:", layout="wide")

st.navigation(
    [
        st.Page(
            "pages/verdict.py", title="Verdict Explorer", icon=":material/fact_check:", default=True
        ),
        st.Page("pages/batch.py", title="Batch results", icon=":material/table_rows:"),
        st.Page("pages/analytics.py", title="Evaluation analytics", icon=":material/monitoring:"),
    ]
).run()
