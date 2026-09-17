"""DriftSentinel faculty presentation dashboard.

Launch from the repository root:
    python -m streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

from dashboard.data_loader import DashboardDataError, load_dashboard_data
from dashboard.pages import comparison, findings, overview, recovery, threshold, timeline
from dashboard.styles import apply_styles


st.set_page_config(
    page_title="DriftSentinel · Early-warning research dashboard",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="auto",
)
apply_styles()

try:
    import driftsentinel  # noqa: F401
except ModuleNotFoundError:
    interpreter = sys.executable
    st.error("The DriftSentinel project package is not installed in this Python environment.")
    st.markdown(f"**Active interpreter:** `{interpreter}`")
    st.markdown("From the repository root, install and launch with the same interpreter:")
    st.code(
        f'"{interpreter}" -m pip install -r requirements.txt\n'
        f'"{interpreter}" -m pip install -e .\n'
        f'"{interpreter}" -m streamlit run dashboard/app.py',
        language="bash",
    )
    st.stop()

try:
    data = load_dashboard_data(ROOT)
except DashboardDataError as exc:
    st.error(f"Dashboard data validation failed: {exc}")
    st.stop()

with st.sidebar:
    st.markdown('<div class="eyebrow">DriftSentinel</div>', unsafe_allow_html=True)
    st.markdown("### Early-warning lab")
    st.caption("Committed results · offline-ready · no training at startup")
    section = st.radio(
        "Navigate",
        [
            "Overview", "Drift Timeline", "Model Comparison",
            "Threshold Trade-off", "Recovery & Adaptation", "Research Findings",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"Scientific horizon: N={data.horizon} batches")
    st.caption("Repository state: finalized benchmark artifacts")

renderers = {
    "Overview": overview.render,
    "Drift Timeline": timeline.render,
    "Model Comparison": comparison.render,
    "Threshold Trade-off": threshold.render,
    "Recovery & Adaptation": recovery.render,
    "Research Findings": findings.render,
}
renderers[section](data)

st.divider()
st.caption("DriftSentinel · University research prototype · Interpret coverage together with false alarms, misses, and adaptation cost.")
