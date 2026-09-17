"""DriftSentinel faculty presentation dashboard.

Launch from the repository root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.data_loader import DashboardDataError, load_dashboard_data  # noqa: E402
from dashboard.pages import comparison, findings, overview, recovery, threshold, timeline  # noqa: E402
from dashboard.styles import apply_styles  # noqa: E402


st.set_page_config(
    page_title="DriftSentinel · Early-warning research dashboard",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="auto",
)
apply_styles()

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
