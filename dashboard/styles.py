"""Shared visual system for the projector-friendly Streamlit dashboard."""

from __future__ import annotations

import streamlit as st


BACKGROUND = "#07111f"
SURFACE = "#0d1b2d"
SURFACE_ALT = "#11243a"
TEXT = "#edf7ff"
MUTED = "#9bb0c8"
CYAN = "#4cc9f0"
TEAL = "#2dd4bf"
AMBER = "#f59e0b"
RED = "#fb7185"
GRID = "rgba(155,176,200,0.14)"

MODEL_COLORS = {
    "GRU": CYAN,
    "LOGISTIC REGRESSION": TEAL,
    "LSTM": "#a78bfa",
    "TRANSFORMER": AMBER,
    "ADWIN": "#f472b6",
    "DDM": RED,
    "KSWIN": "#94a3b8",
}


def apply_styles() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {BACKGROUND}; color: {TEXT}; }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #081729 0%, #0b1d31 100%);
            border-right: 1px solid rgba(76,201,240,.18);
        }}
        [data-testid="stSidebar"] * {{ color: {TEXT}; }}
        .block-container {{ max-width: 1420px; padding-top: 2rem; padding-bottom: 4rem; }}
        h1, h2, h3 {{ color: {TEXT}; letter-spacing: -0.02em; }}
        h1 {{ font-size: clamp(2.1rem, 4vw, 3.5rem) !important; margin-bottom: .2rem; }}
        h2 {{ font-size: 1.65rem !important; margin-top: 1.2rem; }}
        p, li, label, [data-testid="stMarkdownContainer"] {{ font-size: 1rem; line-height: 1.65; }}
        .eyebrow {{ color: {CYAN}; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; font-size: .76rem; }}
        .subtitle {{ color: {MUTED}; font-size: 1.12rem; max-width: 900px; margin-bottom: 1.25rem; }}
        .metric-card {{
            min-height: 128px; padding: 1.05rem 1.15rem; border-radius: 16px;
            background: linear-gradient(145deg, {SURFACE_ALT}, {SURFACE});
            border: 1px solid rgba(76,201,240,.18); box-shadow: 0 10px 28px rgba(0,0,0,.18);
        }}
        .metric-label {{ color: {MUTED}; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }}
        .metric-value {{ color: {TEXT}; font-size: 1.82rem; line-height: 1.2; font-weight: 720; margin: .35rem 0; }}
        .metric-note {{ color: {MUTED}; font-size: .78rem; line-height: 1.35; }}
        .info-panel {{
            padding: 1.05rem 1.15rem; border-radius: 14px; background: rgba(17,36,58,.78);
            border-left: 3px solid {CYAN}; margin: .5rem 0 1.1rem;
        }}
        .warning-panel {{ border-left-color: {AMBER}; }}
        .pipeline {{ display:flex; align-items:stretch; gap:.55rem; flex-wrap:wrap; margin:1rem 0 1.5rem; }}
        .pipe-node {{
            flex:1 1 135px; min-width:125px; padding:.85rem .75rem; border-radius:12px;
            background:{SURFACE_ALT}; border:1px solid rgba(76,201,240,.22); text-align:center;
            color:{TEXT}; font-weight:650; font-size:.86rem;
        }}
        .pipe-arrow {{ align-self:center; color:{CYAN}; font-size:1.25rem; }}
        [data-testid="stDataFrame"] {{ border: 1px solid rgba(76,201,240,.14); border-radius: 12px; overflow: hidden; }}
        [data-testid="stMetric"] {{ background:{SURFACE}; border:1px solid rgba(76,201,240,.15); padding:.8rem; border-radius:12px; }}
        .stTabs [data-baseweb="tab-list"] {{ gap:.3rem; }}
        .stTabs [data-baseweb="tab"] {{ background:{SURFACE}; border-radius:9px 9px 0 0; padding:.55rem .9rem; }}
        a {{ color: {CYAN} !important; }}
        @media (max-width: 760px) {{
            .block-container {{ padding: 1rem 1rem 3rem; }}
            .pipe-arrow {{ display:none; }}
            .metric-card {{ min-height: 110px; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def plotly_layout(title: str | None = None, height: int = 430) -> dict:
    return {
        "title": {"text": title or "", "x": 0.01, "font": {"size": 18, "color": TEXT}},
        "height": height,
        "paper_bgcolor": BACKGROUND,
        "plot_bgcolor": SURFACE,
        "font": {"color": TEXT, "family": "Inter, ui-sans-serif, system-ui"},
        "margin": {"l": 55, "r": 28, "t": 58, "b": 54},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        "hoverlabel": {"bgcolor": SURFACE_ALT, "font_color": TEXT},
        "xaxis": {"gridcolor": GRID, "zerolinecolor": GRID},
        "yaxis": {"gridcolor": GRID, "zerolinecolor": GRID},
    }
