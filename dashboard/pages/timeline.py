from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard.components.common import dataset_name, info_panel, method_name, page_header
from dashboard.data_loader import (
    BASELINE_METHODS,
    SUPERVISED_METHODS,
    DashboardData,
    get_prediction,
    valid_timeline_datasets,
    warning_annotations,
)
from dashboard.styles import AMBER, CYAN, MODEL_COLORS, RED, TEAL, plotly_layout


def render(data: DashboardData) -> None:
    page_header(
        "Event-level inspection",
        "Drift timeline explorer",
        "Inspect committed test predictions, strict warning windows, deduplicated episodes, false alerts, and missed events.",
    )
    datasets = valid_timeline_datasets(data)
    preferred_dataset = "synth_sea_abrupt"
    c1, c2, c3 = st.columns([1.35, 1.15, .8])
    with c1:
        dataset = st.selectbox(
            "Dataset", datasets, format_func=dataset_name, key="timeline_dataset",
            index=datasets.index(preferred_dataset) if preferred_dataset in datasets else 0,
        )
    available_methods = sorted({key[1] for key in data.predictions if key[0] == dataset}) + list(BASELINE_METHODS)
    presentation_order = ("GRU", "LOGISTIC REGRESSION", "LSTM", "TRANSFORMER", *BASELINE_METHODS)
    ordered = [name for name in presentation_order if name in available_methods]
    with c2:
        method = st.selectbox("Model / detector", ordered, format_func=method_name, key="timeline_method")
    seeds = sorted({key[2] for key in data.predictions if key[0] == dataset and key[1] == method})
    with c3:
        seed = st.selectbox("Seed", seeds or [11], disabled=method in BASELINE_METHODS, key="timeline_seed")

    frame = get_prediction(data, dataset, method, int(seed))
    onsets = [value for value in data.onsets(dataset) if frame.batch_id.min() <= value <= frame.batch_id.max() + 1]
    annotated = warning_annotations(frame.batch_id, frame.alert, onsets, data.horizon, data.cooldown)

    fig = go.Figure()
    for onset in onsets:
        fig.add_vrect(
            x0=onset - data.horizon, x1=onset, fillcolor="rgba(45,212,191,.09)",
            line_width=0, layer="below",
        )
        fig.add_vline(x=onset, line_color=AMBER, line_dash="dash", line_width=1.4)
    if method in BASELINE_METHODS:
        fig.add_trace(go.Scatter(
            x=frame.batch_id, y=frame.display_value, mode="lines", line_shape="hv",
            line={"color": MODEL_COLORS[method], "width": 2.3}, name=f"{method} binary alarm",
            hovertemplate="Batch %{x}<br>Alarm %{y:.0f}<extra></extra>",
        ))
        y_title = "Binary alarm flag"
    else:
        fig.add_trace(go.Scatter(
            x=frame.batch_id, y=frame.probability, mode="lines",
            line={"color": MODEL_COLORS[method], "width": 2.1}, name="Predicted probability",
            hovertemplate="Batch %{x}<br>Probability %{y:.3f}<extra></extra>",
        ))
        threshold = float(frame.decision_threshold.iloc[0])
        fig.add_hline(y=threshold, line_color="rgba(237,247,255,.65)", line_dash="dot", annotation_text=f"selected τ={threshold:.2f}")
        y_title = "P(drift onset in next 5 batches)"

    markers = [
        (annotated["successful"], TEAL, "diamond", "Successful warning episode"),
        (annotated["false"], RED, "x", "False alert episode"),
    ]
    y_max = max(1.0, float(frame.display_value.max()))
    for batches, color, symbol, label in markers:
        if batches:
            fig.add_trace(go.Scatter(
                x=batches, y=[y_max * .94] * len(batches), mode="markers",
                marker={"color": color, "size": 12, "symbol": symbol}, name=label,
                hovertemplate=f"{label}<br>Batch %{{x}}<extra></extra>",
            ))
    if annotated["missed"]:
        fig.add_trace(go.Scatter(
            x=annotated["missed"], y=[y_max * .08] * len(annotated["missed"]), mode="markers",
            marker={"color": AMBER, "size": 12, "symbol": "triangle-down"}, name="Missed onset",
            hovertemplate="Missed onset<br>Batch %{x}<extra></extra>",
        ))
    fig.update_layout(**plotly_layout(height=520))
    fig.update_xaxes(title="Stream batch")
    fig.update_yaxes(title=y_title, range=[-0.05, 1.08])
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "responsive": True})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Alert episodes", len(annotated["episodes"]))
    m2.metric("Successful warnings", len(annotated["successful"]))
    m3.metric("False episodes", len(annotated["false"]))
    m4.metric("Missed events", len(annotated["missed"]))
    info_panel(
        f"Shaded bands are the only eligible windows: [onset−{data.horizon}, onset). Episodes are deduplicated with a "
        f"{data.cooldown}-batch cooldown and matched one-to-one, so one alert cannot receive credit twice."
    )
    st.caption("Elec2 is intentionally absent: without verified onsets, timeline success, misses, and lead time would be undefined.")
