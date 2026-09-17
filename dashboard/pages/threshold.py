from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard.components.common import dataset_name, info_panel, method_name, page_header
from dashboard.data_loader import DashboardData, selected_threshold, threshold_options, threshold_row
from dashboard.styles import AMBER, CYAN, RED, TEAL, plotly_layout


def render(data: DashboardData) -> None:
    page_header(
        "Retrospective sensitivity analysis",
        "Threshold trade-off",
        "Explore the committed probability sweep. The operational threshold remains the validation-selected value; test curves are not a tuning surface.",
    )
    tradeoff = data.metrics["threshold_tradeoff"]
    c1, c2, c3 = st.columns([1.3, 1, .7])
    with c1:
        dataset = st.selectbox("Dataset", sorted(tradeoff.dataset.unique()), format_func=dataset_name, key="threshold_dataset")
    models = [model for model in ("GRU", "LSTM", "TRANSFORMER", "LOGISTIC REGRESSION") if model in tradeoff[tradeoff.dataset == dataset].model.unique()]
    with c2:
        model = st.selectbox("Model", models, format_func=method_name, key="threshold_model")
    seeds = sorted(tradeoff[(tradeoff.dataset == dataset) & (tradeoff.model == model)].seed.unique().astype(int))
    with c3:
        seed = st.selectbox("Seed", seeds, key="threshold_seed")

    options = threshold_options(data, dataset, model, int(seed))
    selected = selected_threshold(data, dataset, model, int(seed))
    default_index = int(abs(options - selected).argmin())
    threshold = st.select_slider(
        "Explored threshold (snaps to committed sweep points)", options=options.tolist(),
        value=float(options[default_index]), format_func=lambda value: f"{value:.2f}",
    )
    row = threshold_row(data, dataset, model, int(seed), float(threshold))
    metrics = st.columns(5)
    metrics[0].metric("Coverage", f"{row.detection_coverage * 100:.1f}%")
    metrics[1].metric("Batch FAR", f"{row.batch_far * 100:.1f}%")
    metrics[2].metric("Lead time", f"{row.mean_lead_time:.2f} batches")
    metrics[3].metric("Miss rate", f"{row.missed_drift_rate * 100:.1f}%")
    metrics[4].metric("Adapt./100", f"{row.adaptations_per_100_batches:.2f}")

    subset = tradeoff[(tradeoff.dataset == dataset) & (tradeoff.model == model) & (tradeoff.seed == seed)].sort_values("threshold")
    fig = go.Figure()
    series = [
        ("detection_coverage", "Warning coverage", TEAL),
        ("batch_far", "Batch FAR", RED),
        ("missed_drift_rate", "Missed-drift rate", AMBER),
    ]
    for column, label, color in series:
        fig.add_trace(go.Scatter(x=subset.threshold, y=subset[column], mode="lines+markers", name=label, line={"color": color, "width": 2.3}))
    fig.add_vline(x=selected, line_color=CYAN, line_dash="dash", annotation_text=f"validation-selected {selected:.2f}")
    fig.add_vline(x=float(threshold), line_color="white", line_dash="dot", annotation_text="explored")
    fig.update_layout(**plotly_layout("Coverage, false alarms, and misses", height=470))
    fig.update_xaxes(title="Decision threshold")
    fig.update_yaxes(title="Rate", range=[-0.03, 1.03], tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    lead_fig = go.Figure()
    lead_fig.add_trace(go.Scatter(
        x=subset.batch_far, y=subset.mean_lead_time, mode="lines+markers+text",
        marker={"color": subset.threshold, "colorscale": "Tealgrn", "showscale": True,
                "colorbar": {"title": "Threshold"}},
        line={"color": "rgba(76,201,240,.55)", "width": 2},
        text=[f"{value:.2f}" if value in {selected, float(threshold)} else "" for value in subset.threshold],
        textposition="top center", name="Committed threshold sweep",
        customdata=subset[["threshold", "detection_coverage"]],
        hovertemplate="Threshold %{customdata[0]:.2f}<br>Batch FAR %{x:.1%}<br>Lead %{y:.2f} batches<br>Coverage %{customdata[1]:.1%}<extra></extra>",
    ))
    points = subset[subset.threshold.map(lambda value: value in {selected, float(threshold)})]
    lead_fig.add_trace(go.Scatter(
        x=points.batch_far, y=points.mean_lead_time, mode="markers",
        marker={"color": [CYAN if value == selected else "white" for value in points.threshold], "size": 14, "symbol": "diamond"},
        name="Selected / explored",
        hovertemplate="Batch FAR %{x:.1%}<br>Lead %{y:.2f} batches<extra></extra>",
    ))
    lead_fig.update_layout(**plotly_layout("Lead time versus false-alarm rate", height=430))
    lead_fig.update_xaxes(title="Batch FAR", tickformat=".0%")
    lead_fig.update_yaxes(title="Mean lead time (batches)")
    st.plotly_chart(lead_fig, use_container_width=True, config={"displaylogo": False})

    info_panel(
        "The cyan marker is the threshold selected on validation data before test evaluation. Moving the white marker only inspects a precomputed test sensitivity curve; it does not replace the reported threshold or result.",
        warning=True,
    )
    st.caption("The committed sweep contains thresholds 0.05–0.95. No detector curve appears because binary-only alarm flags do not define a genuine continuous threshold sweep.")
