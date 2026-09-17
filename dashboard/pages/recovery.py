from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.common import dataset_name, info_panel, method_name, page_header
from dashboard.data_loader import DashboardData, comparison_view
from dashboard.styles import MODEL_COLORS, plotly_layout


def _recovery_frame(data: DashboardData, view: str, dataset: str | None) -> pd.DataFrame:
    frame = comparison_view(data, view, dataset)
    columns = [
        "method", "accuracy_recovery_batches", "recovery_coverage",
        "total_adaptations", "adaptations_per_100_batches", "false_adaptations",
        "true_event_related_adaptations", "adaptation_precision",
    ]
    return frame[[column for column in columns if column in frame]].copy()


def render(data: DashboardData) -> None:
    page_header(
        "Causal raw-stream replay",
        "Recovery & adaptation",
        "Compare observed accuracy recovery with the reset frequency and precision required to obtain it.",
    )
    c1, c2 = st.columns([1.25, 1])
    with c1:
        view = st.selectbox("Aggregation", ["Macro (equal dataset weight)", "Pooled (event/row weighted)", "Per dataset"], key="recovery_view")
    dataset = None
    with c2:
        if view == "Per dataset":
            valid = sorted(data.tables["per_dataset_results"].query("evaluation_valid == True").dataset.unique())
            dataset = st.selectbox("Dataset", valid, format_func=dataset_name, key="recovery_dataset")
        else:
            st.selectbox("Dataset", ["Defined by aggregation"], disabled=True, key="recovery_dataset_disabled")
    frame = _recovery_frame(data, view, dataset)
    chart = frame.copy()
    chart["Method"] = chart.method.map(method_name)
    colors = {method_name(key): value for key, value in MODEL_COLORS.items()}

    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            chart, x="adaptations_per_100_batches", y="recovery_coverage", color="Method",
            size="total_adaptations" if "total_adaptations" in chart else None,
            color_discrete_map=colors, text="Method", title="Recovery coverage vs adaptation cost",
        )
        fig.update_layout(**plotly_layout(height=440))
        fig.update_xaxes(title="Adaptations per 100 batches")
        fig.update_yaxes(title="Recovery coverage", tickformat=".0%")
        fig.update_traces(textposition="top center", hovertemplate="%{text}<br>Adapt./100 %{x:.2f}<br>Recovery coverage %{y:.1%}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    with right:
        fig = px.scatter(
            chart.dropna(subset=["accuracy_recovery_batches"]),
            x="adaptations_per_100_batches", y="accuracy_recovery_batches", color="Method",
            color_discrete_map=colors, text="Method", title="Observed recovery vs reset frequency",
        )
        fig.update_layout(**plotly_layout(height=440))
        fig.update_xaxes(title="Adaptations per 100 batches")
        fig.update_yaxes(title="Batches until rolling accuracy recovery")
        fig.update_traces(textposition="top center", hovertemplate="%{text}<br>Adapt./100 %{x:.2f}<br>Recovery %{y:.2f} batches<extra></extra>")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    table = frame.copy()
    table["method"] = table.method.map(method_name)
    table = table.rename(columns={
        "method": "Method", "accuracy_recovery_batches": "Recovery batches",
        "recovery_coverage": "Recovery coverage", "total_adaptations": "Resets",
        "adaptations_per_100_batches": "Resets / 100", "false_adaptations": "False resets",
        "true_event_related_adaptations": "Event-related resets", "adaptation_precision": "Adaptation precision",
    })
    for column in table.columns[1:]:
        table[column] = table[column].map(lambda value: "N/A" if pd.isna(value) else f"{float(value):.4f}")
    st.dataframe(table, hide_index=True, use_container_width=True)

    info_panel(
        "At each deduplicated alert episode, replay resets a Hoeffding tree and warm-starts it with only the latest 300 already-labelled instances. It then resumes predict-before-learn. Recovery is the first five-batch rolling accuracy to reach 95% of the pre-drift baseline.",
    )
    info_panel(
        "Fast observed recovery is not automatically better. Read it with recovery coverage, resets per 100 batches, false resets, and adaptation precision; frequent resets can create apparently fast recovery while imposing high operational cost.",
        warning=True,
    )
    with st.expander("Inspect event-level recovery records"):
        events = data.metrics["recovery_events"].copy()
        if dataset:
            events = events[events.dataset == dataset]
        events["dataset"] = events.dataset.map(dataset_name)
        events["method"] = events.method.map(method_name)
        st.dataframe(events, hide_index=True, use_container_width=True, height=360)
