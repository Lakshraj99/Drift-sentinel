from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.common import dataset_name, info_panel, method_name, page_header
from dashboard.data_loader import DashboardData, comparison_view
from dashboard.styles import MODEL_COLORS, plotly_layout


METRICS = {
    "F1": "f1",
    "PR-AUC": "pr_auc",
    "Mean lead time": "mean_lead_time",
    "Warning coverage": "detection_coverage",
    "Batch FAR": "batch_far",
    "Missed-drift rate": "missed_drift_rate",
}


def _format_table(frame: pd.DataFrame) -> pd.DataFrame:
    cols = ["method", *METRICS.values()]
    output = frame[[column for column in cols if column in frame]].copy()
    output["method"] = output.method.map(method_name)
    output.columns = ["Method", *[label for label, column in METRICS.items() if column in output.columns[1:]]]
    for column in output.columns[1:]:
        output[column] = output[column].map(lambda value: "N/A" if pd.isna(value) else f"{float(value):.4f}")
    return output


def render(data: DashboardData) -> None:
    page_header(
        "Seven-method benchmark",
        "Model comparison",
        "Change aggregation explicitly. Macro, pooled, and per-dataset results answer different questions and are never mixed into one ± value.",
    )
    c1, c2 = st.columns([1.25, 1])
    with c1:
        view = st.selectbox("Aggregation", ["Macro (equal dataset weight)", "Pooled (event/row weighted)", "Per dataset"])
    dataset = None
    with c2:
        if view == "Per dataset":
            valid = sorted(data.tables["per_dataset_results"].query("evaluation_valid == True").dataset.unique())
            dataset = st.selectbox("Dataset", valid, format_func=dataset_name)
        else:
            st.selectbox("Dataset", ["Defined by aggregation"], disabled=True)
    frame = comparison_view(data, view, dataset)

    chosen_labels = st.multiselect("Metrics", list(METRICS), default=["F1", "Warning coverage", "Batch FAR"])
    if not chosen_labels:
        st.info("Select at least one metric to draw the comparison chart.")
    else:
        selected = [METRICS[label] for label in chosen_labels]
        chart = frame[["method", *selected]].melt("method", var_name="metric", value_name="value").dropna()
        reverse = {value: key for key, value in METRICS.items()}
        chart["Metric"] = chart.metric.map(reverse)
        chart["Method"] = chart.method.map(method_name)
        colors = {method_name(key): value for key, value in MODEL_COLORS.items()}
        fig = px.bar(
            chart, x="Method", y="value", color="Method", facet_row="Metric",
            color_discrete_map=colors, text_auto=".3f", custom_data=["Metric"],
        )
        fig.update_layout(**plotly_layout(height=max(420, 235 * len(chosen_labels))))
        fig.update_layout(showlegend=False)
        fig.update_yaxes(matches=None, title=None)
        fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
        fig.update_traces(hovertemplate="%{x}<br>%{customdata[0]}: %{y:.4f}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    st.dataframe(_format_table(frame), hide_index=True, width="stretch")
    if view.startswith("Macro"):
        info_panel("Macro means average seeds within each dataset, then weight the three event-labelled datasets equally. They do not carry a ± because between-dataset variation is not seed variation.")
    elif view.startswith("Pooled"):
        info_panel("Pooled values aggregate row/event counts. Supervised methods contain 18 event trials across seeds; deterministic detectors contain six. Nonlinear metrics may reorder methods.")
    else:
        info_panel("This table is the dataset-level mean. Open the seed-variation table below for supervised mean ± sample standard deviation.")

    with st.expander("Within-dataset random-seed variation"):
        seed_frame = data.tables["per_dataset_seed_summary"].copy()
        if dataset:
            seed_frame = seed_frame[seed_frame.dataset == dataset]
        seed_frame["Dataset"] = seed_frame.dataset.map(dataset_name)
        seed_frame["Method"] = seed_frame.method.map(method_name)
        for label, stem in (("F1", "f1"), ("PR-AUC", "pr_auc"), ("Coverage", "detection_coverage"), ("Batch FAR", "batch_far")):
            seed_frame[label] = seed_frame.apply(lambda row: f"{row[f'{stem}_mean']:.4f} ± {row[f'{stem}_std']:.4f}", axis=1)
        st.dataframe(seed_frame[["Dataset", "Method", "F1", "PR-AUC", "Coverage", "Batch FAR"]], hide_index=True, width="stretch")
        st.caption("Every ± is variability over seeds 11, 22, and 33 within one dataset—not variation across datasets.")

    info_panel("ADWIN, DDM, and KSWIN expose binary flags only. Their PR-AUC and ROC-AUC are N/A, not zero, and are excluded from area comparisons.", warning=True)
