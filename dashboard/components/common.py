"""Small, accessible presentation components."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from dashboard.data_loader import DATASET_LABELS, METHOD_LABELS, format_value


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="eyebrow">{html.escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="subtitle">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        '<div class="metric-card">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-note">{html.escape(note)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def info_panel(text: str, warning: bool = False) -> None:
    cls = "info-panel warning-panel" if warning else "info-panel"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def method_name(value: str) -> str:
    return METHOD_LABELS.get(str(value), str(value).title())


def dataset_name(value: str) -> str:
    return DATASET_LABELS.get(str(value), str(value))


def presentation_table(frame: pd.DataFrame, percent_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    output = frame.copy()
    if "dataset" in output:
        output["dataset"] = output["dataset"].map(dataset_name)
    if "method" in output:
        output["method"] = output["method"].map(method_name)
    for column in output.columns:
        if pd.api.types.is_numeric_dtype(output[column]):
            output[column] = output[column].map(
                lambda value: format_value(value, percent=column in percent_columns)
            )
    return output
