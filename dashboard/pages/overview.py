from __future__ import annotations

import streamlit as st

from dashboard.components.common import dataset_name, info_panel, metric_card, page_header, presentation_table
from dashboard.data_loader import DashboardData, format_value


def render(data: DashboardData) -> None:
    page_header(
        "Research system overview",
        "Predicting concept drift before it arrives",
        "A reproducible early-warning benchmark across recurrent, attention-based, linear, and reactive methods.",
    )

    macro = data.tables["macro_dataset_summary"].set_index("method")
    gru = macro.loc["GRU"]
    ddm = macro.loc["DDM"]
    primary = st.columns(3)
    with primary[0]:
        metric_card("Strict warning horizon", f"N = {data.horizon} batches", "Target and event eligibility are aligned.")
    with primary[1]:
        metric_card("Compared methods", "4 + 3", "Four predictive models; three reactive detectors.")
    with primary[2]:
        metric_card("GRU warning coverage", format_value(gru.detection_coverage, percent=True), "Equal-dataset macro mean across seeds.")
    secondary = st.columns(2)
    with secondary[0]:
        metric_card("GRU batch FAR", format_value(gru.batch_far, percent=True), "High operational cost must be read with coverage.")
    with secondary[1]:
        metric_card("DDM batch FAR", format_value(ddm.batch_far, percent=True), "Conservative, but with lower event coverage.")

    st.subheader("System architecture")
    st.markdown(
        """
        <div class="pipeline">
          <div class="pipe-node">Raw stream</div><div class="pipe-arrow">→</div>
          <div class="pipe-node">Predict-then-learn classifier</div><div class="pipe-arrow">→</div>
          <div class="pipe-node">Rolling shifts, PSI & confidence</div><div class="pipe-arrow">→</div>
          <div class="pipe-node">LR · GRU · LSTM · Transformer</div><div class="pipe-arrow">→</div>
          <div class="pipe-node">Alert episodes</div><div class="pipe-arrow">→</div>
          <div class="pipe-node">Event metrics & causal replay</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    info_panel(
        f"<b>Scientific contract:</b> target(t)=1 only when an onset lies in (t, t+{data.horizon}]. "
        f"A true warning must start in [onset−{data.horizon}, onset); onset−{data.horizon + 1} is a false warning."
    )
    info_panel(
        "<b>Predictive versus reactive:</b> Logistic Regression, GRU, LSTM, and Transformer estimate future-onset risk. "
        "ADWIN, DDM, and KSWIN react to observed error-stream changes and expose binary alarms only."
    )

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Benchmarked data")
        distribution = data.tables["class_event_distribution"].copy()
        distribution["dataset"] = distribution.dataset.map(dataset_name)
        distribution.columns = [
            "Dataset", "Split", "Rows", "Sequences", "Positive targets",
            "Negative targets", "Drift events",
        ]
        st.dataframe(distribution, hide_index=True, use_container_width=True, height=430)
    with right:
        st.subheader("What the results support")
        st.markdown(
            """
            - Temporal context adds measurable event usefulness: GRU is the strongest macro early-warning model.
            - That benefit is limited, not universal. Pooled F1 changes the model ordering.
            - High warning coverage comes with substantial false alarms and adaptation cost.
            - Binary detector areas are not comparable with probability-based neural PR-AUC and remain N/A.
            - The Transformer passed implementation checks but underperformed under sparse, imbalanced supervision.
            """
        )
        info_panel(
            "<b>Elec2 guardrail:</b> it is audited but not event-scored because no authoritative onset truth or positive targets are available.",
            warning=True,
        )

    with st.expander("Dataset provenance and benchmark scope"):
        provenance = data.config["datasets"]["insects_abrupt_balanced"]["provenance"]
        st.markdown(
            f"""
            The proposal named Electricity, Airlines, Forest CoverType, and USENET. The Phase 1 repository actually
            contained Elec2, INSECTS abrupt-balanced, and SEA; this final benchmark truthfully uses those available
            streams, with abrupt and gradual deterministic SEA variants.

            INSECTS verified mirror: **{provenance['rows']:,} rows**, **{provenance['features']} features**,
            **{provenance['classes']} classes**. SHA-256: `{provenance['sha256']}`.
            """
        )
