from __future__ import annotations

import streamlit as st

from dashboard.components.common import info_panel, metric_card, page_header
from dashboard.data_loader import DashboardData, format_value


def render(data: DashboardData) -> None:
    page_header(
        "Evidence, not a victory lap",
        "Research findings",
        "A faculty-ready summary of what the corrected experiment supports, what it does not, and where the system should go next.",
    )
    macro = data.tables["macro_dataset_summary"].set_index("method")
    pooled = data.tables["pooled_event_results"].set_index("method")
    gru, logistic = macro.loc["GRU"], macro.loc["LOGISTIC REGRESSION"]
    cols = st.columns(4)
    with cols[0]:
        metric_card("Macro GRU F1", format_value(gru.f1, 4), "vs Logistic Regression " + format_value(logistic.f1, 4))
    with cols[1]:
        metric_card("Macro GRU coverage", format_value(gru.detection_coverage, percent=True), "Strict five-batch event window")
    with cols[2]:
        metric_card("Pooled GRU F1", format_value(pooled.loc["GRU"].f1, 4), "Pooled Logistic Regression " + format_value(pooled.loc["LOGISTIC REGRESSION"].f1, 4))
    with cols[3]:
        metric_card("GRU adaptation precision", format_value(pooled.loc["GRU"].adaptation_precision, percent=True), "Pooled event-level replay")

    tab1, tab2, tab3, tab4 = st.tabs(["Core result", "Transformer audit", "Limitations", "Next steps"])
    with tab1:
        st.subheader("Temporal context helps—but at a price")
        st.markdown(
            """
            On the equal-dataset macro view, GRU improves F1, probability-based PR-AUC, and strict pre-onset coverage over the current-vector Logistic Regression baseline. Its operational distinction is event usefulness: GRU matches valid warning events while Logistic Regression's episode starts fall outside the eligible windows.

            The conclusion is deliberately bounded. Pooled F1 changes the ordering, so current rolling statistics already contain useful classification signal and aggregation choice matters. GRU's coverage also arrives with high false-alarm and reset cost. Classical detectors are substantially more conservative but cover fewer events.
            """
        )
        info_panel("Defensible claim: temporal context provides a measurable early-warning signal under this protocol. It does not establish universal superiority or deployment readiness.")
    with tab2:
        st.subheader("No implementation bug was found")
        st.markdown(
            """
            The audit verified class weighting, checkpoint restoration, `[batch, 10, features]` inputs, fixed-length masking assumptions, sinusoidal positional dimensions, and logits-versus-sigmoid handling. The Transformer genuinely underperformed in a sparse setting: SEA training has 30 positives, while INSECTS has only 10.

            The result was not tuned after test inspection. Its low score remains part of the final evidence.
            """
        )
    with tab3:
        st.markdown(
            """
            - Only three event-labelled datasets are evaluable, and only INSECTS is real.
            - SEA episodes are deterministic and reproducible, but correlated within one generator family.
            - INSECTS onsets are published instance positions mapped to batches.
            - Elec2 has no authoritative onset truth and cannot be event-scored without fabrication.
            - Three seeds characterize optimizer variation; they do not establish statistical significance.
            - Threshold sensitivity, false alarms, and adaptation cost remain practical constraints.
            - Replay is causal, but not a deployed system with delayed labels or approval constraints.
            """
        )
    with tab4:
        st.markdown(
            """
            1. Add independent real streams with authoritative event onset metadata.
            2. Improve calibration and alert suppression without changing the strict eligibility rule.
            3. Evaluate delayed labels and adaptation-approval budgets.
            4. Compare more compact temporal encoders before increasing model capacity.
            5. Report uncertainty over independent streams, not only optimizer seeds.
            """
        )

    st.subheader("Three-phase research protocol")
    phases = st.columns(3)
    with phases[0]:
        info_panel("<b>Phase 1 · Stream evidence</b><br>Replay each stream predict-then-learn and record rolling distribution, confidence, and detector features using past data only.")
    with phases[1]:
        info_panel("<b>Phase 2 · Early warning</b><br>Construct five-batch future-onset targets, split chronologically, scale on training data, and select thresholds on validation data.")
    with phases[2]:
        info_panel("<b>Phase 3 · Adaptation</b><br>Deduplicate alert episodes, reset the online classifier, warm-start on already-labelled history, and measure recovery plus reset cost.")

    st.subheader("Methodology safeguards")
    safeguards = st.columns(4)
    safeguards[0].success("Chronological splits")
    safeguards[1].success("Train-only scaling")
    safeguards[2].success("Validation-only thresholds")
    safeguards[3].success("Past-only adaptation")
    info_panel("All values on this page are read from committed result tables. The dashboard performs no training, downloading, threshold reselection, or metric recomputation at startup.")
