from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dashboard.data_loader import (
    BASELINE_METHODS,
    DashboardDataError,
    ROOT,
    _read_csv,
    comparison_view,
    load_dashboard_data,
    selected_threshold,
    threshold_options,
    threshold_row,
    valid_timeline_datasets,
    validate_baseline_auc,
    warning_annotations,
    METHOD_LABELS,
)


@pytest.fixture(scope="module")
def dashboard_data():
    return load_dashboard_data(ROOT)


def test_all_committed_dashboard_csvs_load_and_match_schema(dashboard_data):
    assert dashboard_data.tables
    assert dashboard_data.metrics
    assert dashboard_data.predictions
    assert all(not frame.empty for frame in dashboard_data.tables.values())


def test_loader_reports_missing_file_and_columns(tmp_path: Path):
    with pytest.raises(DashboardDataError, match="missing"):
        _read_csv(tmp_path / "missing.csv", {"required"})
    path = tmp_path / "bad.csv"
    pd.DataFrame({"wrong": [1]}).to_csv(path, index=False)
    with pytest.raises(DashboardDataError, match="required"):
        _read_csv(path, {"required"})


def test_macro_and_pooled_views_remain_distinct(dashboard_data):
    macro = comparison_view(dashboard_data, "Macro (equal dataset weight)").set_index("method")
    pooled = comparison_view(dashboard_data, "Pooled (event/row weighted)").set_index("method")
    assert macro.loc["GRU", "f1"] != pooled.loc["GRU", "f1"]
    assert macro.loc["LOGISTIC REGRESSION", "f1"] != pooled.loc["LOGISTIC REGRESSION", "f1"]


def test_per_dataset_view_excludes_invalid_elec2(dashboard_data):
    elec = comparison_view(dashboard_data, "Per dataset", "elec2")
    assert elec.empty
    assert "elec2" not in valid_timeline_datasets(dashboard_data)


def test_threshold_selection_uses_only_available_sweep_points(dashboard_data):
    options = threshold_options(dashboard_data, "synth_sea_abrupt", "GRU", 11)
    assert 0.05 in options and 0.95 in options
    selected = selected_threshold(dashboard_data, "synth_sea_abrupt", "GRU", 11)
    assert selected in options
    row = threshold_row(dashboard_data, "synth_sea_abrupt", "GRU", 11, selected)
    assert row.threshold == pytest.approx(selected)
    with pytest.raises(DashboardDataError):
        threshold_row(dashboard_data, "synth_sea_abrupt", "GRU", 11, 0.333)


def test_binary_detector_auc_is_unavailable(dashboard_data):
    frame = dashboard_data.tables["per_dataset_results"]
    baseline = frame[frame.method.isin(BASELINE_METHODS)]
    assert baseline.pr_auc.isna().all()
    assert baseline.roc_auc.isna().all()
    validate_baseline_auc(frame)
    corrupted = frame.copy()
    corrupted.loc[corrupted.method == "ADWIN", "pr_auc"] = 0.5
    with pytest.raises(DashboardDataError, match="must be N/A"):
        validate_baseline_auc(corrupted)


def test_consistent_model_labels_cover_all_committed_methods(dashboard_data):
    methods = set(dashboard_data.tables["macro_dataset_summary"].method)
    assert methods == set(METHOD_LABELS)
    assert METHOD_LABELS["LOGISTIC REGRESSION"] == "Logistic Regression"
    assert METHOD_LABELS["TRANSFORMER"] == "Transformer"


def test_warning_annotation_boundaries_and_one_to_one_matching():
    batches = list(range(0, 21))
    alerts = [batch in {4, 5, 6, 15} for batch in batches]
    # With cooldown=1, 4/5/6 are one episode starting at 4. Onset-6 is false;
    # it cannot borrow the later batch 5 from the same episode. Batch 15 is onset-5.
    result = warning_annotations(batches, alerts, [10, 20], horizon=5, cooldown=1)
    assert result["successful"] == [15]
    assert result["false"] == [4]
    assert result["missed"] == [10]


def test_warning_episode_cannot_match_two_events():
    batches = list(range(20))
    alerts = [batch == 8 for batch in batches]
    result = warning_annotations(batches, alerts, [10, 12], horizon=5, cooldown=1)
    assert result["successful"] == [8]
    assert len(result["missed"]) == 1


def test_streamlit_overview_smoke():
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(ROOT / "dashboard/app.py"))
    app.run(timeout=45)
    assert not app.exception
    assert app.title[0].value == "Predicting concept drift before it arrives"
