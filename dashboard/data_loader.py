"""Validated, cached access to committed dashboard artifacts.

This module deliberately reads only tracked experiment outputs and configuration.
It never trains a model, downloads data, or recomputes a scientific result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

try:
    import streamlit as st

    _cache_data = st.cache_data(show_spinner=False)
except ImportError:  # pragma: no cover - gives a clear loader error without Streamlit
    def _cache_data(func):
        return func


ROOT = Path(__file__).resolve().parents[1]

DATASET_LABELS = {
    "synth_sea_abrupt": "SEA · abrupt",
    "synth_sea_gradual": "SEA · gradual",
    "insects_abrupt_balanced": "INSECTS · abrupt-balanced",
    "elec2": "Elec2 · onset truth unavailable",
}
METHOD_LABELS = {
    "LOGISTIC REGRESSION": "Logistic Regression",
    "GRU": "GRU",
    "LSTM": "LSTM",
    "TRANSFORMER": "Transformer",
    "ADWIN": "ADWIN",
    "DDM": "DDM",
    "KSWIN": "KSWIN",
}
SUPERVISED_METHODS = ("LOGISTIC REGRESSION", "GRU", "LSTM", "TRANSFORMER")
BASELINE_METHODS = ("ADWIN", "DDM", "KSWIN")

TABLE_SCHEMAS = {
    "class_event_distribution": {
        "dataset", "split", "rows", "sequences", "positive_targets",
        "negative_targets", "number_of_drift_events",
    },
    "per_dataset_results": {
        "dataset", "method", "evaluation_valid", "f1", "pr_auc",
        "mean_lead_time", "detection_coverage", "batch_far",
        "missed_drift_rate", "accuracy_recovery_batches", "recovery_coverage",
        "adaptations_per_100_batches", "adaptation_precision",
    },
    "per_dataset_seed_summary": {
        "dataset", "method", "f1_mean", "f1_std", "pr_auc_mean", "pr_auc_std",
        "detection_coverage_mean", "detection_coverage_std", "batch_far_mean",
        "batch_far_std", "adaptations_per_100_batches_mean",
        "adaptations_per_100_batches_std",
    },
    "macro_dataset_summary": {
        "method", "datasets", "f1", "pr_auc", "mean_lead_time",
        "detection_coverage", "batch_far", "missed_drift_rate",
        "accuracy_recovery_batches", "recovery_coverage",
        "adaptations_per_100_batches", "adaptation_precision",
    },
    "pooled_event_results": {
        "method", "f1", "pr_auc", "mean_lead_time", "detection_coverage",
        "batch_far", "missed_drift_rate", "accuracy_recovery_batches",
        "recovery_coverage", "adaptations_per_100_batches", "adaptation_precision",
    },
    "adaptation_summary": {
        "dataset", "method", "adaptations_per_100_batches_mean",
        "adaptation_precision_mean",
    },
}

METRIC_SCHEMAS = {
    "model_performance": {
        "dataset", "method", "seed", "split", "evaluation_valid",
        "decision_threshold", "f1", "pr_auc",
    },
    "threshold_tradeoff": {
        "dataset", "model", "seed", "threshold", "mean_lead_time",
        "detection_coverage", "missed_drift_rate", "batch_far",
        "adaptations_per_100_batches", "adaptation_precision",
    },
    "early_warning_metrics": {
        "dataset", "method", "seed", "evaluation_valid", "mean_lead_time",
        "detection_coverage", "missed_drift_rate", "batch_far",
    },
    "recovery_events": {
        "dataset", "method", "seed", "drift_onset", "trigger_batch",
        "accuracy_recovery_batches", "recovered",
    },
}

PREDICTION_SCHEMA = {
    "dataset", "model", "seed", "batch_id", "probability",
    "decision_threshold", "predicted_label", "ground_truth_future_drift",
    "actual_drift_onset", "distance_to_drift", "adwin_alarm", "ddm_alarm",
    "kswin_alarm",
}


class DashboardDataError(RuntimeError):
    """Raised when a committed result required by the dashboard is invalid."""


@dataclass(frozen=True)
class DashboardData:
    root: Path
    config: dict
    tables: dict[str, pd.DataFrame]
    metrics: dict[str, pd.DataFrame]
    predictions: dict[tuple[str, str, int], pd.DataFrame]
    recovery_accuracy: dict[str, pd.DataFrame]

    @property
    def horizon(self) -> int:
        return int(self.config["lead_horizon"])

    @property
    def cooldown(self) -> int:
        return int(self.config["cooldown"])

    def onsets(self, dataset: str) -> list[int]:
        return [int(value) for value in self.config["datasets"][dataset]["drift_onsets"]]


def _read_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise DashboardDataError(f"Required dashboard artifact is missing: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas supplies useful details
        raise DashboardDataError(f"Could not read {path}: {exc}") from exc
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise DashboardDataError(f"{path} is missing required columns: {', '.join(missing)}")
    return frame


@_cache_data
def load_dashboard_data(root: str | Path = ROOT) -> DashboardData:
    root = Path(root).resolve()
    config_path = root / "configs/default.yaml"
    if not config_path.is_file():
        raise DashboardDataError(f"Required dashboard configuration is missing: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for key in ("lead_horizon", "cooldown", "datasets"):
        if key not in config:
            raise DashboardDataError(f"{config_path} is missing required key: {key}")

    tables = {
        name: _read_csv(root / f"results/tables/{name}.csv", schema)
        for name, schema in TABLE_SCHEMAS.items()
    }
    metrics = {
        name: _read_csv(root / f"results/metrics/{name}.csv", schema)
        for name, schema in METRIC_SCHEMAS.items()
    }

    predictions: dict[tuple[str, str, int], pd.DataFrame] = {}
    prediction_paths = sorted((root / "results/predictions").glob("*.csv"))
    if not prediction_paths:
        raise DashboardDataError("No committed prediction files were found.")
    for path in prediction_paths:
        frame = _read_csv(path, PREDICTION_SCHEMA)
        if frame.empty:
            raise DashboardDataError(f"Prediction artifact is empty: {path}")
        key = (str(frame.dataset.iloc[0]), str(frame.model.iloc[0]).upper(), int(frame.seed.iloc[0]))
        predictions[key] = frame

    recovery_accuracy = {}
    for path in sorted((root / "results/metrics").glob("recovery_accuracy_*.csv")):
        dataset = path.stem.removeprefix("recovery_accuracy_")
        recovery_accuracy[dataset] = _read_csv(path, {"batch_id"})

    validate_baseline_auc(tables["per_dataset_results"])
    return DashboardData(root, config, tables, metrics, predictions, recovery_accuracy)


def validate_baseline_auc(frame: pd.DataFrame) -> None:
    """Binary-only detector areas must remain unavailable, never zero-filled."""
    baseline = frame[frame["method"].isin(BASELINE_METHODS)]
    for column in ("pr_auc", "roc_auc"):
        if column in baseline and baseline[column].notna().any():
            raise DashboardDataError(
                f"Binary-only detector {column} must be N/A without a continuous score."
            )


def comparison_view(data: DashboardData, view: str, dataset: str | None = None) -> pd.DataFrame:
    """Return one explicitly named aggregation; never blend aggregation levels."""
    if view == "Per dataset":
        if dataset is None:
            raise ValueError("dataset is required for the per-dataset view")
        frame = data.tables["per_dataset_results"]
        return frame[(frame.dataset == dataset) & frame.evaluation_valid.fillna(False)].copy()
    if view == "Macro (equal dataset weight)":
        return data.tables["macro_dataset_summary"].copy()
    if view == "Pooled (event/row weighted)":
        return data.tables["pooled_event_results"].copy()
    raise ValueError(f"Unknown comparison view: {view}")


def threshold_options(data: DashboardData, dataset: str, model: str, seed: int) -> np.ndarray:
    frame = data.metrics["threshold_tradeoff"]
    values = frame[(frame.dataset == dataset) & (frame.model == model) & (frame.seed == seed)].threshold
    return np.sort(values.astype(float).unique())


def threshold_row(
    data: DashboardData, dataset: str, model: str, seed: int, threshold: float
) -> pd.Series:
    frame = data.metrics["threshold_tradeoff"]
    rows = frame[
        (frame.dataset == dataset)
        & (frame.model == model)
        & (frame.seed == seed)
        & np.isclose(frame.threshold.astype(float), float(threshold))
    ]
    if len(rows) != 1:
        raise DashboardDataError(
            f"Expected one threshold row for {dataset}/{model}/seed {seed}/{threshold}; got {len(rows)}"
        )
    return rows.iloc[0]


def selected_threshold(data: DashboardData, dataset: str, model: str, seed: int) -> float:
    frame = data.metrics["model_performance"]
    rows = frame[
        (frame.dataset == dataset)
        & (frame.method == model)
        & (frame.seed == seed)
        & (frame.split == "test")
        & frame.evaluation_valid.fillna(False)
    ]
    if len(rows) != 1:
        raise DashboardDataError(
            f"Expected one selected threshold for {dataset}/{model}/seed {seed}; got {len(rows)}"
        )
    return float(rows.iloc[0].decision_threshold)


def get_prediction(data: DashboardData, dataset: str, method: str, seed: int) -> pd.DataFrame:
    """Get model probabilities or detector flags from a canonical prediction artifact."""
    if method in BASELINE_METHODS:
        candidates = sorted(key for key in data.predictions if key[0] == dataset)
        if not candidates:
            raise DashboardDataError(f"No predictions available for {dataset}")
        frame = data.predictions[candidates[0]].copy()
        alarm = f"{method.lower()}_alarm"
        frame["display_value"] = frame[alarm].astype(float)
        frame["alert"] = frame[alarm].astype(bool)
        return frame
    key = (dataset, method, int(seed))
    if key not in data.predictions:
        raise DashboardDataError(f"No prediction artifact for {dataset}/{method}/seed {seed}")
    frame = data.predictions[key].copy()
    frame["display_value"] = frame.probability.astype(float)
    frame["alert"] = frame.predicted_label.astype(bool)
    return frame


def warning_annotations(
    batch_ids: Iterable[int], alerts: Iterable[bool], onsets: Iterable[int],
    horizon: int, cooldown: int,
) -> dict[str, list[int]]:
    """Classify deduplicated episode starts with strict one-to-one event matching."""
    from driftsentinel.evaluation import alert_episode_starts

    batches = np.asarray(list(batch_ids), dtype=int)
    raw_alerts = np.asarray(list(alerts), dtype=bool)
    episodes = alert_episode_starts(batches[raw_alerts], cooldown=cooldown)
    ordered_onsets = sorted(set(int(value) for value in onsets))
    matched: list[int] = []
    matched_set: set[int] = set()
    detected_onsets: list[int] = []
    for onset in ordered_onsets:
        candidates = [
            int(ep) for ep in episodes
            if onset - horizon <= int(ep) < onset and int(ep) not in matched_set
        ]
        if candidates:
            episode = candidates[0]
            matched.append(episode)
            matched_set.add(episode)
            detected_onsets.append(onset)
    false = [int(ep) for ep in episodes if int(ep) not in matched_set]
    missed = [onset for onset in ordered_onsets if onset not in detected_onsets]
    return {"successful": matched, "false": false, "missed": missed, "episodes": episodes.tolist()}


def valid_timeline_datasets(data: DashboardData) -> list[str]:
    datasets = sorted({key[0] for key in data.predictions})
    return [name for name in datasets if data.onsets(name)]


def format_value(value: object, digits: int = 3, percent: bool = False) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    number = float(value)
    if percent:
        return f"{number * 100:.1f}%"
    return f"{number:.{digits}f}"
