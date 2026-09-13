"""Validated Phase 1 loading, future targets, temporal splits, and sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

LABEL_COLUMNS = {"ground_truth_drift_label", "future_drift_target"}
BASELINE_COLUMNS = {"adwin_alarm", "ddm_alarm", "kswin_alarm", "adwin_flag", "ddm_flag", "kswin_flag"}
METADATA_COLUMNS = {"batch_id", "timestamp", "index", "actual_drift_onset", "distance_to_drift", "inside_drift", "dataset", "truth_quality"}


@dataclass(frozen=True)
class TemporalSplit:
    train_end: int
    validation_end: int
    size: int


@dataclass
class PreparedData:
    frame: pd.DataFrame
    feature_columns: list[str]
    scaler: StandardScaler
    split: TemporalSplit
    x: np.ndarray
    y: np.ndarray
    end_rows: np.ndarray


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def load_phase1(path: str | Path) -> pd.DataFrame:
    """Load CSV/Parquet and enforce a strictly ordered, finite batch contract."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    elif source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    else:
        raise ValueError(f"Unsupported Phase 1 file: {source.suffix}")
    if "batch_id" not in frame:
        raise ValueError("Phase 1 data must contain batch_id")
    if frame.empty:
        raise ValueError("Phase 1 data is empty")
    if frame["batch_id"].duplicated().any():
        raise ValueError("Duplicate batch_id values")
    if not pd.api.types.is_numeric_dtype(frame["batch_id"]):
        raise TypeError("batch_id must be numeric")
    ids = frame["batch_id"].to_numpy(dtype=int)
    if np.any(np.diff(ids) <= 0):
        raise ValueError("batch_id must be strictly increasing")
    if np.any(np.diff(ids) != 1):
        raise ValueError("Unexpected gaps in batch_id")
    numeric = frame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("NaN or infinite numeric values in Phase 1 data")
    return frame.reset_index(drop=True)


def identify_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Select numeric predictors while excluding labels, ordering, and detectors."""
    excluded = LABEL_COLUMNS | BASELINE_COLUMNS | METADATA_COLUMNS
    return [c for c in frame.columns if c.lower() not in excluded and pd.api.types.is_numeric_dtype(frame[c])]


def construct_future_targets(frame: pd.DataFrame, drift_onsets: Sequence[int], horizon: int, inside_duration: int = 1) -> pd.DataFrame:
    """Label batch t iff an onset occurs in (t, t+horizon], never at/after onset."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    out = frame.copy()
    batches = out["batch_id"].to_numpy(dtype=int)
    onsets = np.asarray(sorted(set(int(v) for v in drift_onsets)), dtype=int)
    target = np.zeros(len(out), dtype=np.int8)
    distance = np.full(len(out), np.nan)
    next_onset = np.full(len(out), np.nan)
    inside = np.zeros(len(out), dtype=np.int8)
    for i, current in enumerate(batches):
        future = onsets[onsets > current]
        if future.size:
            distance[i] = future[0] - current
            next_onset[i] = future[0]
            target[i] = int(distance[i] <= horizon)
        if onsets.size and np.any((current >= onsets) & (current < onsets + inside_duration)):
            inside[i] = 1
    out["future_drift_target"] = target
    out["actual_drift_onset"] = next_onset
    out["distance_to_drift"] = distance
    out["inside_drift"] = inside
    return out


def chronological_split(size: int, train_ratio: float, validation_ratio: float) -> TemporalSplit:
    if size < 3:
        raise ValueError("At least three observations are required")
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1 or train_ratio + validation_ratio >= 1:
        raise ValueError("Invalid temporal split ratios")
    train_end = max(1, int(size * train_ratio))
    validation_end = max(train_end + 1, int(size * (train_ratio + validation_ratio)))
    validation_end = min(validation_end, size - 1)
    return TemporalSplit(train_end, validation_end, size)


def prepare_sequences(frame: pd.DataFrame, sequence_length: int, train_ratio: float, validation_ratio: float) -> PreparedData:
    """Fit a train-only scaler and create sequences ending at each target row."""
    features = identify_feature_columns(frame)
    if not features:
        raise ValueError("No usable feature columns")
    split = chronological_split(len(frame), train_ratio, validation_ratio)
    train_values = frame.iloc[: split.train_end][features].to_numpy(dtype=np.float64)
    nonconstant = np.ptp(train_values, axis=0) > 0
    features = [name for name, keep in zip(features, nonconstant) if keep]
    if not features:
        raise ValueError("All candidate features are constant in training")
    scaler = StandardScaler().fit(frame.iloc[: split.train_end][features])
    scaled = scaler.transform(frame[features]).astype(np.float32)
    labels = frame["future_drift_target"].to_numpy(dtype=np.float32)
    x, y, end_rows = [], [], []
    for end in range(sequence_length - 1, len(frame)):
        x.append(scaled[end - sequence_length + 1 : end + 1])
        y.append(labels[end])
        end_rows.append(end)
    return PreparedData(frame, features, scaler, split, np.asarray(x), np.asarray(y), np.asarray(end_rows))


def sequence_partition(data: PreparedData, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounds = {"train": (0, data.split.train_end), "validation": (data.split.train_end, data.split.validation_end), "test": (data.split.validation_end, data.split.size)}
    if name not in bounds:
        raise KeyError(name)
    lo, hi = bounds[name]
    mask = (data.end_rows >= lo) & (data.end_rows < hi)
    return data.x[mask], data.y[mask], data.end_rows[mask]
