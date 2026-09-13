from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from driftsentinel.data import chronological_split, construct_future_targets, identify_feature_columns, load_phase1, prepare_sequences, sequence_partition


def sample_frame(size: int = 30) -> pd.DataFrame:
    return pd.DataFrame({
        "batch_id": np.arange(size),
        "signal": np.arange(size, dtype=float),
        "variable": np.sin(np.arange(size)),
        "batch_size": 50,
        "ground_truth_drift_label": 0,
        "adwin_alarm": 0,
        "ddm_alarm": 0,
        "kswin_alarm": 0,
    })


def test_phase1_csv_loader(tmp_path: Path):
    path = tmp_path / "data.csv"
    sample_frame().to_csv(path, index=False)
    loaded = load_phase1(path)
    assert len(loaded) == 30
    assert loaded.batch_id.is_monotonic_increasing


def test_phase1_parquet_loader(tmp_path: Path):
    path = tmp_path / "data.parquet"
    sample_frame().to_parquet(path)
    assert len(load_phase1(path)) == 30


def test_loader_rejects_duplicate_and_gap(tmp_path: Path):
    duplicate = sample_frame()
    duplicate.loc[2, "batch_id"] = 1
    path = tmp_path / "bad.csv"
    duplicate.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        load_phase1(path)
    gap = sample_frame()
    gap.loc[2:, "batch_id"] += 1
    gap.to_csv(path, index=False)
    with pytest.raises(ValueError, match="gaps"):
        load_phase1(path)


def test_future_target_is_strictly_pre_onset():
    result = construct_future_targets(sample_frame(20), [10], horizon=3)
    assert result.loc[result.future_drift_target == 1, "batch_id"].tolist() == [7, 8, 9]
    assert result.loc[result.batch_id == 10, "future_drift_target"].item() == 0
    assert result.loc[result.batch_id == 7, "distance_to_drift"].item() == 3


def test_chronological_split_is_ordered():
    split = chronological_split(100, 0.6, 0.2)
    assert (split.train_end, split.validation_end, split.size) == (60, 80, 100)


def test_features_exclude_labels_metadata_and_detector_flags():
    assert identify_feature_columns(sample_frame()) == ["signal", "variable", "batch_size"]


def test_sequence_shape_partitions_and_train_only_scaler():
    frame = construct_future_targets(sample_frame(30), [25], horizon=3)
    prepared = prepare_sequences(frame, sequence_length=5, train_ratio=0.5, validation_ratio=0.2)
    train_x, _, train_rows = sequence_partition(prepared, "train")
    _, _, val_rows = sequence_partition(prepared, "validation")
    _, _, test_rows = sequence_partition(prepared, "test")
    assert train_x.shape[1:] == (5, 2)  # constant batch_size removed
    assert train_rows.max() < val_rows.min() < test_rows.min()
    expected_mean = frame.iloc[:15][["signal", "variable"]].mean().to_numpy()
    np.testing.assert_allclose(prepared.scaler.mean_, expected_mean)
    assert prepared.scaler.mean_[0] != frame["signal"].mean()


@pytest.mark.parametrize("dataset", ["synth_sea_abrupt", "synth_sea_gradual"])
def test_regenerated_synthetic_splits_all_contain_positive_targets(dataset):
    root = Path(__file__).parents[1]
    config = yaml.safe_load((root / "configs/default.yaml").read_text())
    spec = config["datasets"][dataset]
    frame = construct_future_targets(load_phase1(root / spec["path"]), spec["drift_onsets"], config["lead_horizon"])
    prepared = prepare_sequences(frame, config["sequence_length"], spec["train_ratio"], spec["validation_ratio"])
    for split in ("train", "validation", "test"):
        _, labels, _ = sequence_partition(prepared, split)
        assert labels.sum() > 0


def test_future_rows_do_not_change_train_scaler_or_historical_sequences():
    original = construct_future_targets(sample_frame(30), [25], horizon=5)
    changed = original.copy()
    changed.loc[20:, "signal"] = 1_000_000
    first = prepare_sequences(original, 5, 0.5, 0.2)
    second = prepare_sequences(changed, 5, 0.5, 0.2)
    np.testing.assert_allclose(first.scaler.mean_, second.scaler.mean_)
    np.testing.assert_allclose(first.x[first.end_rows < 15], second.x[second.end_rows < 15])


def test_validation_sequence_uses_only_past_lookback_and_destination_target():
    frame = construct_future_targets(sample_frame(30), [18], horizon=5)
    prepared = prepare_sequences(frame, sequence_length=5, train_ratio=0.5, validation_ratio=0.2)
    validation_x, validation_y, validation_rows = sequence_partition(prepared, "validation")
    assert validation_rows[0] == 15
    assert validation_y[0] == frame.loc[15, "future_drift_target"]
    expected = prepared.scaler.transform(frame.loc[11:15, prepared.feature_columns])
    np.testing.assert_allclose(validation_x[0], expected, rtol=1e-6)
    assert 11 < prepared.split.train_end <= validation_rows[0]
