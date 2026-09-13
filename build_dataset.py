"""
build_dataset.py
-----------------
Phase 1 main pipeline.

For each dataset:
  1. Stream instances one at a time through a simple online classifier
     (Hoeffding Tree) -> get prediction + confidence.
  2. Feed raw features + confidence into RollingStatTracker -> get one
     summary row per batch (mean/var shift, PSI, confidence trend).
  3. Feed correctness (0/1) into BaselineDetectorSuite -> ADWIN/DDM/KSWIN
     alarm log.
  4. Attach a ground-truth drift label to each batch:
       - for REAL datasets with known/labeled drift regions (e.g. Insects
         variant, or Elec2 known volatile periods): label batches within
         `label_lead_window` batches BEFORE a known drift point as 1.
       - for SYNTHETIC datasets: we control the drift injection point
         exactly, so ground truth is exact.
  5. Write one CSV per dataset:
       columns = [batch_id, <rolling stat columns...>, ground_truth_drift_label,
                  adwin_alarm, ddm_alarm, kswin_alarm]

Output goes to ../outputs/<dataset_name>.csv -- this is the handoff
artifact for Member 2.
"""

from pathlib import Path

import pandas as pd
import yaml
from river import tree
from rolling_stats import RollingStatTracker
from baseline_detectors import BaselineDetectorSuite


def build_dataset(
    stream,
    feature_names,
    dataset_name: str,
    reference_size: int = 300,
    batch_size: int = 50,
    known_drift_points: list = None,   # batch indices (post-reference) where drift TRULY starts
    label_lead_window: int = 5,        # how many batches before a drift point count as "positive"
    max_instances: int = None,
    out_dir: str = "outputs",
):
    model = tree.HoeffdingTreeClassifier()
    tracker = RollingStatTracker(feature_names, reference_size=reference_size, batch_size=batch_size)
    baselines = BaselineDetectorSuite()

    rows = []
    batch_idx = 0
    n_seen = 0

    for x, y in stream:
        # --- online predict-then-learn ---
        y_pred = model.predict_one(x)
        proba = model.predict_proba_one(x)
        confidence = max(proba.values()) if proba else 0.5
        correct = int(y_pred == y)
        model.learn_one(x, y)

        # --- rolling stats ---
        stat_row = tracker.add(x, confidence)

        # --- baseline detectors (only meaningful once reference window is locked) ---
        if tracker._reference_locked:
            baselines.update(1 - correct, batch_idx)

        if stat_row is not None:
            stat_row["batch_id"] = batch_idx
            rows.append(stat_row)
            batch_idx += 1

        n_seen += 1
        if max_instances and n_seen >= max_instances:
            break

    df = pd.DataFrame(rows)

    # --- ground truth labeling ---
    df["ground_truth_drift_label"] = 0
    if known_drift_points:
        for dp in known_drift_points:
            lo = max(0, dp - label_lead_window)
            df.loc[(df["batch_id"] >= lo) & (df["batch_id"] < dp), "ground_truth_drift_label"] = 1

    # --- attach baseline alarms as columns ---
    alarm_batches = baselines.alarm_batches()
    for name, batches in alarm_batches.items():
        col = f"{name.lower()}_alarm"
        df[col] = df["batch_id"].isin(batches).astype(int)

    out_path = f"{out_dir}/{dataset_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"[{dataset_name}] wrote {len(df)} batch rows -> {out_path}")
    print(f"[{dataset_name}] baseline alarm counts: { {k: len(v) for k, v in alarm_batches.items()} }")
    return df


# ---------------------------------------------------------------------
# Dataset-specific runners
# ---------------------------------------------------------------------

def run_elec2():
    from river import datasets
    stream = datasets.Elec2()
    feature_names = ["date", "day", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
    # Elec2 has no single labeled drift point (it's continuously volatile) ->
    # for now leave known_drift_points=None; Phase 3 will use lead-time
    # analysis relative to baseline alarms instead of a fixed ground truth.
    build_dataset(stream, feature_names, "elec2", known_drift_points=None)


def _cast_numeric(stream):
    """Insects/Elec2 raw features can arrive as strings; cast numeric ones to float."""
    for x, y in stream:
        x_cast = {}
        for k, v in x.items():
            try:
                x_cast[k] = float(v)
            except (TypeError, ValueError):
                x_cast[k] = v
        yield x_cast, y


def run_insects(variant="abrupt_balanced"):
    from river import datasets
    ds = datasets.Insects(variant=variant)
    feature_names = [f"f{i}" for i in range(1, 34)]  # Insects has 33 numeric attributes (named f1..f33)
    build_dataset(_cast_numeric(ds), feature_names, f"insects_{variant}", known_drift_points=None)


def run_synthetic_sea_drift():
    """
    Fully controlled synthetic benchmark: SEA generator with recurring
    ABRUPT concept changes at known instance counts -> exact ground truth.
    This is the dataset to use for validating lead-time / false-alarm
    metrics precisely (stand-in for the proposal's 'Forest Covertype with
    synthetic drift injection').
    """
    from driftsentinel.streams import make_multi_sea_stream

    config = yaml.safe_load((Path(__file__).parent / "configs/default.yaml").read_text())
    generation = config["synthetic_sea"]
    drift_batches = tuple((int(point) - int(config["phase1_reference_size"])) // int(config["phase1_batch_size"]) for point in generation["event_instances"])
    stream = make_multi_sea_stream("abrupt", generation)
    feature_names = [0, 1, 2]  # SEA yields integer-keyed features, not named ones

    build_dataset(
        stream, feature_names, "synth_sea_abrupt",
        reference_size=int(config["phase1_reference_size"]), batch_size=int(config["phase1_batch_size"]),
        known_drift_points=list(drift_batches),
        label_lead_window=int(config["lead_horizon"]),
        max_instances=int(generation["max_instances"]),
    )


def run_synthetic_sea_gradual():
    """Controlled SEA stream with recurring gradual changes across every split."""
    from driftsentinel.streams import make_multi_sea_stream

    config = yaml.safe_load((Path(__file__).parent / "configs/default.yaml").read_text())
    generation = config["synthetic_sea"]
    drift_batches = tuple((int(point) - int(config["phase1_reference_size"])) // int(config["phase1_batch_size"]) for point in generation["event_instances"])
    build_dataset(
        make_multi_sea_stream("gradual", generation), [0, 1, 2], "synth_sea_gradual",
        reference_size=int(config["phase1_reference_size"]), batch_size=int(config["phase1_batch_size"]),
        known_drift_points=list(drift_batches), label_lead_window=int(config["lead_horizon"]),
        max_instances=int(generation["max_instances"]),
    )


def run_from_csv(
    csv_path: str,
    target_col: str,
    dataset_name: str,
    feature_cols: list = None,
    reference_size: int = 300,
    batch_size: int = 100,
    known_drift_instance: int = None,   # raw instance index of a true/suspected drift, if known
    out_dir: str = "outputs",
):
    """
    Generic loader for ANY external CSV dataset -- e.g. the real Forest
    Covertype or Airlines datasets downloaded from Kaggle/UCI, if you want
    an exact name-match to the original proposal instead of the River
    substitutes (Elec2/Insects).

    Download the CSV yourself first (Kaggle requires a login, so this
    can't be automated here), then call e.g.:

        run_from_csv(
            csv_path="../data/covtype.csv",
            target_col="Cover_Type",
            dataset_name="covertype",
            reference_size=1000,
            batch_size=200,
        )

    Large files (Covertype is ~581K rows, Airlines ~540K rows) will take
    a few minutes to stream through -- that's expected and fine.
    """
    import pandas as pd
    from river import stream as river_stream

    raw_df = pd.read_csv(csv_path)
    if feature_cols is None:
        feature_cols = [c for c in raw_df.columns if c != target_col]

    data_stream = river_stream.iter_pandas(raw_df[feature_cols], raw_df[target_col])

    known_drift_points = None
    if known_drift_instance is not None:
        drift_batch = max(0, (known_drift_instance - reference_size) // batch_size)
        known_drift_points = [drift_batch]

    build_dataset(
        data_stream, feature_cols, dataset_name,
        reference_size=reference_size, batch_size=batch_size,
        known_drift_points=known_drift_points,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    print("Running Phase 1 dataset builds...\n")
    run_synthetic_sea_drift()   # fast, no download, exact ground truth -> already verified working
    run_synthetic_sea_gradual() # controlled second benchmark with recurring gradual changes
    run_insects()                # real, ~52K rows -- downloads automatically on first run
    run_elec2()                  # real, ~45K rows -- downloads automatically on first run

    # To use a real Kaggle dataset (e.g. actual Forest Covertype or Airlines),
    # download the CSV yourself and uncomment/edit this:
    # run_from_csv(
    #     csv_path="../data/covtype.csv",
    #     target_col="Cover_Type",
    #     dataset_name="covertype",
    #     reference_size=1000,
    #     batch_size=200,
    # )
