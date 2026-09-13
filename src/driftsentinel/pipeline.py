"""End-to-end Phase 2 training and Phase 3 evaluation orchestration."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PreparedData, construct_future_targets, load_phase1, prepare_sequences, sequence_partition
from .evaluation import classification_metrics, early_warning_metrics, threshold_sweep
from .figures import generate_figures
from .logistic import fit_logistic_baseline, load_logistic_checkpoint, logistic_probabilities, save_logistic_checkpoint
from .models import build_model
from .recovery import raw_stream, replay_alert_adaptation
from .training import load_checkpoint, predict_probabilities, save_checkpoint, train_model
from .utils import choose_device, save_json, seed_everything

LOGGER = logging.getLogger(__name__)
NEURAL_MODEL_NAMES = ("gru", "transformer", "lstm")
LOGISTIC_METHOD = "LOGISTIC REGRESSION"
PROBABILITY_METHODS = {name.upper() for name in NEURAL_MODEL_NAMES} | {LOGISTIC_METHOD}
BASELINES = {"ADWIN": "adwin_alarm", "DDM": "ddm_alarm", "KSWIN": "kswin_alarm"}


def _validation_threshold(labels: np.ndarray, probabilities: np.ndarray, fallback: float) -> float:
    """Choose an operating point on validation only; keep fallback for one-class validation."""
    if np.unique(labels).size < 2:
        return fallback
    candidates = np.linspace(0.05, 0.95, 19)
    scored = [(classification_metrics(labels, probabilities, float(value))["f1"], float(value)) for value in candidates]
    return max(scored, key=lambda item: (item[0], item[1]))[1]


def _mkdirs(root: Path) -> None:
    for path in (
        root / "artifacts/checkpoints", root / "artifacts/predictions", root / "artifacts/preprocessing",
        root / "results/metrics", root / "results/tables", root / "results/figures", root / "results/predictions",
    ):
        path.mkdir(parents=True, exist_ok=True)


def _clean_generated(root: Path) -> None:
    patterns = (
        "artifacts/checkpoints/*.pt", "artifacts/checkpoints/*.pkl", "artifacts/predictions/*.csv", "artifacts/preprocessing/*.json",
        "results/predictions/*.csv", "results/metrics/training_*.csv", "results/metrics/recovery_accuracy_*.csv",
        "results/figures/*.png", "results/figures/*.pdf",
    )
    for pattern in patterns:
        for path in root.glob(pattern):
            path.unlink()


def _dataset_ratios(config: dict, spec: dict) -> tuple[float, float]:
    return float(spec.get("train_ratio", config["train_ratio"])), float(spec.get("validation_ratio", config["validation_ratio"]))


def _split_distributions(dataset: str, prepared: PreparedData, onsets: list[int]) -> list[dict]:
    partitions = {
        "train": (0, prepared.split.train_end),
        "validation": (prepared.split.train_end, prepared.split.validation_end),
        "test": (prepared.split.validation_end, prepared.split.size),
    }
    rows = []
    for name, (lo, hi) in partitions.items():
        _, labels, end_rows = sequence_partition(prepared, name)
        first_batch = int(prepared.frame.iloc[lo]["batch_id"])
        last_batch = int(prepared.frame.iloc[hi - 1]["batch_id"])
        event_count = sum(first_batch < onset <= last_batch for onset in onsets)
        rows.append({
            "dataset": dataset, "split": name, "rows": hi - lo, "sequences": len(end_rows),
            "positive_targets": int(labels.sum()), "negative_targets": int(len(labels) - labels.sum()),
            "number_of_drift_events": event_count,
        })
    return rows


def _empty_metrics() -> dict[str, float]:
    return {key: float("nan") for key in ("precision", "recall", "f1", "roc_auc", "pr_auc", "accuracy", "tn", "fp", "fn", "tp")}


def _probability_diagnostics(probabilities: np.ndarray, prefix: str) -> dict[str, float]:
    if not len(probabilities):
        return {f"{prefix}_{key}": float("nan") for key in ("min", "p10", "median", "p90", "max", "mean")}
    quantiles = np.quantile(probabilities, [0, 0.1, 0.5, 0.9, 1])
    return {
        f"{prefix}_min": float(quantiles[0]), f"{prefix}_p10": float(quantiles[1]),
        f"{prefix}_median": float(quantiles[2]), f"{prefix}_p90": float(quantiles[3]),
        f"{prefix}_max": float(quantiles[4]), f"{prefix}_mean": float(np.mean(probabilities)),
    }


def _attach_recovery(warnings: pd.DataFrame, recovery_events: pd.DataFrame) -> pd.DataFrame:
    warnings = warnings.copy()
    warnings["accuracy_recovery_batches"] = np.nan
    warnings["recovery_coverage"] = np.nan
    if recovery_events.empty:
        return warnings
    grouped = recovery_events.groupby(["dataset", "method", "seed"], dropna=False).agg(
        accuracy_recovery_batches=("accuracy_recovery_batches", "mean"),
        recovery_coverage=("recovered", "mean"),
    ).reset_index()
    return warnings.drop(columns=["accuracy_recovery_batches", "recovery_coverage"]).merge(grouped, on=["dataset", "method", "seed"], how="left")


def _write_aggregate_tables(root: Path, performance: pd.DataFrame, warnings: pd.DataFrame, pooled_scores: list[dict]) -> None:
    merged = performance.merge(warnings, on=["dataset", "method", "seed", "truth_quality", "evaluation_valid"], how="left")
    valid = merged[merged["evaluation_valid"]].copy()
    numeric = [column for column in merged.select_dtypes(include=[np.number]).columns if column != "seed"]
    comparison = merged.groupby(["dataset", "method", "truth_quality"], dropna=False)[numeric].mean().reset_index()
    flags = merged.groupby(["dataset", "method", "truth_quality"], dropna=False)["evaluation_valid"].all().reset_index()
    comparison = comparison.merge(flags, on=["dataset", "method", "truth_quality"])
    comparison.to_csv(root / "results/tables/baseline_comparison.csv", index=False)
    comparison.to_csv(root / "results/tables/per_dataset_results.csv", index=False)

    supervised = valid[valid["seed"] >= 0]
    metrics = [
        "precision", "recall", "f1", "roc_auc", "pr_auc", "accuracy", "mean_lead_time",
        "detection_coverage", "batch_far", "event_far", "missed_drift_rate",
        "accuracy_recovery_batches", "recovery_coverage", "total_adaptations",
        "adaptations_per_100_batches", "false_adaptations", "true_event_related_adaptations",
        "adaptation_precision",
    ]
    summary = supervised.groupby(["dataset", "method"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(root / "results/tables/per_dataset_seed_summary.csv", index=False)

    valid_comparison = comparison[comparison["evaluation_valid"]]
    macro = valid_comparison.groupby("method")[metrics].mean(numeric_only=True).reset_index()
    macro.insert(1, "datasets", valid_comparison.groupby("method").size().reindex(macro["method"]).to_numpy())
    macro.to_csv(root / "results/tables/macro_dataset_summary.csv", index=False)

    adaptation_metrics = [
        "total_adaptations", "adaptations_per_100_batches", "false_adaptations",
        "true_event_related_adaptations", "adaptation_precision",
    ]
    adaptation = valid.groupby(["dataset", "method"])[adaptation_metrics].agg(["mean", "std"])
    adaptation.columns = [f"{metric}_{stat}" for metric, stat in adaptation.columns]
    adaptation.reset_index().to_csv(root / "results/tables/adaptation_summary.csv", index=False)

    score_frame = pd.DataFrame(pooled_scores)
    pooled_rows = []
    for method, group in valid.groupby("method"):
        tp, fp, fn, tn = (group[key].sum() for key in ("tp", "fp", "fn", "tn"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        matching_warning = warnings[(warnings["method"] == method) & warnings["evaluation_valid"]]
        events = int(matching_warning["drift_events"].sum())
        detected = int(matching_warning["detected_events"].sum())
        episodes = int(matching_warning["alert_episodes"].sum())
        scores = score_frame[score_frame["method"] == method] if not score_frame.empty else pd.DataFrame()
        pr_auc = float("nan")
        if method in PROBABILITY_METHODS and not scores.empty and scores["label"].sum():
            pr_auc = classification_metrics(scores["label"], scores["score"])["pr_auc"]
        total_adaptations = int(matching_warning["total_adaptations"].sum())
        false_adaptations = int(matching_warning["false_adaptations"].sum())
        evaluated_batches = int(matching_warning["evaluated_batches"].sum())
        pooled_rows.append({
            "method": method, "precision": precision, "recall": recall, "f1": f1, "pr_auc": pr_auc,
            "mean_lead_time": matching_warning["lead_time_sum"].sum() / detected if detected else 0.0,
            "detection_coverage": detected / events if events else float("nan"),
            "batch_far": matching_warning["false_batch_alerts"].sum() / matching_warning["eligible_batches"].sum() if matching_warning["eligible_batches"].sum() else 0.0,
            "event_far": matching_warning["false_alert_episodes"].sum() / episodes if episodes else 0.0,
            "missed_drift_rate": 1 - detected / events if events else float("nan"),
            "accuracy_recovery_batches": matching_warning["accuracy_recovery_batches"].mean(),
            "recovery_coverage": matching_warning["recovery_coverage"].mean(),
            "drift_events": events,
            "total_adaptations": total_adaptations,
            "adaptations_per_100_batches": 100.0 * total_adaptations / evaluated_batches if evaluated_batches else 0.0,
            "false_adaptations": false_adaptations,
            "true_event_related_adaptations": total_adaptations - false_adaptations,
            "adaptation_precision": (total_adaptations - false_adaptations) / total_adaptations if total_adaptations else 0.0,
        })
    pd.DataFrame(pooled_rows).to_csv(root / "results/tables/pooled_event_results.csv", index=False)


def run_pipeline(config: dict, root: str | Path = ".", quick: bool = False) -> dict[str, Path]:
    root = Path(root).resolve()
    _mkdirs(root)
    _clean_generated(root)
    device = choose_device(config.get("device", "auto"))
    seeds = [int(config["seeds"][0])] if quick else [int(v) for v in config["seeds"]]
    epochs = int(config["quick_epochs"] if quick else config["epochs"])
    horizon = int(config["lead_horizon"])
    performance_rows: list[dict] = []
    warning_rows: list[dict] = []
    tradeoff_rows: list[dict] = []
    distribution_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    recovery_rows: list[pd.DataFrame] = []
    pooled_scores: list[dict] = []
    LOGGER.info("Running on %s with %d seed(s), %d epoch cap; target/evaluation horizon=%d", device, len(seeds), epochs, horizon)

    for dataset_name, spec in config["datasets"].items():
        frame = construct_future_targets(load_phase1(root / spec["path"]), spec.get("drift_onsets", []), horizon)
        onsets = [int(v) for v in spec.get("drift_onsets", [])]
        train_ratio, validation_ratio = _dataset_ratios(config, spec)
        prepared = prepare_sequences(frame, int(config["sequence_length"]), train_ratio, validation_ratio)
        train_x, train_y, _ = sequence_partition(prepared, "train")
        validation_x, validation_y, _ = sequence_partition(prepared, "validation")
        test_x, test_y, test_rows = sequence_partition(prepared, "test")
        distribution_rows.extend(_split_distributions(dataset_name, prepared, onsets))
        test_frame = frame.iloc[test_rows].copy()
        test_batches = test_frame["batch_id"].to_numpy(dtype=int)
        test_onsets = [v for v in onsets if int(test_batches.min()) < v <= int(test_batches.max())]
        valid_supervised = bool(train_y.sum() > 0 and validation_y.sum() > 0 and test_y.sum() > 0 and test_onsets)
        save_json({
            "dataset": dataset_name, "source": spec["path"], "truth_quality": spec.get("truth_quality", "unspecified"),
            "feature_columns": prepared.feature_columns, "scaler_mean": prepared.scaler.mean_.tolist(), "scaler_scale": prepared.scaler.scale_.tolist(),
            "train_end_batch": int(frame.iloc[prepared.split.train_end - 1]["batch_id"]),
            "validation_end_batch": int(frame.iloc[prepared.split.validation_end - 1]["batch_id"]),
        }, root / f"artifacts/preprocessing/{dataset_name}.json")
        LOGGER.info("%s: train/val/test positives=%d/%d/%d; supervised evaluation=%s", dataset_name, train_y.sum(), validation_y.sum(), test_y.sum(), valid_supervised)
        alert_policies: dict[str, np.ndarray] = {}

        for model_name in NEURAL_MODEL_NAMES:
            for seed in seeds:
                method = model_name.upper()
                if not valid_supervised:
                    performance_rows.append({"dataset": dataset_name, "method": method, "seed": seed, "split": "test", "truth_quality": spec.get("truth_quality"), "evaluation_valid": False, "decision_threshold": float("nan"), **_empty_metrics()})
                    diagnostic_rows.append({"dataset": dataset_name, "model": method, "seed": seed, "status": "skipped_no_positive_split", "train_positive": int(train_y.sum()), "train_negative": int(len(train_y) - train_y.sum())})
                    continue
                seed_everything(seed)
                kwargs = {"hidden_dim": int(config["hidden_dim"]), "num_layers": int(config["num_layers"]), "dropout": float(config["dropout"])}
                trained = train_model(build_model(model_name, len(prepared.feature_columns), **kwargs), train_x, train_y, validation_x, validation_y, device, epochs, int(config["batch_size"]), float(config["learning_rate"]), float(config["weight_decay"]), int(config["patience"]))
                checkpoint = root / f"artifacts/checkpoints/{dataset_name}_{model_name}_seed{seed}.pt"
                metadata = {"dataset": dataset_name, "model": model_name, "seed": seed, "input_dim": len(prepared.feature_columns), "best_epoch": trained.best_epoch, "sequence_length": int(config["sequence_length"])}
                save_checkpoint(trained.model, checkpoint, metadata)
                restored, restored_metadata = load_checkpoint(checkpoint, build_model(model_name, len(prepared.feature_columns), **kwargs), device)
                if restored_metadata != metadata:
                    raise RuntimeError("Checkpoint metadata failed round-trip verification")
                validation_probabilities = predict_probabilities(restored, validation_x, device)
                decision_threshold = _validation_threshold(validation_y, validation_probabilities, float(config["threshold"]))
                probabilities = predict_probabilities(restored, test_x, device)
                predicted = probabilities >= decision_threshold
                metrics = classification_metrics(test_y, probabilities, decision_threshold)
                performance_rows.append({"dataset": dataset_name, "method": method, "seed": seed, "split": "test", "truth_quality": spec.get("truth_quality"), "evaluation_valid": True, "decision_threshold": decision_threshold, **metrics})
                warning = early_warning_metrics(test_batches, predicted, test_onsets, horizon, int(config["cooldown"]))
                warning_rows.append({"dataset": dataset_name, "method": method, "seed": seed, "truth_quality": spec.get("truth_quality"), "evaluation_valid": True, **warning})
                alert_policies[f"{method}|{seed}"] = test_batches[predicted]
                pooled_scores.extend({"method": method, "label": int(label), "score": float(score)} for label, score in zip(test_y, probabilities))
                pos_weight = (len(train_y) - train_y.sum()) / train_y.sum()
                diagnostic_rows.append({
                    "dataset": dataset_name, "model": method, "seed": seed, "status": "trained_and_restored",
                    "train_positive": int(train_y.sum()), "train_negative": int(len(train_y) - train_y.sum()), "pos_weight": float(pos_weight),
                    "decision_threshold": decision_threshold, "fraction_test_predicted_positive": float(predicted.mean()),
                    "best_epoch": trained.best_epoch, "best_validation_loss": min(row["validation_loss"] for row in trained.history),
                    "input_features": test_x.shape[2], "sequence_length": test_x.shape[1], "uses_padding_mask": False,
                    **_probability_diagnostics(validation_probabilities, "validation_probability"), **_probability_diagnostics(probabilities, "test_probability"),
                })
                prediction = pd.DataFrame({
                    "dataset": dataset_name, "model": method, "seed": seed, "batch_id": test_batches,
                    "probability": probabilities, "decision_threshold": decision_threshold, "predicted_label": predicted.astype(int),
                    "ground_truth_future_drift": test_y.astype(int), "actual_drift_onset": test_frame["actual_drift_onset"].to_numpy(),
                    "distance_to_drift": test_frame["distance_to_drift"].to_numpy(),
                })
                for column in BASELINES.values():
                    prediction[column] = test_frame[column].to_numpy(dtype=int) if column in test_frame else 0
                prediction.to_csv(root / f"results/predictions/{dataset_name}_{model_name}_seed{seed}.csv", index=False)
                prediction.to_csv(root / f"artifacts/predictions/{dataset_name}_{model_name}_seed{seed}.csv", index=False)
                pd.DataFrame(trained.history).to_csv(root / f"results/metrics/training_{dataset_name}_{model_name}_seed{seed}.csv", index=False)
                sweep = threshold_sweep(test_batches, probabilities, test_onsets, horizon, int(config["cooldown"]))
                sweep.insert(0, "seed", seed); sweep.insert(0, "model", method); sweep.insert(0, "dataset", dataset_name)
                tradeoff_rows.extend(sweep.to_dict("records"))

        for seed in seeds:
            if not valid_supervised:
                performance_rows.append({"dataset": dataset_name, "method": LOGISTIC_METHOD, "seed": seed, "split": "test", "truth_quality": spec.get("truth_quality"), "evaluation_valid": False, "decision_threshold": float("nan"), **_empty_metrics()})
                diagnostic_rows.append({"dataset": dataset_name, "model": LOGISTIC_METHOD, "seed": seed, "status": "skipped_no_positive_split", "train_positive": int(train_y.sum()), "train_negative": int(len(train_y) - train_y.sum())})
                continue
            seed_everything(seed)
            model = fit_logistic_baseline(
                train_x,
                train_y,
                seed,
                c=float(config["logistic_c"]),
                max_iter=int(config["logistic_max_iter"]),
            )
            checkpoint = root / f"artifacts/checkpoints/{dataset_name}_logistic_regression_seed{seed}.pkl"
            metadata = {
                "dataset": dataset_name,
                "model": "logistic_regression",
                "seed": seed,
                "input_dim": len(prepared.feature_columns),
                "sequence_length": 1,
                "class_weight": "balanced",
            }
            save_logistic_checkpoint(model, checkpoint, metadata)
            restored, restored_metadata = load_logistic_checkpoint(checkpoint)
            if restored_metadata != metadata:
                raise RuntimeError("Logistic checkpoint metadata failed round-trip verification")
            validation_probabilities = logistic_probabilities(restored, validation_x)
            decision_threshold = _validation_threshold(validation_y, validation_probabilities, float(config["threshold"]))
            probabilities = logistic_probabilities(restored, test_x)
            predicted = probabilities >= decision_threshold
            metrics = classification_metrics(test_y, probabilities, decision_threshold)
            performance_rows.append({"dataset": dataset_name, "method": LOGISTIC_METHOD, "seed": seed, "split": "test", "truth_quality": spec.get("truth_quality"), "evaluation_valid": True, "decision_threshold": decision_threshold, **metrics})
            warning = early_warning_metrics(test_batches, predicted, test_onsets, horizon, int(config["cooldown"]))
            warning_rows.append({"dataset": dataset_name, "method": LOGISTIC_METHOD, "seed": seed, "truth_quality": spec.get("truth_quality"), "evaluation_valid": True, **warning})
            alert_policies[f"{LOGISTIC_METHOD}|{seed}"] = test_batches[predicted]
            pooled_scores.extend({"method": LOGISTIC_METHOD, "label": int(label), "score": float(score)} for label, score in zip(test_y, probabilities))
            diagnostic_rows.append({
                "dataset": dataset_name, "model": LOGISTIC_METHOD, "seed": seed, "status": "trained_and_restored",
                "train_positive": int(train_y.sum()), "train_negative": int(len(train_y) - train_y.sum()),
                "class_weight": "balanced", "decision_threshold": decision_threshold,
                "fraction_test_predicted_positive": float(predicted.mean()), "input_features": test_x.shape[2],
                "sequence_length": 1, "uses_padding_mask": False,
                **_probability_diagnostics(validation_probabilities, "validation_probability"),
                **_probability_diagnostics(probabilities, "test_probability"),
            })
            prediction = pd.DataFrame({
                "dataset": dataset_name, "model": LOGISTIC_METHOD, "seed": seed, "batch_id": test_batches,
                "probability": probabilities, "decision_threshold": decision_threshold,
                "predicted_label": predicted.astype(int), "ground_truth_future_drift": test_y.astype(int),
                "actual_drift_onset": test_frame["actual_drift_onset"].to_numpy(),
                "distance_to_drift": test_frame["distance_to_drift"].to_numpy(),
            })
            for column in BASELINES.values():
                prediction[column] = test_frame[column].to_numpy(dtype=int) if column in test_frame else 0
            prediction.to_csv(root / f"results/predictions/{dataset_name}_logistic_regression_seed{seed}.csv", index=False)
            prediction.to_csv(root / f"artifacts/predictions/{dataset_name}_logistic_regression_seed{seed}.csv", index=False)
            sweep = threshold_sweep(test_batches, probabilities, test_onsets, horizon, int(config["cooldown"]))
            sweep.insert(0, "seed", seed); sweep.insert(0, "model", LOGISTIC_METHOD); sweep.insert(0, "dataset", dataset_name)
            tradeoff_rows.extend(sweep.to_dict("records"))

        for method, column in BASELINES.items():
            flags = test_frame[column].to_numpy(dtype=int) if column in test_frame else np.zeros(len(test_frame), dtype=int)
            metrics = classification_metrics(test_y, flags, continuous_scores=False) if valid_supervised else _empty_metrics()
            performance_rows.append({"dataset": dataset_name, "method": method, "seed": -1, "split": "test", "truth_quality": spec.get("truth_quality"), "evaluation_valid": valid_supervised, "decision_threshold": 0.5, **metrics})
            if valid_supervised:
                warning_rows.append({"dataset": dataset_name, "method": method, "seed": -1, "truth_quality": spec.get("truth_quality"), "evaluation_valid": True, **early_warning_metrics(test_batches, flags.astype(bool), test_onsets, horizon, int(config["cooldown"]))})
                alert_policies[f"{method}|-1"] = test_batches[flags.astype(bool)]

        if valid_supervised and dataset_name != "elec2":
            events, accuracy = replay_alert_adaptation(
                raw_stream(dataset_name, spec, root, config.get("synthetic_sea")), alert_policies, test_onsets, dataset_name, horizon,
                reference_size=int(config["phase1_reference_size"]), batch_size=int(config["phase1_batch_size"]),
                retraining_window=int(config["retraining_window_instances"]), pre_window=int(config["recovery_pre_window"]),
                recovery_fraction=float(config["recovery_fraction"]), cooldown=int(config["cooldown"]),
            )
            identities = events["method_key"].str.split("|", regex=False, expand=True)
            events["method"] = identities[0]
            events["seed"] = identities[1].astype(int)
            recovery_rows.append(events)
            accuracy.to_csv(root / f"results/metrics/recovery_accuracy_{dataset_name}.csv", index=False)

    distributions = pd.DataFrame(distribution_rows)
    performance = pd.DataFrame(performance_rows)
    warnings = _attach_recovery(pd.DataFrame(warning_rows), pd.concat(recovery_rows, ignore_index=True) if recovery_rows else pd.DataFrame())
    tradeoff = pd.DataFrame(tradeoff_rows)
    recovery_events = pd.concat(recovery_rows, ignore_index=True) if recovery_rows else pd.DataFrame()
    distributions.to_csv(root / "results/tables/class_event_distribution.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(root / "results/metrics/model_diagnostics.csv", index=False)
    performance.to_csv(root / "results/metrics/model_performance.csv", index=False)
    warnings.to_csv(root / "results/metrics/early_warning_metrics.csv", index=False)
    tradeoff.to_csv(root / "results/metrics/threshold_tradeoff.csv", index=False)
    recovery_events.to_csv(root / "results/metrics/recovery_events.csv", index=False)
    _write_aggregate_tables(root, performance, warnings, pooled_scores)
    generate_figures(root, config, seeds[0])
    return {
        "model_performance": root / "results/metrics/model_performance.csv",
        "early_warning_metrics": root / "results/metrics/early_warning_metrics.csv",
        "class_distribution": root / "results/tables/class_event_distribution.csv",
        "baseline_comparison": root / "results/tables/baseline_comparison.csv",
        "threshold_tradeoff": root / "results/metrics/threshold_tradeoff.csv",
        "adaptation_summary": root / "results/tables/adaptation_summary.csv",
    }
