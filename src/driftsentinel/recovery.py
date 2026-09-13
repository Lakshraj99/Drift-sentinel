"""Raw-stream replay for alert-triggered online classifier adaptation."""

from __future__ import annotations

import csv
import hashlib
import urllib.request
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from river import tree

from .evaluation import alert_episode_starts
from .streams import make_multi_sea_stream


def validate_insects_file(path: Path, provenance: Mapping[str, object]) -> None:
    """Fail closed when the mirrored INSECTS file differs from recorded metadata."""
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha != provenance["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual_sha}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "Class" not in reader.fieldnames:
            raise ValueError("INSECTS CSV must contain a Class column")
        feature_count = len(reader.fieldnames) - 1
        rows = 0
        classes: set[str] = set()
        for row in reader:
            rows += 1
            classes.add(row["Class"])
    expected = (int(provenance["rows"]), int(provenance["features"]), int(provenance["classes"]))
    actual = (rows, feature_count, len(classes))
    if actual != expected:
        raise ValueError(f"INSECTS metadata mismatch for {path}: expected {expected}, got {actual}")


def _verified_download(url: str, path: Path, provenance: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != provenance["sha256"]:
        urllib.request.urlretrieve(url, path)
    validate_insects_file(path, provenance)
    return path


def raw_stream(dataset: str, spec: dict, root: Path, sea_generation: Mapping[str, object] | None = None) -> Iterator[tuple[dict, int]]:
    if dataset == "synth_sea_abrupt":
        return make_multi_sea_stream("abrupt", sea_generation)
    if dataset == "synth_sea_gradual":
        return make_multi_sea_stream("gradual", sea_generation)
    if dataset == "insects_abrupt_balanced":
        provenance = spec["provenance"]
        path = _verified_download(provenance["source_url"], root / "data/cache/INSECTS-abrupt_balanced_norm.csv", provenance)

        def iterate() -> Iterator[tuple[dict, int]]:
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    label = int(row.pop("Class"))
                    yield {key: float(value) for key, value in row.items()}, label

        return iterate()
    raise ValueError(f"No verified recovery stream for {dataset}")


def replay_alert_adaptation(
    stream: Iterable[tuple[dict, int]],
    alert_batches: Mapping[str, Iterable[int]],
    drift_onsets: Iterable[int],
    dataset: str,
    horizon: int,
    reference_size: int = 300,
    batch_size: int = 50,
    retraining_window: int = 300,
    pre_window: int = 5,
    recovery_fraction: float = 0.95,
    cooldown: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reset and warm-start one online model per alert policy using past data only."""
    episode_alerts = {method: set(alert_episode_starts(values, cooldown=cooldown).tolist()) for method, values in alert_batches.items()}
    models = {method: tree.HoeffdingTreeClassifier() for method in episode_alerts}
    recent: deque[tuple[dict, int]] = deque(maxlen=retraining_window)
    batch_correct = {method: [] for method in models}
    accuracy = {method: [] for method in models}
    batch_ids: list[int] = []
    batch_id = 0
    for index, (features, label) in enumerate(stream):
        recent.append((features, label))
        for method, model in models.items():
            prediction = model.predict_one(features)
            if index >= reference_size:
                batch_correct[method].append(int(prediction == label))
            model.learn_one(features, label)
        if index >= reference_size and (index - reference_size + 1) % batch_size == 0:
            batch_ids.append(batch_id)
            for method in models:
                accuracy[method].append(float(np.mean(batch_correct[method])))
                batch_correct[method].clear()
                if batch_id in episode_alerts[method]:
                    replacement = tree.HoeffdingTreeClassifier()
                    for old_features, old_label in recent:
                        replacement.learn_one(old_features, old_label)
                    models[method] = replacement
            batch_id += 1
    accuracy_frame = pd.DataFrame({"batch_id": batch_ids, **{method: values for method, values in accuracy.items()}})
    event_rows = []
    onsets = sorted(set(int(v) for v in drift_onsets))
    batches = accuracy_frame["batch_id"].to_numpy(dtype=int)
    for method in models:
        values = accuracy_frame[method].to_numpy(dtype=float)
        rolling = pd.Series(values).rolling(pre_window, min_periods=pre_window).mean().to_numpy()
        alerts = np.asarray(sorted(episode_alerts[method]), dtype=int)
        for event_index, onset in enumerate(onsets):
            next_onset = onsets[event_index + 1] if event_index + 1 < len(onsets) else int(batches.max()) + 1
            triggers = alerts[(alerts >= onset - horizon) & (alerts < next_onset)]
            before = values[(batches >= onset - pre_window) & (batches < onset)]
            trigger = int(triggers[0]) if triggers.size else None
            recovery_batch = None
            target = float(np.mean(before) * recovery_fraction) if before.size == pre_window else float("nan")
            if trigger is not None and np.isfinite(target):
                candidates = batches[(batches >= max(onset, trigger + 1)) & (batches < next_onset) & (rolling >= target)]
                if candidates.size:
                    recovery_batch = int(candidates[0])
            event_rows.append({
                "dataset": dataset, "method_key": method, "drift_onset": onset,
                "trigger_batch": trigger, "pre_drift_accuracy": float(np.mean(before)) if before.size else float("nan"),
                "recovery_threshold": target, "recovery_batch": recovery_batch,
                "accuracy_recovery_batches": recovery_batch - onset if recovery_batch is not None else float("nan"),
                "recovered": int(recovery_batch is not None),
            })
    return pd.DataFrame(event_rows), accuracy_frame
