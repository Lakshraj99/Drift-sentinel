"""Classification and event-aware early-warning metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(labels: Sequence[int], probabilities: Sequence[float], threshold: float = 0.5, continuous_scores: bool = True) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    both_classes = np.unique(y).size == 2
    return {
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y, p) if continuous_scores and both_classes else float("nan"),
        "pr_auc": average_precision_score(y, p) if continuous_scores and y.sum() else float("nan"),
        "accuracy": accuracy_score(y, predicted),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def alert_episode_starts(alert_batches: Sequence[int], cooldown: int = 1) -> np.ndarray:
    alerts = np.asarray(sorted(set(int(v) for v in alert_batches)), dtype=int)
    if not alerts.size:
        return alerts
    return alerts[np.r_[True, np.diff(alerts) > cooldown]]


def early_warning_metrics(
    batch_ids: Sequence[int],
    alerts: Sequence[int] | Sequence[bool],
    drift_onsets: Sequence[int],
    warning_window: int,
    cooldown: int = 1,
) -> dict[str, float | int]:
    """Measure only warnings strictly before each onset within the accepted window."""
    batches = np.asarray(batch_ids, dtype=int)
    raw = np.asarray(alerts)
    alert_batches = batches[raw.astype(bool)] if raw.size == batches.size else raw.astype(int)
    episodes = alert_episode_starts(alert_batches, cooldown)
    onsets = np.asarray(sorted(set(int(v) for v in drift_onsets)), dtype=int)
    valid_leads: list[int] = []
    matched_alerts: set[int] = set()
    for onset in onsets:
        candidates = episodes[(episodes >= onset - warning_window) & (episodes < onset) & ~np.isin(episodes, list(matched_alerts))]
        if candidates.size:
            warning = int(candidates[0])
            valid_leads.append(int(onset - warning))
            matched_alerts.add(warning)
    eligible = np.ones(len(batches), dtype=bool)
    for onset in onsets:
        eligible &= ~((batches >= onset - warning_window) & (batches < onset))
    false_batch_alerts = int(np.isin(batches[eligible], alert_batches).sum())
    false_episodes = int(sum(int(v) not in matched_alerts for v in episodes))
    detected_events = len(valid_leads)
    eligible_batches = int(eligible.sum())
    total_adaptations = int(len(episodes))
    false_adaptations = false_episodes
    true_event_related_adaptations = total_adaptations - false_adaptations
    return {
        "mean_lead_time": float(np.mean(valid_leads)) if valid_leads else 0.0,
        "median_lead_time": float(np.median(valid_leads)) if valid_leads else 0.0,
        "std_lead_time": float(np.std(valid_leads)) if valid_leads else 0.0,
        "detection_coverage": detected_events / len(onsets) if len(onsets) else float("nan"),
        "missed_drift_rate": 1.0 - detected_events / len(onsets) if len(onsets) else float("nan"),
        "batch_far": false_batch_alerts / eligible_batches if eligible_batches else 0.0,
        "event_far": false_episodes / len(episodes) if len(episodes) else 0.0,
        "alert_episodes": int(len(episodes)),
        "false_alert_episodes": false_episodes,
        "drift_events": int(len(onsets)),
        "detected_events": detected_events,
        "eligible_batches": eligible_batches,
        "false_batch_alerts": false_batch_alerts,
        "lead_time_sum": int(sum(valid_leads)),
        "evaluated_batches": int(len(batches)),
        "total_adaptations": total_adaptations,
        "adaptations_per_100_batches": 100.0 * total_adaptations / len(batches) if len(batches) else 0.0,
        "false_adaptations": false_adaptations,
        "true_event_related_adaptations": true_event_related_adaptations,
        "adaptation_precision": true_event_related_adaptations / total_adaptations if total_adaptations else 0.0,
    }


def threshold_sweep(batch_ids: Sequence[int], probabilities: Sequence[float], drift_onsets: Sequence[int], warning_window: int, cooldown: int, thresholds: Sequence[float] | None = None) -> pd.DataFrame:
    values = np.asarray(probabilities, dtype=float)
    thresholds = np.linspace(0.05, 0.95, 19) if thresholds is None else thresholds
    rows = []
    for threshold in thresholds:
        metrics = early_warning_metrics(batch_ids, values >= threshold, drift_onsets, warning_window, cooldown)
        rows.append({"threshold": float(threshold), **metrics})
    return pd.DataFrame(rows)
