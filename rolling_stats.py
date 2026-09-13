"""
rolling_stats.py
-----------------
Computes rolling-window summary statistics per batch of a data stream.
These statistics are the INPUT FEATURES fed to the DriftSentinel
sequence model (Member 2's job) later on.

For every batch we compute, per numeric feature:
  - mean shift   (current window mean  -  reference window mean)
  - variance shift (current window var -  reference window var)
  - PSI (Population Stability Index) vs. the reference window
And overall:
  - prediction-confidence trend (mean predicted probability of the
    predicted class, from a simple online model, e.g. Hoeffding Tree)

Usage: see build_dataset.py for how this is called batch-by-batch.
"""

import numpy as np
import pandas as pd


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index between two 1D numeric arrays."""
    if len(reference) < 2 or len(current) < 2:
        return 0.0
    # bin edges from the reference distribution
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(reference, quantiles))
    if len(edges) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1) + eps
    cur_pct = cur_counts / max(cur_counts.sum(), 1) + eps

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


class RollingStatTracker:
    """
    Maintains a reference window and a sliding current window of raw
    feature vectors + prediction confidences, and emits one summary
    row per `flush()` call (i.e. once per batch).
    """

    def __init__(self, feature_names, reference_size: int = 500, batch_size: int = 100):
        self.feature_names = list(feature_names)
        self.reference_size = reference_size
        self.batch_size = batch_size

        self._reference_buffer = []   # list of dict(feature -> value)
        self._current_buffer = []
        self._confidence_buffer = []
        self._reference_locked = False

    def add(self, x: dict, confidence: float):
        """Add one streaming instance (features dict) + its prediction confidence."""
        if not self._reference_locked:
            self._reference_buffer.append(x)
            if len(self._reference_buffer) >= self.reference_size:
                self._reference_locked = True
            return None  # no stats yet while building the reference window

        self._current_buffer.append(x)
        self._confidence_buffer.append(confidence)

        if len(self._current_buffer) >= self.batch_size:
            return self.flush()
        return None

    def flush(self):
        """Compute summary stats for the accumulated current batch and reset it."""
        if not self._current_buffer:
            return None

        ref_df = pd.DataFrame(self._reference_buffer)
        cur_df = pd.DataFrame(self._current_buffer)

        row = {}
        for feat in self.feature_names:
            if feat not in ref_df.columns or feat not in cur_df.columns:
                continue
            ref_vals = pd.to_numeric(ref_df[feat], errors="coerce").dropna().to_numpy()
            cur_vals = pd.to_numeric(cur_df[feat], errors="coerce").dropna().to_numpy()
            if len(ref_vals) == 0 or len(cur_vals) == 0:
                continue

            row[f"{feat}_mean_shift"] = float(cur_vals.mean() - ref_vals.mean())
            row[f"{feat}_var_shift"] = float(cur_vals.var() - ref_vals.var())
            row[f"{feat}_psi"] = psi(ref_vals, cur_vals)

        row["mean_confidence"] = float(np.mean(self._confidence_buffer))
        row["confidence_trend"] = float(
            np.polyfit(range(len(self._confidence_buffer)), self._confidence_buffer, 1)[0]
        ) if len(self._confidence_buffer) > 1 else 0.0
        row["batch_size"] = len(self._current_buffer)

        # slide the reference window forward (simple rolling-reference strategy)
        self._reference_buffer = (self._reference_buffer + self._current_buffer)[-self.reference_size:]
        self._current_buffer = []
        self._confidence_buffer = []

        return row
