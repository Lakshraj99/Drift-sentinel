"""Leakage-safe current-batch Logistic Regression baseline."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_logistic_baseline(
    train_sequences: np.ndarray,
    train_labels: np.ndarray,
    seed: int,
    *,
    c: float = 1.0,
    max_iter: int = 1_000,
) -> LogisticRegression:
    """Fit on the scaled current batch only (the final token of each sequence)."""
    model = LogisticRegression(
        C=c,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=seed,
        solver="liblinear",
    )
    model.fit(train_sequences[:, -1, :], train_labels.astype(int))
    return model


def logistic_probabilities(model: LogisticRegression, sequences: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities from current-batch features only."""
    return model.predict_proba(sequences[:, -1, :])[:, 1]


def save_logistic_checkpoint(model: LogisticRegression, path: str | Path, metadata: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump({"model": model, "metadata": metadata}, handle)


def load_logistic_checkpoint(path: str | Path) -> tuple[LogisticRegression, dict]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    return payload["model"], payload["metadata"]
