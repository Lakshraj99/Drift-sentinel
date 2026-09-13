"""Reusable deterministic training, prediction, and checkpoint helpers."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import SequenceDataset


@dataclass
class TrainingResult:
    model: nn.Module
    history: list[dict[str, float]]
    best_epoch: int


def _loss_for_labels(labels: np.ndarray, device: torch.device) -> nn.Module:
    positives = float(labels.sum())
    negatives = float(len(labels) - positives)
    weight = negatives / positives if positives > 0 else 1.0
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))


def train_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
) -> TrainingResult:
    if not len(train_y) or not len(validation_y):
        raise ValueError("Training and validation partitions must be non-empty")
    model = model.to(device)
    loss_fn = _loss_for_labels(train_y, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader = DataLoader(SequenceDataset(train_x, train_y), batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(SequenceDataset(validation_x, validation_y), batch_size=batch_size, shuffle=False)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_total += loss.item() * len(labels)
        model.eval()
        validation_total = 0.0
        with torch.no_grad():
            for features, labels in validation_loader:
                features, labels = features.to(device), labels.to(device)
                validation_total += loss_fn(model(features), labels).item() * len(labels)
        train_loss = train_total / len(train_y)
        validation_loss = validation_total / len(validation_y)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return TrainingResult(model, history, best_epoch)


def predict_probabilities(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    loader = DataLoader(SequenceDataset(x, np.zeros(len(x), dtype=np.float32)), batch_size=batch_size, shuffle=False)
    values = []
    with torch.no_grad():
        for features, _ in loader:
            values.append(torch.sigmoid(model(features.to(device))).cpu().numpy())
    return np.concatenate(values) if values else np.empty(0)


def save_checkpoint(model: nn.Module, path: str | Path, metadata: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, target)


def load_checkpoint(path: str | Path, model: nn.Module, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device), checkpoint["metadata"]
