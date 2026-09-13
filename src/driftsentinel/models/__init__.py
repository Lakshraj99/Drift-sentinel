"""Lightweight model factory."""

from torch import nn

from .gru import GRUClassifier
from .lstm import LSTMClassifier
from .transformer import TransformerClassifier


def build_model(name: str, input_dim: int, **kwargs: float | int) -> nn.Module:
    models = {"gru": GRUClassifier, "lstm": LSTMClassifier, "transformer": TransformerClassifier}
    try:
        return models[name.lower()](input_dim=input_dim, **kwargs)
    except KeyError as exc:
        raise ValueError(f"Unknown model: {name}") from exc


__all__ = ["GRUClassifier", "LSTMClassifier", "TransformerClassifier", "build_model"]
