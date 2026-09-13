"""LSTM ablation model."""

from .common import RecurrentClassifier


class LSTMClassifier(RecurrentClassifier):
    def __init__(self, input_dim: int, hidden_dim: int = 48, num_layers: int = 1, dropout: float = 0.15):
        super().__init__(input_dim, hidden_dim, num_layers, dropout, "lstm")
