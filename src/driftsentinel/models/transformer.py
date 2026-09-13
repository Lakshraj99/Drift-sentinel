"""Compact Transformer drift forecaster."""

import torch
from torch import nn

from .common import SinusoidalEncoding


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 48, num_layers: int = 1, dropout: float = 0.15):
        super().__init__()
        heads = 4 if hidden_dim % 4 == 0 else 2 if hidden_dim % 2 == 0 else 1
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.position = SinusoidalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 2, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.position(self.projection(x)))
        return self.head(encoded[:, -1]).squeeze(-1)
