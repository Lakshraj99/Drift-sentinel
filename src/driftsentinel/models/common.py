"""Shared sequence-model components."""

from __future__ import annotations

import math
import torch
from torch import nn


class RecurrentClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, kind: str):
        super().__init__()
        recurrent = nn.GRU if kind == "gru" else nn.LSTM
        self.recurrent = recurrent(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.recurrent(x)
        return self.head(output[:, -1]).squeeze(-1)


class SinusoidalEncoding(nn.Module):
    def __init__(self, dimension: int, max_length: int = 512):
        super().__init__()
        positions = torch.arange(max_length).unsqueeze(1)
        div = torch.exp(torch.arange(0, dimension, 2) * (-math.log(10_000.0) / dimension))
        encoding = torch.zeros(max_length, dimension)
        encoding[:, 0::2] = torch.sin(positions * div)
        encoding[:, 1::2] = torch.cos(positions * div[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.shape[1]]
