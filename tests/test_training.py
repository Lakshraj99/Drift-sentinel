from pathlib import Path

import torch

from driftsentinel.models import GRUClassifier
from driftsentinel.training import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path: Path):
    source = GRUClassifier(3, hidden_dim=8)
    path = tmp_path / "model.pt"
    save_checkpoint(source, path, {"input_dim": 3})
    loaded, metadata = load_checkpoint(path, GRUClassifier(3, hidden_dim=8), torch.device("cpu"))
    assert metadata["input_dim"] == 3
    for expected, actual in zip(source.parameters(), loaded.parameters()):
        assert torch.equal(expected, actual)
