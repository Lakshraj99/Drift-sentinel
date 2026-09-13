from pathlib import Path

import numpy as np

from driftsentinel.logistic import (
    fit_logistic_baseline,
    load_logistic_checkpoint,
    logistic_probabilities,
    save_logistic_checkpoint,
)


def test_logistic_uses_current_batch_and_checkpoint_round_trip(tmp_path: Path):
    sequences = np.zeros((20, 4, 2), dtype=np.float32)
    sequences[:, -1, 0] = np.arange(20)
    labels = (sequences[:, -1, 0] >= 10).astype(int)
    model = fit_logistic_baseline(sequences, labels, seed=11)
    original = logistic_probabilities(model, sequences)

    changed_history = sequences.copy()
    changed_history[:, :-1, :] = 1_000_000
    np.testing.assert_allclose(logistic_probabilities(model, changed_history), original)

    checkpoint = tmp_path / "logistic.pkl"
    metadata = {"seed": 11, "sequence_length": 1}
    save_logistic_checkpoint(model, checkpoint, metadata)
    restored, restored_metadata = load_logistic_checkpoint(checkpoint)
    assert restored_metadata == metadata
    np.testing.assert_allclose(logistic_probabilities(restored, sequences), original)
