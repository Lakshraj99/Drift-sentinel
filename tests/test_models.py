import pytest
import torch

from driftsentinel.models import GRUClassifier, LSTMClassifier, TransformerClassifier


@pytest.mark.parametrize("model_class", [GRUClassifier, LSTMClassifier, TransformerClassifier])
def test_model_forward_shape(model_class):
    model = model_class(input_dim=7, hidden_dim=16, num_layers=1, dropout=0.1)
    assert model(torch.randn(4, 10, 7)).shape == (4,)
