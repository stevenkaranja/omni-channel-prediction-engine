import torch
import pytest
from models.transformer_forecaster import DemandForecaster

@pytest.fixture
def model():
    return DemandForecaster(input_dim=7, d_model=64, nhead=4, num_layers=2, pred_horizon=30)

def test_output_shape(model):
    x = torch.randn(8, 90, 7)
    out = model(x)
    assert out.shape == (8, 30)

def test_no_nan(model):
    x = torch.randn(4, 90, 7)
    assert not torch.isnan(model(x)).any()
