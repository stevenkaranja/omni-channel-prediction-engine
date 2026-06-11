import numpy as np
import pytest
from models.evaluation import smape, wape, forecast_accuracy

def test_perfect_forecast():
    y = np.array([10., 20., 30.])
    assert forecast_accuracy(y, y) == 100.0

def test_zero_baseline():
    y_true = np.array([10., 20., 30.])
    y_pred = np.zeros_like(y_true)
    assert wape(y_true, y_pred) == pytest.approx(100.0)
