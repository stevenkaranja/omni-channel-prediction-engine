"""Forecast evaluation metrics."""
import numpy as np


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error."""
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8) * 100)


def forecast_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return max(0.0, 100.0 - wape(y_true, y_pred))


def inventory_impact(y_true: np.ndarray, y_pred: np.ndarray, unit_cost: float) -> dict:
    over  = np.maximum(y_pred - y_true, 0)
    under = np.maximum(y_true - y_pred, 0)
    return {
        "overstock_units": float(over.sum()),
        "understock_units": float(under.sum()),
        "overstock_cost_usd": float(over.sum() * unit_cost * 0.3),
        "lost_sales_usd": float(under.sum() * unit_cost * 0.6),
    }
