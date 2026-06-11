"""PSI-based feature drift monitor."""
import numpy as np


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index — flag if > 0.2."""
    eps = 1e-8
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    e_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected) + eps
    a_pct = np.histogram(actual,   bins=breakpoints)[0] / len(actual)   + eps
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def check_drift(baseline: dict, current: dict, threshold: float = 0.2) -> dict:
    alerts = {}
    for feat in baseline:
        score = psi(np.array(baseline[feat]), np.array(current[feat]))
        if score > threshold:
            alerts[feat] = {"psi": round(score, 4), "status": "DRIFT_DETECTED"}
    return alerts
