"""Feature engineering: Snowflake → training-ready tensors (v2)."""
import numpy as np
import pandas as pd
import yaml
from typing import Tuple


class FeatureEngineer:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.seq_len = self.cfg["model"]["seq_len"]
        self.horizon = self.cfg["model"]["pred_horizon"]

    def build_training_windows(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        feature_cols = [
            "units_sold_7d", "units_sold_14d", "units_sold_28d", "units_sold_90d",
            "avg_price_7d", "seasonality_idx", "promotion_active",
        ]
        X, y = [], []
        for _, grp in df.groupby(["location_id", "sku"]):
            grp = grp.sort_values("feature_date").reset_index(drop=True)
            vals = grp[feature_cols].fillna(0).values
            # FIX: shift target forward instead of backward — prevents data leakage
            targets = grp["units_sold_7d"].values
            for i in range(len(vals) - self.seq_len - self.horizon):
                X.append(vals[i : i + self.seq_len])
                # target window starts after the input sequence ends
                y.append(targets[i + self.seq_len : i + self.seq_len + self.horizon])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def add_seasonality(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        doy = pd.to_datetime(df["feature_date"]).dt.dayofyear
        df["seasonality_idx"] = (
            0.5 * np.sin(2 * np.pi * doy / 365)
            + 0.5 * np.cos(4 * np.pi * doy / 365)
        )
        return df

    def normalise(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu = X.mean(axis=(0, 1), keepdims=True)
        sigma = X.std(axis=(0, 1), keepdims=True) + 1e-8
        return (X - mu) / sigma, mu, sigma
