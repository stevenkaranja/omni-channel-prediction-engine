"""Feature engineering: Snowflake → training-ready tensors."""
import numpy as np
import pandas as pd
import snowflake.connector
import yaml
from typing import Tuple

class FeatureEngineer:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

    def build_training_windows(
        self, df: pd.DataFrame, seq_len: int = 90, horizon: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Slide a window over sorted time series to build (X, y) pairs."""
        feature_cols = [
            "units_sold_7d", "units_sold_14d", "units_sold_28d", "units_sold_90d",
            "avg_price_7d", "seasonality_idx", "promotion_active",
        ]
        X, y = [], []
        for _, grp in df.groupby(["location_id", "sku"]):
            grp = grp.sort_values("feature_date")
            vals = grp[feature_cols].fillna(0).values
            targets = grp["units_sold_7d"].shift(-horizon).fillna(0).values
            for i in range(len(vals) - seq_len - horizon):
                X.append(vals[i : i + seq_len])
                y.append(targets[i + seq_len : i + seq_len + horizon])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def add_seasonality(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["day_of_year"] = pd.to_datetime(df["feature_date"]).dt.dayofyear
        df["seasonality_idx"] = (
            0.5 * np.sin(2 * np.pi * df["day_of_year"] / 365)
            + 0.5 * np.cos(2 * np.pi * df["day_of_year"] / 365 * 2)
        )
        return df
