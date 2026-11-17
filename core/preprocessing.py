"""Training-only standardization (scikit-learn StandardScaler)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import FEATURE_COLUMNS


class FeatureScaler:
    """Fit on TRAIN only; transform validation/test/current with the fitted stats."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns or FEATURE_COLUMNS
        self.scaler = StandardScaler()
        self.fitted = False

    def _check(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.columns].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("Non-finite values in feature matrix")
        return x

    def fit(self, train: pd.DataFrame) -> "FeatureScaler":
        if len(train) == 0:
            raise ValueError("Cannot fit scaler on empty training set")
        self.scaler.fit(self._check(train))   # constant columns get scale_=1 (no divide-by-zero)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Scaler is not fitted")
        return self.scaler.transform(self._check(df)).astype(np.float32)

    def stats(self) -> pd.DataFrame:
        var = self.scaler.var_
        return pd.DataFrame({"Feature": self.columns, "Mean": self.scaler.mean_,
                             "Std": np.sqrt(var), "Constant": var == 0})
