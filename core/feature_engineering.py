"""Feature engineering + continuous forward-return label.

Every feature at row t uses only data up to and including day t, computed
independently per ticker. Label: forward_return = Close[t+h]/Close[t] - 1
(never used as a feature). `label_end_date` records the date the label needs,
so leakage can be audited.
Complexity: O(n) per ticker (rolling windows).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS, LABEL_COLUMN


def build_features(prices: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    parts = []
    for ticker, g in prices.groupby("Ticker", sort=True):
        g = g.sort_values("Date").reset_index(drop=True)
        c, v = g["Close"], g["Volume"].astype(float)
        f = pd.DataFrame({"Date": g["Date"], "Ticker": ticker, "Close": c})
        f["daily_return"] = c.pct_change()
        f["return_5d"] = c.pct_change(5)
        f["return_20d"] = c.pct_change(20)
        f["ma5_ratio"] = c / c.rolling(5).mean() - 1
        f["ma20_ratio"] = c / c.rolling(20).mean() - 1
        f["volatility_20d"] = f["daily_return"].rolling(20).std()
        for w in (5, 20):  # zero/missing volume -> neutral ratio 1.0
            f[f"volume_ratio_{w}d"] = (v / v.rolling(w).mean().replace(0, np.nan)).fillna(1.0)
        f["momentum_10d"] = c / c.shift(10) - 1
        f[LABEL_COLUMN] = c.shift(-horizon) / c - 1          # FUTURE info: label only
        f["label_end_date"] = g["Date"].shift(-horizon)
        parts.append(f)
    out = pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=FEATURE_COLUMNS)
    return out.sort_values(["Date", "Ticker"]).reset_index(drop=True)
