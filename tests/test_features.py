import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS
from core.feature_engineering import build_features


def test_label_formula_and_no_cross_ticker_mixing(prices):
    f = build_features(prices, 20)
    for t, g in prices.groupby("Ticker"):
        g = g.sort_values("Date").reset_index(drop=True)
        ft = f[f.Ticker == t].set_index("Date")
        d = ft.index[5]
        i = g.index[g.Date == d][0]
        if i + 20 < len(g):
            assert np.isclose(ft.loc[d, "forward_return"], g.Close[i + 20] / g.Close[i] - 1)
            assert ft.loc[d, "label_end_date"] == g.Date[i + 20]


def test_features_use_only_past(prices):
    """Truncating future data must not change feature values at earlier dates."""
    full = build_features(prices, 20)
    cut_date = sorted(prices.Date.unique())[300]
    part = build_features(prices[prices.Date <= cut_date], 20)
    m = full.merge(part, on=["Date", "Ticker"], suffixes=("_f", "_p"))
    for c in FEATURE_COLUMNS:
        assert np.allclose(m[c + "_f"], m[c + "_p"])


def test_columns_finite(prices):
    f = build_features(prices, 20)
    assert np.isfinite(f[FEATURE_COLUMNS].to_numpy()).all()
    assert f["forward_return"].isna().sum() == 20 * prices.Ticker.nunique()   # unlabeled tail only
