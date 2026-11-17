"""Chronological train/validation/test split with an embargo.

Rows with Date < train_end -> train;  train_end <= Date < val_end -> validation;
Date >= val_end -> test. A row's label looks `horizon` days ahead, so the last
`embargo` trading days before each boundary are purged. Extra safety: any row
whose label_end_date reaches the next split's start is also dropped.
Calendar is shared by all tickers so split dates are consistent.
Complexity: O(n log n).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import LABEL_COLUMN


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    summary: pd.DataFrame
    embargo: int
    horizon: int
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    excluded_total: int


def default_split_dates(features: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2):
    dates = np.sort(features["Date"].unique())
    return pd.Timestamp(dates[int(len(dates) * train_frac)]), pd.Timestamp(dates[int(len(dates) * (train_frac + val_frac))])


def chronological_split(features: pd.DataFrame, train_end, val_end, embargo: int = 20,
                        horizon: int = 20) -> SplitResult:
    train_end, val_end = pd.Timestamp(train_end), pd.Timestamp(val_end)
    if embargo < horizon:
        raise ValueError(f"embargo ({embargo}) must be >= horizon ({horizon}) to prevent label leakage")
    if not train_end < val_end:
        raise ValueError("train_end must be before val_end")
    dates = np.sort(features["Date"].unique())
    if not (dates[0] < np.datetime64(train_end) and np.datetime64(val_end) <= dates[-1]):
        raise ValueError("Split dates must lie inside the data's date range")

    def cutoff(boundary: pd.Timestamp) -> pd.Timestamp:
        pos = int(np.searchsorted(dates, np.datetime64(boundary)))
        return pd.Timestamp(dates[max(pos - embargo, 0)])  # first purged date

    tr_cut, va_cut = cutoff(train_end), cutoff(val_end)
    labelled = features.dropna(subset=[LABEL_COLUMN, "label_end_date"])
    n_unlabelled = len(features) - len(labelled)
    d, e = labelled["Date"], labelled["label_end_date"]
    train = labelled[(d < tr_cut) & (e < train_end)]
    val = labelled[(d >= train_end) & (d < va_cut) & (e < val_end)]
    test = labelled[d >= val_end]
    excl_train = int(((d < train_end)).sum() - len(train))
    excl_val = int(((d >= train_end) & (d < val_end)).sum() - len(val))
    summary = pd.DataFrame([
        {"Split": "Train", "Start": train["Date"].min(), "End": train["Date"].max(), "Rows": len(train), "Excluded (embargo)": excl_train},
        {"Split": "Validation", "Start": val["Date"].min(), "End": val["Date"].max(), "Rows": len(val), "Excluded (embargo)": excl_val},
        {"Split": "Test", "Start": test["Date"].min(), "End": test["Date"].max(), "Rows": len(test), "Excluded (embargo)": n_unlabelled},
    ])
    return SplitResult(train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True),
                       summary, embargo, horizon, train_end, val_end, excl_train + excl_val + n_unlabelled)


def assert_no_label_leakage(split: SplitResult) -> None:
    """Raise AssertionError if any label window crosses into the next split."""
    if len(split.train) and not (split.train["label_end_date"] < split.train_end).all():
        raise AssertionError("Train labels reach into validation period")
    if len(split.val) and not (split.val["label_end_date"] < split.val_end).all():
        raise AssertionError("Validation labels reach into test period")
    if len(split.train) and len(split.val) and not split.train["Date"].max() < split.val["Date"].min():
        raise AssertionError("Train/validation overlap")
    if len(split.val) and len(split.test) and not split.val["Date"].max() < split.test["Date"].min():
        raise AssertionError("Validation/test overlap")
