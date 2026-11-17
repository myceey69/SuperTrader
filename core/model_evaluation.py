"""Regression metrics + model comparison (validation for selection, test for final)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be non-empty and equal length")
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": float(np.mean(np.abs(err))), "MSE": mse, "RMSE": float(np.sqrt(mse)), "R2": r2,
            "Directional Accuracy": float(np.mean(np.sign(y_pred) == np.sign(y_true)))}


def compare_models(preds_val: dict[str, np.ndarray], y_val, preds_test: dict[str, np.ndarray], y_test) -> pd.DataFrame:
    rows = []
    for name in preds_test:
        v, t = regression_metrics(y_val, preds_val[name]), regression_metrics(y_test, preds_test[name])
        rows.append({"Model": name, **{f"Val {k}": x for k, x in v.items()}, **{f"Test {k}": x for k, x in t.items()}})
    df = pd.DataFrame(rows)
    df["Best on Validation (RMSE)"] = df["Val RMSE"] == df["Val RMSE"].min()
    return df
