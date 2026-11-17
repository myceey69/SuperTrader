"""Baseline + Random Forest *regressor* (Assignment 2's classifier reconstructed
as a regressor so all models are compared on the same continuous target)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class MeanReturnBaseline:
    """Always predicts the training-set mean forward return."""
    def fit(self, x, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, x):
        return np.full(len(x), self.mean_)


def build_random_forest(n_estimators: int = 100, seed: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=8, min_samples_leaf=10,
                                 random_state=seed, n_jobs=1)
