import numpy as np
import pytest

from core.model_evaluation import compare_models, regression_metrics
from core.neural_network import ANNRegressor, ReturnNet
from core.random_forest import MeanReturnBaseline, build_random_forest


def toy(n=300, seed=0):
    r = np.random.default_rng(seed)
    x = r.normal(size=(n, 9)).astype(np.float32)
    return x, 0.05 * x[:, 0] + r.normal(0, 0.005, n)


def test_architecture_default():
    shapes = [tuple(p.shape) for p in ReturnNet(9).parameters()]
    assert shapes == [(64, 9), (64,), (32, 64), (32,), (1, 32), (1,)]


def test_ann_learns_and_is_reproducible():
    xt, yt = toy(300, 0)
    xv, yv = toy(100, 1)
    a = ANNRegressor(epochs=30, seed=1)
    a.fit(xt, yt, xv, yv)
    b = ANNRegressor(epochs=30, seed=1)
    b.fit(xt, yt, xv, yv)
    assert np.allclose(a.predict(xv), b.predict(xv))
    assert a.history.val_loss[-1] < a.history.val_loss[0]
    assert regression_metrics(yv, a.predict(xv))["R2"] > 0.5
    assert a.predict(xv).shape == (100,)


def test_ann_input_validation():
    x, y = toy(50)
    with pytest.raises(ValueError):
        ANNRegressor(hidden=(0,))
    with pytest.raises(ValueError):
        ANNRegressor(epochs=1).fit(np.full_like(x, np.nan), y, x, y)


def test_metrics_values():
    m = regression_metrics([1, -1, 2], [1, 1, 1])
    assert np.isclose(m["MAE"], 1.0) and np.isclose(m["MSE"], 5 / 3)
    assert np.isclose(m["Directional Accuracy"], 2 / 3)
    assert np.isclose(regression_metrics([1, 2, 3], [1, 2, 3])["R2"], 1.0)


def test_comparison_table():
    xt, yt = toy(300)
    xv, yv = toy(100, 1)
    models = {"base": MeanReturnBaseline().fit(xt, yt), "rf": build_random_forest(20).fit(xt, yt)}
    df = compare_models({k: m.predict(xv) for k, m in models.items()}, yv, {k: m.predict(xv) for k, m in models.items()}, yv)
    assert len(df) == 2 and df["Best on Validation (RMSE)"].sum() == 1
    assert df.set_index("Model").loc["rf", "Test RMSE"] < df.set_index("Model").loc["base", "Test RMSE"]
