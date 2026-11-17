import numpy as np
import pytest

from core.data_splitter import assert_no_label_leakage, chronological_split, default_split_dates
from core.feature_engineering import build_features
from core.preprocessing import FeatureScaler


@pytest.fixture(scope="module")
def feats(prices):
    return build_features(prices, 20)


def test_chronological_and_embargo(feats):
    ts, vs = default_split_dates(feats)
    sp = chronological_split(feats, ts, vs, 20, 20)
    assert sp.train.Date.max() < sp.val.Date.min() <= sp.val.Date.max() < sp.test.Date.min()
    dates = np.sort(feats.Date.unique())
    gap_tv = np.searchsorted(dates, sp.val.Date.min()) - np.searchsorted(dates, sp.train.Date.max())
    assert gap_tv >= 20                       # embargo between train and validation
    assert_no_label_leakage(sp)
    assert sp.excluded_total > 0 and len(sp.summary) == 3


def test_no_label_crosses_boundary(feats):
    ts, vs = default_split_dates(feats)
    sp = chronological_split(feats, ts, vs, 25, 20)
    assert (sp.train.label_end_date < ts).all() and (sp.val.label_end_date < vs).all()


def test_rejects_small_embargo_and_bad_dates(feats):
    ts, vs = default_split_dates(feats)
    with pytest.raises(ValueError):
        chronological_split(feats, ts, vs, 5, 20)
    with pytest.raises(ValueError):
        chronological_split(feats, vs, ts, 20, 20)


def test_leakage_checker_detects_violation(feats):
    ts, vs = default_split_dates(feats)
    sp = chronological_split(feats, ts, vs, 20, 20)
    sp.train.loc[sp.train.index[-1], "label_end_date"] = vs
    with pytest.raises(AssertionError):
        assert_no_label_leakage(sp)


def test_scaler_fit_on_train_only(feats):
    ts, vs = default_split_dates(feats)
    sp = chronological_split(feats, ts, vs, 20, 20)
    sc = FeatureScaler().fit(sp.train)
    assert np.allclose(sc.transform(sp.train).mean(axis=0), 0, atol=1e-4)
    assert not np.allclose(sc.transform(sp.test).mean(axis=0), 0, atol=1e-6)  # test not re-centred
    assert np.allclose(sc.scaler.mean_, sp.train[sc.columns].mean().to_numpy())
