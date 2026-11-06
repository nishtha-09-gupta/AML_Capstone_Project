import joblib
import numpy as np
import pytest
import torch
from sklearn.base import clone
from data_loader import load_clean_dataset
from phase2_models import HeartMLP, cross_fit_bn


@pytest.fixture
def sample():
    data, x = load_clean_dataset()
    return x.iloc[:90], (data.num.iloc[:90] > 0).astype(int)


def test_cross_fitting_covers_rows_once_and_excludes_own_label(sample):
    x, y = sample
    p, folds = cross_fit_bn(x, y)
    covered = []
    for fold in folds:
        train, valid = set(fold['train_indices']), set(fold['validation_indices'])
        assert train.isdisjoint(valid)
        assert train | valid == set(x.index)
        covered.extend(valid)
    assert sorted(covered) == sorted(x.index)
    assert np.isfinite(p).all()
    # Changing one validation label (keeping fold allocation fixed) must not
    # alter its own score: spy on actual BN fits rather than the recorded audit.
    from unittest.mock import patch
    from phase2_baselines import DiscreteHeartModel
    events = []
    original = DiscreteHeartModel.fit
    def record(model, frame, target):
        events.append(set(frame.index))
        return original(model, frame, target)
    with patch.object(DiscreteHeartModel, 'fit', record):
        cross_fit_bn(x, y)
    for event, fold in zip(events, folds):
        assert event == set(fold['train_indices'])
        assert event.isdisjoint(fold['validation_indices'])


@pytest.mark.parametrize('hybrid', ['none', 'bn', 'nb'])
def test_deterministic_roundtrip_and_early_stop_isolation(sample, hybrid, tmp_path):
    torch.set_num_threads(1)
    x, y = sample
    estimator = HeartMLP(hybrid=hybrid, max_epochs=3, patience=2)
    first = clone(estimator).fit(x, y)
    second = clone(estimator).fit(x, y)
    np.testing.assert_array_equal(first.predict_proba(x), second.predict_proba(x))
    assert 1 <= first.best_epoch_ <= 3
    a = first.audit_
    assert set(a['early_train_indices']).isdisjoint(a['early_validation_indices'])
    for fold in a['early_bn_folds']:
        assert set(fold['train_indices']) <= set(a['early_train_indices'])
        assert set(fold['validation_indices']) <= set(a['early_train_indices'])
    path = tmp_path / 'model.joblib'
    joblib.dump(first, path)
    np.testing.assert_array_equal(first.predict_proba(x), joblib.load(path).predict_proba(x))
    assert np.isfinite(first.predict_proba(x)).all()
    np.testing.assert_allclose(first.predict_proba(x).sum(axis=1), 1, atol=1e-7)


def test_real_preprocessing_excludes_early_validation(sample):
    from unittest.mock import patch
    from sklearn.impute import SimpleImputer
    x, y = sample
    calls = []
    original = SimpleImputer.fit
    def record(model, frame, *args, **kwargs):
        calls.append(set(frame.index))
        return original(model, frame, *args, **kwargs)
    with patch.object(SimpleImputer, 'fit', record):
        model = HeartMLP(max_epochs=2).fit(x, y)
    assert calls[:2] == [set(model.audit_['early_train_indices'])] * 2
    assert calls[2:] == [set(x.index)] * 2
