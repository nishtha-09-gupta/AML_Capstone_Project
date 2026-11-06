import joblib
import numpy as np
import pytest
import torch
from unittest.mock import patch
from sklearn.base import clone
from data_loader import load_clean_dataset
from phase2_hybrids import BNResidualMLP, CalibratedHybrid, base_pair
from phase2_models import HeartMLP
from phase2_baselines import DiscreteHeartModel


@pytest.fixture
def sample():
    torch.set_num_threads(1)
    data,x=load_clean_dataset()
    return x.iloc[:90],(data.num.iloc[:90]>0).astype(int)


def test_zero_residual_is_exact_bn_offset(sample):
    x,y=sample
    model=BNResidualMLP(max_epochs=2)
    matrix,bundle=model._prepare_fit(x,y)
    net=model._network(matrix.shape[1]);net.eval()
    actual=net(torch.as_tensor(matrix)).detach().numpy().ravel()
    np.testing.assert_allclose(actual,matrix[:,-1],atol=1e-7)


@pytest.mark.parametrize('estimator',[BNResidualMLP(max_epochs=2),CalibratedHybrid(max_epochs=2)])
def test_roundtrip_determinism_and_training_isolation(sample,estimator,tmp_path):
    x,y=sample
    a=clone(estimator).fit(x,y);b=clone(estimator).fit(x,y)
    np.testing.assert_allclose(a.predict_proba(x),b.predict_proba(x),atol=0,rtol=0)
    file=tmp_path/'model.joblib';joblib.dump(a,file)
    np.testing.assert_allclose(a.predict_proba(x),joblib.load(file).predict_proba(x))
    folds=a.cross_fit_audit_ if hasattr(a,'cross_fit_audit_') else a.audit_['final_bn_folds']
    for fold in folds:
        assert set(fold['train_indices']).isdisjoint(fold['validation_indices'])
        assert set(fold['train_indices'])|set(fold['validation_indices'])==set(x.index)


def test_meta_inputs_never_use_own_rows(sample):
    x,y=sample
    events=[];original=HeartMLP.fit
    def record(model,frame,target):
        events.append(set(frame.index))
        return original(model,frame,target)
    with patch.object(HeartMLP,'fit',record):
        package=base_pair(x,y,max_epochs=2)
    for fitted,fold in zip(events[:3],package[2]):
        assert fitted==set(fold['train_indices'])
        assert fitted.isdisjoint(fold['validation_indices'])
    for component,features in [('bn',1),('mlp',1),('both',2)]:
        model=CalibratedHybrid(component=component,max_epochs=2).fit(x,y,package)
        assert model.meta_.coef_.shape==(1,features)


def test_training_cache_keys_include_target_values(sample):
    from phase2_improve import TrainingCache
    x,y=sample
    cache=TrainingCache(42,smoke=True)
    first=cache.fit('BN',None,x,y)
    assert cache.fit('BN',None,x,y) is first
    changed=cache.fit('BN',None,x,1-y)
    assert changed is not first
