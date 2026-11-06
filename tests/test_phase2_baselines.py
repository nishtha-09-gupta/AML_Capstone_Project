import numpy as np
import pytest
from sklearn.base import clone
from advanced_modeling import clean_features
from data_loader import load_clean_dataset
from bayesian_network import HeartBayesianNetwork
from phase2_baselines import DiscreteHeartModel


@pytest.fixture
def sample():
    data, x = load_clean_dataset()
    return x.iloc[:150], (data.num.iloc[:150] > 0).astype(int)


@pytest.mark.parametrize("structured", [True, False])
def test_pgmpy_matches_fast_inference_with_missing_evidence(sample, structured):
    x, y = sample
    model = clone(DiscreteHeartModel(structured=structured)).fit(x, y)
    query = x.iloc[:12].copy()
    query.iloc[0] = np.nan
    query.loc[query.index[1:4], ["oldpeak", "slope"]] = np.nan
    np.testing.assert_allclose(model.predict_proba(query), model.predict_pgmpy(query), atol=1e-12)
    assert len(model.model_.edges()) == (12 if structured else 11)
    np.testing.assert_allclose(model.predict_proba(query).sum(axis=1), 1)


def test_structured_model_preserves_phase1_probabilities(sample):
    x, y = sample
    previous = HeartBayesianNetwork(allow_provisional=True).fit(x, y)
    current = DiscreteHeartModel().fit(x, y)
    np.testing.assert_allclose(previous.predict_proba(x), current.predict_proba(x), atol=1e-12)
    for feature in current.bin_edges_:
        np.testing.assert_array_equal(previous.bin_edges_[feature], current.bin_edges_[feature])


def test_inference_never_changes_training_bins(sample):
    x, y = sample
    model = DiscreteHeartModel().fit(x, y)
    before = {k: v.copy() for k, v in model.bin_edges_.items()}
    query = x.iloc[:3].copy()
    query["chol"] = 699
    model.predict_proba(query)
    for key in before:
        np.testing.assert_array_equal(before[key], model.bin_edges_[key])
