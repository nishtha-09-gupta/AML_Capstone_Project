import numpy as np
import pytest
from phase2_train import metrics, frozen_evaluation
from phase2_verify import require, compare
from data_loader import load_clean_dataset


def test_metrics_distinguish_pr_auc_and_average_precision():
    y = np.array([0,1,0,1])
    p = np.array([.1,.4,.7,.9])
    result = metrics(y,p)
    assert result['sensitivity'] == result['recall']
    assert result['average_precision'] != result['pr_auc_trapezoid']
    assert sum(b['n'] for b in result['reliability']) == len(y)
    assert result['confusion_matrix'] == [[1,1],[1,1]]


def test_partition_overlap_rejected_before_any_training(tmp_path):
    data,_ = load_clean_dataset()
    with pytest.raises(ValueError,match='disjoint'):
        frozen_evaluation(data.iloc[:50],data.iloc[40:60],data.iloc[60:80],
                          tmp_path/'experiment',['MLP'],[42],True)


def test_verifier_rejects_altered_metric():
    original = metrics(np.array([0,1,0,1]),np.array([.1,.4,.7,.9]))
    corrupt = dict(original, roc_auc=0.)
    with pytest.raises(ValueError,match='Metric mismatch'):
        compare(original,corrupt)
