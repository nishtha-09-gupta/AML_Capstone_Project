# Phase 2 results

Selected primary family by S-only CV: **MLP**.

Full run; all evaluation results are retrospective. Primary saved seed is 42.

## Cross-validation (fixed configurations; common five folds)

| Model | Mean F1 | Mean ROC-AUC | Mean PR-AUC | Mean Brier |
|---|---:|---:|---:|---:|
| BN | 0.8179 | 0.8780 | 0.8919 | 0.1487 |
| BN + MLP | 0.8283 | 0.8677 | 0.8746 | 0.1436 |
| Logistic regression | 0.8253 | 0.8699 | 0.8777 | 0.1424 |
| MLP | 0.8296 | 0.8686 | 0.8687 | 0.1430 |
| MLP no dropout | 0.8296 | 0.8723 | 0.8734 | 0.1411 |
| MLP no weight decay | 0.8310 | 0.8692 | 0.8694 | 0.1426 |
| NB + MLP | 0.8294 | 0.8668 | 0.8731 | 0.1444 |
| Naive Bayes | 0.8167 | 0.8740 | 0.8858 | 0.1519 |
| Random forest | 0.8317 | 0.8666 | 0.8713 | 0.1472 |

## Retrospective evaluation: seed 42, C-selected cutoffs

| Model | Cutoff | F1 | ROC-AUC | PR-AUC | Sensitivity | Specificity | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| BN | 0.54 | 0.8098 | 0.8765 | 0.8936 | 0.8137 | 0.7561 | 0.1544 |
| Naive Bayes | 0.13 | 0.8103 | 0.8772 | 0.8958 | 0.9216 | 0.5610 | 0.1532 |
| MLP | 0.48 | 0.8411 | 0.9071 | 0.9192 | 0.8824 | 0.7317 | 0.1236 |
| BN + MLP | 0.64 | 0.8358 | 0.9020 | 0.9172 | 0.8235 | 0.8171 | 0.1252 |
| Logistic regression | 0.38 | 0.8520 | 0.9022 | 0.9138 | 0.9314 | 0.6829 | 0.1241 |
| Random forest | 0.41 | 0.8634 | 0.8950 | 0.9046 | 0.9608 | 0.6707 | 0.1289 |
| MLP no dropout | 0.41 | 0.8393 | 0.9082 | 0.9219 | 0.9216 | 0.6585 | 0.1219 |
| MLP no weight decay | 0.48 | 0.8411 | 0.9067 | 0.9189 | 0.8824 | 0.7317 | 0.1236 |
| NB + MLP | 0.66 | 0.8283 | 0.9009 | 0.9164 | 0.8039 | 0.8293 | 0.1270 |

## Hybrid finding

BN + MLP does not outperform the MLP on mean selection-CV F1 or seed-42 retrospective F1/AUC/Brier. This is a functioning, technically motivated feature-stacking hybrid with a negative empirical result; the evidence does not establish an additional-marks performance claim. No evaluation-driven retuning was performed.

Paired hybrid-minus-MLP intervals (seed 42; fixed models and cutoffs):
- f1: 95% interval [-0.0407, 0.0276].
- roc_auc: 95% interval [-0.0157, 0.0058].
- brier: 95% interval [-0.0046, 0.0078].

These intervals condition on row-level predictions, omit training/selection and patient-cluster uncertainty, and are not formal superiority tests.

## Secondary analyses

See hospital_summary.csv and cleveland_summary.csv. Hospital models are independently fitted on other D hospitals; Cleveland CV uses the Cleveland slice of D only. All use fixed settings and seed 42.

## Interpretation limits

The outcome is recorded disease presence, not future risk. The supplied merged CSV lacks verified patient linkage. The graph is provisional. Five-bin calibration is descriptive, not proof of clinical calibration. Phase 1 tuned-model scores and Phase 2 fixed-setting control scores have unequal search budgets. Neither hospital transfer nor prospective clinical utility is established.
