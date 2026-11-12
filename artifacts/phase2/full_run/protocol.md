# Phase 2 protocol — frozen before experiments

Project 15: Heart Disease Risk Modelling via Bayesian Networks and Feed Forward
Neural Networks. This is historical disease-presence classification, num > 0,
not future-event prediction or clinical decision support.

## Data and comparisons

Preserve the source CSV, Phase 1 artifacts, eleven predictors, cleaning and seed-42
D/T split (734/184). Within D retain seed-43 S/C (550/184). T is retrospective.
All learned preprocessing is training-only. Row IDs are not proven patient IDs;
exact-record deduplication cannot establish patient independence.

Primary comparison: pgmpy structured BN, categorical Naive Bayes (same bins,
remove oldpeak -> slope), regularized PyTorch MLP, and cross-fitted BN-assisted
MLP. Include Logistic Regression and Random Forest as freshly fitted contextual
controls. The BN retains the explicitly provisional Phase 1 graph, alpha=1 per
CPT cell, training-only median/mode imputation and tertile cut points. Missing
inference evidence is marginalized exactly. Export all bins and CPTs.

Use fixed, modest architectures/settings rather than a large tuning search.
Five common stratified S folds, seed 44; neural seeds 42, 52, 62. A fixed model's
outer training rows are split 80/20 (seed 1729) for early stopping. No outer
validation, C or T data may fit preprocessing, bins, or weights or choose epochs.
Epoch selection uses validation BCE, patience 20, min_delta=0.0001, max 150 epochs.
After selecting epochs, refit preprocessing and network on all permitted training
rows for that many epochs. Batch size 32, AdamW learning rate 0.001.
MLP hidden widths (32,16), ReLU, dropout 0.2, weight decay 0.001.

## Hybrid hypothesis and leakage prevention

Append the standardized log-odds of BN probability to the ordinary numeric and
one-hot inputs. Hypothesis: a structured joint-distribution summary adds a useful
inductive bias while the network learns continuous residual interactions lost
by binning. This is feature-level stacking, not a claim of a novel algorithm.
Generate every hybrid training score using three-fold stratified cross-fitting
(seed 1729) within the permitted training rows. Each fold learns its own bins and
CPTs. For early stopping, construct cross-fitted scores within the optimization
subset alone; its BN and preprocessor never see early-stop validation rows.
Refit a BN on all permitted training rows only for future inference.
Probability clipping to [1e-6,1-1e-6] is only for the logit transform.

## Selection and evaluation

Rank the four primary families by mean five-fold F1 at 0.5, averaged over the
three predeclared neural seeds, with ROC-AUC then declared family order as ties.
No seed selection: retain every seed; use seed 42 for the primary saved-model
comparison and report the others separately. This is fixed-configuration CV,
not the Phase 1 nested hyperparameter search; report the unequal search budgets.
Fit final models on S, choose each cutoff on C via the existing 101-point F1
rule, freeze weights/cutoffs before T scoring; no S+C refit.

Report both 0.5 and C-selected operating points, ROC-AUC, trapezoidal PR-AUC,
average precision (separately), sensitivity, specificity, F1, Brier, log loss,
confusion matrices and five-bin reliability diagrams with counts. Calibration
is measured, not assumed or post-hoc optimized on T. Include constant-prior
probabilities and always-positive decisions. Paired row bootstrap intervals for
hybrid-minus-MLP metrics condition on the fitted predictions; they are not a
formal superiority test and omit training/patient-cluster uncertainty.

Ablations on identical S folds/seeds: remove BN input (the MLP), remove dropout,
remove weight decay, and replace BN input with Naive Bayes probability. These
are explanatory comparisons, not additional families selected using T.

Secondary analyses: four D-only hospital holdouts with local S/C and training;
Cleveland-only five-fold CV on the Cleveland records in D, excluding all T rows.
This is the supplied merged CSV's Cleveland slice, not an independently acquired
processed Cleveland file. Retain the same fixed settings and seed 42 for these
resource-bounded analyses. Report all outcomes, including negative results.

## Reproducibility and scope

Persist configuration before training, exact source IDs/folds, preprocessing,
BN cross-fit memberships and early-stop memberships, training curves, all seed
scores, frozen models, predictions, dataset/code hashes and package versions.
Verify saved-model round trips, metrics, cutoffs and disjoint partitions. No
clinical claim, faculty graph approval or guaranteed hybrid improvement is implied.
The requested scope is repository implementation and documentation; a final
presentation and IEEE report can use these verified results later.
