# Stronger hybrid: what changed and how to reproduce it

The original BN-feature MLP remains in the repository with its negative result.
The follow-up adds a **calibrated BN–MLP stack** and a **BN residual MLP**. The
calibrated stack shows lower nested-CV Brier error than every tested control.
The residual model does not match it and is retained as an unfavorable ablation.

## Observed added value

All numbers below are means of outer validation scores on S only, with five
outer folds, three inner tuning folds and all three neural seeds retained.
They are not the original historical T metrics.

| Model | Brier ↓ | ROC-AUC | PR-AUC | F1 at 0.5 |
|---|---:|---:|---:|---:|
| Calibrated BN–MLP | 0.13913 | 0.87670 | 0.88437 | 0.82635 |
| Calibrated BN | 0.14038 | 0.87521 | 0.88280 | 0.82276 |
| Tuned MLP | 0.14168 | 0.86896 | 0.87384 | 0.82166 |
| Calibrated MLP | 0.14228 | 0.86883 | 0.87358 | 0.82174 |

The stack reduces mean Brier error by **1.80% relative to tuned MLP** and **0.89%
relative to calibrated BN**, the strongest single-base control. Each of the three
seed means improves on both controls; 11/15 seed/fold pairs beat calibrated BN.
The conditional 95% interval for the stack-minus-calibrated-BN Brier difference
is [-0.00405, 0.00162], which includes zero. The gain is observed and modest;
confirmatory superiority and bonus marks are not guaranteed.

## Why this is more than an arbitrary combination

The BN represents coarse class-conditional structure. The MLP retains continuous
inputs and nonlinear interactions. Fresh cross-fitted probabilities train a
penalized logistic meta-model that adjusts the confidence and relative influence
of these two different sources of evidence. Calibration-only BN and MLP controls
show whether a gain is caused merely by calibration; the observed stack Brier is
lower than both. Inner-fold model selection prevents the fusion weights and
regularization choices from seeing outer validation records. The code records
all competing configurations, coefficients, fold memberships and unfavorable
results. It implements standard stacking, not an invented algorithm.

Exact inference explanations decompose the final log-odds into the intercept,
BN contribution and MLP contribution. Contributions reconstruct the predicted
probability exactly; they are neither causal effects nor medical explanations.

## Commands

```bash
python -m pytest tests/test_phase2_hybrids.py -q
python phase2_hybrid_verify.py artifacts/phase2/hybrid_followup_v1
```

Reproduce the finite follow-up in a new directory:

```bash
python phase2_improve.py --output artifacts/phase2/my_hybrid_followup
python phase2_hybrid_verify.py artifacts/phase2/my_hybrid_followup
python phase2_hybrid_summary.py artifacts/phase2/my_hybrid_followup
python phase2_hybrid_freeze.py artifacts/phase2/my_hybrid_followup
python phase2_hybrid_verify.py artifacts/phase2/my_hybrid_followup
```

Batch inference from the saved seed-42 stack:

```bash
python phase2_hybrid_freeze.py artifacts/phase2/hybrid_followup_v1 --input-csv profiles.csv --output hybrid_predictions.csv
python phase2_hybrid_freeze.py artifacts/phase2/hybrid_followup_v1 --input-csv profiles.csv --explain --output hybrid_explanations.csv
```

Only load trusted model files. The revised models use a fixed 0.5 decision cutoff;
the original C-selected cutoffs do not apply to them. No C/T scores were produced
for the revised models. Final-model predictions on S are round-trip evidence,
not generalization estimates.

## Submission evidence

- `docs/phase2/hybrid_improvement_protocol.md`: finite study and declared criterion.
- `artifacts/phase2/hybrid_followup_v1/RESULTS.md`: all seven families and comparisons.
- `summary.csv`, `by_seed.csv`, `paired_comparisons.json`, `stack_coefficients.csv`:
  observed value, variability, uncertainty and inspectable fusion weights.
- `verification.json`: 105 outer models and 12 S-only final models checked.
- `docs/phase2/phase2_report_with_followup.tex` / `.pdf`: original study plus follow-up.
- `docs/phase2/phase2_presentation_with_followup.pptx`: 15 editable slides.

The follow-up was initiated after prior results were inspected. It reuses S for
exploratory research, excludes C/T from fitting and scoring, and cannot be called
new external validation. Patient linkage, provisional graph assumptions, faculty
acceptance, author attribution and viva performance remain qualifications.

References: [scikit-learn stacking documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
and [Guo et al., calibration](https://proceedings.mlr.press/v70/guo17a.html).
