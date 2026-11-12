# Hybrid improvement study: exploratory follow-up

Original Phase 2 results are preserved. This follow-up uses S only (550 records); C and T (368 records) are excluded from fitting and scoring.

## Mechanism and comparison

BN residual MLP starts at BN log-odds and learns continuous corrections. Calibrated BN–MLP learns a penalized logistic combination of fresh, cross-fitted BN and MLP logits. Calibrated BN and calibrated MLP controls distinguish fusion value from calibration alone.

Five outer stratified folds, three inner tuning folds, and all three seeds are retained. Inner Brier chooses each configuration. The primary outcome is Brier; this is a documented follow-up criterion, not a replacement for the original F1-first experiment.

## Nested outer results

| Model | Brier ↓ | ROC-AUC | PR-AUC | F1 at 0.5 | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| Calibrated BN-MLP | 0.13913 | 0.87670 | 0.88437 | 0.82635 | 0.83902 | 0.76422 |
| Calibrated BN | 0.14038 | 0.87521 | 0.88280 | 0.82276 | 0.82585 | 0.77633 |
| Original BN feature MLP | 0.14150 | 0.87068 | 0.87741 | 0.82452 | 0.84891 | 0.74117 |
| Tuned MLP | 0.14168 | 0.86896 | 0.87384 | 0.82166 | 0.84233 | 0.74381 |
| Calibrated MLP | 0.14228 | 0.86883 | 0.87358 | 0.82174 | 0.84124 | 0.74653 |
| BN residual MLP | 0.14775 | 0.87191 | 0.87983 | 0.82247 | 0.83355 | 0.76288 |
| BN | 0.14868 | 0.87521 | 0.88280 | 0.82290 | 0.81929 | 0.78857 |

## What the evidence establishes

Best evaluated hybrid: **Calibrated BN-MLP**. Strongest single-base control by Brier: **Calibrated BN**. Relative Brier reduction: **0.89%** (negative means the hybrid was worse).

Lower Brier than every control: **True**. Improvement versus the strongest control in every seed: **True**.

## Paired differences (hybrid minus control; negative Brier difference favors hybrid)

| Comparison | Brier difference | Conditional 95% interval | Winning seed/fold pairs |
|---|---:|---|---:|
| Calibrated BN-MLP minus BN | -0.00956 | [-0.01701, -0.00248] | 15/15 |
| Calibrated BN-MLP minus Tuned MLP | -0.00255 | [-0.00830, 0.00280] | 10/15 |
| Calibrated BN-MLP minus Calibrated BN | -0.00126 | [-0.00405, 0.00162] | 11/15 |
| Calibrated BN-MLP minus Calibrated MLP | -0.00316 | [-0.00873, 0.00199] | 9/15 |
| Calibrated BN-MLP minus Original BN feature MLP | -0.00237 | [-0.00689, 0.00205] | 10/15 |
| BN residual MLP minus BN | -0.00093 | [-0.00708, 0.00546] | 9/15 |
| BN residual MLP minus Tuned MLP | 0.00608 | [-0.00335, 0.01516] | 2/15 |
| BN residual MLP minus Calibrated BN | 0.00737 | [0.00056, 0.01418] | 0/15 |
| BN residual MLP minus Calibrated MLP | 0.00547 | [-0.00380, 0.01495] | 1/15 |
| BN residual MLP minus Original BN feature MLP | 0.00626 | [-0.00150, 0.01419] | 1/15 |

## How to present this

Describe exactly which metric improves and against which controls, including seed consistency and intervals. Do not claim uniform metric superiority, statistical confirmation, clinical benefit, algorithmic novelty or guaranteed marks. Report unfavorable residual-model and original-hybrid results too.

This is standard feature fusion and residual learning applied to the assigned BN/MLP gap. The implementation rationale, nested selection, calibration-only controls, saved coefficients and ablations make the contribution inspectable.

## Limits

The follow-up was motivated after prior results were inspected. The selection records are reused for exploratory research; outer folds limit within-study tuning leakage but do not create a new untouched dataset. Conditional bootstrap intervals omit overlapping-CV training variation, model-selection variation and patient clustering. Earlier graph/provenance/patient-linkage limitations remain. Additional marks remain a faculty judgment.
