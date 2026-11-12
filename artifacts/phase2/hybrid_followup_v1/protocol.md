# Follow-up protocol: hybrid improvement study

This is an explicitly exploratory follow-up after viewing the original Phase 2
results. Preserve that run and its negative hybrid finding. Do not overwrite its
protocol, models or results, and do not describe the follow-up as confirmation
on a newly untouched test set. No model in this follow-up may use historical T
or cutoff C records for fitting, architecture choice, calibration or selection.
Use S only (550 rows), with persisted common nested folds.

## Motivation and finite alternatives

The original hybrid appended one BN feature and did not improve the MLP. Two
mechanistically different corrections are tested, rather than an unlimited
search for a favorable score:

1. **BN residual MLP:** predict sigmoid(logit(q_BN) + r_theta(x)). The BN provides
   a structured baseline; a small network learns continuous corrections to its
   coarse bins and conditional independences. Train BN offsets by cross-fitting;
   zero-initialize the residual output so the initial model equals the BN.
2. **Calibrated BN–MLP stack:** train a penalized logistic meta-model on the two
   out-of-fold base logits. This learns whether the independently structured and
   continuous models contain complementary probability information and adjusts
   their confidence. An intercept-only calibrated BN and a calibrated MLP are
   essential controls: gains must not be attributed to hybridization when they
   come from calibration alone. This is standard stacking, not a new algorithm.

## Evaluation fixed before follow-up results

Outer CV: five stratified S folds, seed 144. Inner CV: three stratified folds,
seed 145. Neural seeds 42,52,62, all retained without seed selection. Repeat all
training, internal epoch selection, base cross-fitting and calibration within
outer training rows only. No global out-of-fold matrix may be reused across
outer folds. Meta-model training requires **fresh** inner cross-fitted base
predictions; training a base model on its own labels is prohibited.

Residual strengths: [0.25,0.5,1.0]; compare an equally budgeted MLP search over
weight decay [0.0001,0.001,0.01]. Calibrated stack and each calibrated single-base
control use logistic C [0.1,1,10]. All grid choices use inner mean Brier score,
then inner ROC-AUC, then grid order. Fixed pure BN is also reported.

Primary outcome: nested outer Brier (proper probability scoring aligns with the
brief's calibration requirement). Secondary outcomes: ROC-AUC, trapezoidal
PR-AUC, log loss, F1, sensitivity and specificity at **0.5 only**. Original F1-first
results remain unchanged and cannot be replaced. Report every candidate, seed,
inner score and outer prediction, including unfavorable findings. Report seed
means and paired row-bootstrap differences with explicit conditional-uncertainty
limits. A useful hybrid claim requires lower Brier than both single-base
calibrated controls and its own raw components; favorable discrimination and
seed consistency strengthen but do not prove generalization.

## Interpretation and scope

Residual-learning and stacking are established methods. Innovation is the
project-specific combination, leakage prevention and measured value, not an
algorithmic novelty claim. Prior evaluation exposure and many comparisons
preclude a confirmatory superiority claim. Bonus marks remain a faculty decision.
After evaluating this finite study, freeze settings; do not expand the search
based on outer scores. Save the selected models trained on S for inference only;
keep old C/T performance comparisons unchanged. Update project documentation to
separate original results from follow-up evidence. Keep all work on the Phase 2
branch, with actual commit times and no changes to main.

The original BN-feature MLP is retained as a fixed-setting benchmark on the new
outer folds. All original model artifacts remain immutable. Learned logistic
coefficients and residual-correction summaries are saved to make each hybrid's
behavior inspectable.
