# Final interpretation of the preserved hybrid experiments

| Experiment | Preserved finding |
|---|---|
| Original BN-feature MLP | Did not beat standalone MLP on original mean CV F1 |
| BN residual MLP | Follow-up mean Brier 0.14775, worse than tuned MLP (0.14168) and calibrated BN (0.14038) |
| Calibrated BN–MLP stack | Lowest observed follow-up Brier, 0.13913; calibrated BN is 0.14038 |

The stack-minus-calibrated-BN Brier difference is -0.00126, with conditional
95% interval [-0.00405, 0.00162]. This includes zero: the reported interval does
not establish statistically significant superiority. It is neither an external
validation result nor a formal post-selection significance test.

The stack fits an intercept and penalized weights to fresh cross-fitted BN and
MLP log-odds. It is not a fixed arithmetic average of probabilities. The learned
weights and intercept can adjust confidence and combine complementary evidence,
but this is established stacking, not a novel algorithm.

**OOF-to-full-fit score-distribution shift:** the meta-model learns from base
predictions made without the corresponding training rows; inference uses base
models refitted on the full permitted training subset. Their score distributions
may differ. Cross-fitting prevents direct own-label leakage but does not remove
this sample-size shift.

**Post-selection uncertainty:** multiple follow-up alternatives were compared,
and the best was identified using the same outer-CV results later used to
describe it. Inner tuning protects each outer fold from its own hyperparameter
selection; it does not make the subsequently selected family's outer score an
independent confirmation. Bootstrap intervals condition on saved predictions
and omit this complete research-selection process and patient-cluster effects.

**Residual negative result:** the model computes sigmoid(logit(q_BN) + strength
× residual_MLP(x)). A zero-initialized residual output starts at the BN baseline;
it does not guarantee better generalization. Its unfavorable results remain
part of the evidence.

All old T results remain retrospective. The follow-up reused S after earlier
findings; C and T were excluded from follow-up fitting/scoring. No untouched
external cohort or verified patient-level split establishes external generalization.

Raw evidence stays in `artifacts/phase2/full_run` and
`artifacts/phase2/hybrid_followup_v1`. This interpretation supplements those
immutable records without changing predictions, scores, model hashes or protocols.
