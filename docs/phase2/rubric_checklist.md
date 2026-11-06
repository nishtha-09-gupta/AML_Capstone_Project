# Phase 2 rubric coverage and verification

This checklist distinguishes delivered evidence from outcomes that require team or faculty action.
All paths below are relative to the repository root.

| Rubric criterion | Delivered evidence | Status / qualification |
|---|---|---|
| Architecture Logic | `phase2_models.py`; report model section; presentation | Complete: two hidden layers, ReLU, sigmoid, BCE, explicit parameter counts and training-only early stopping |
| Hybrid Innovation | Cross-fitted BN log-odds input; NB-assisted ablation; `literature_review.md` | Implemented and evaluated. Feature stacking is not claimed as a new algorithm; consistent added value was not demonstrated, so bonus marks cannot be guaranteed |
| Ablation Studies | `artifacts/phase2/full_run/cv_by_seed.csv`, `cv_summary.csv`, `evaluation_summary.csv` | Complete: plain MLP, BN-assisted MLP, NB-assisted MLP, no dropout, no weight decay; common folds, three seeds |
| Deep Learning Literature Review | `docs/phase2/literature_review.md`; report references and slide notes | Complete: primary sources and applicability/limits documented |
| DL Dataset & Regularization | `protocol.md`, `phase2_models.py`, training histories and split audits | Complete: target, predictors, missingness, scaling, encoding, dropout, weight decay and early stopping; patient independence cannot be proven without linkage |
| Technical Validation | `phase2_verify.py`, `verification.json`, `tests/test_phase2_*.py`, `.github/workflows/phase2.yml` | Full run PASS for 35 saved models; 60 local tests and 39 subtests passed excluding unchanged Phase 1 Streamlit UI; dedicated GitHub Phase 2 checks passed |
| Theoretical Rigor | IEEE report equations; probability factorization and smoothing; cross-fitting and uncertainty limitations | Complete for implemented comparison; causal validity and clinical utility are not established |
| GitHub Repository & Code Quality | Dedicated Phase 2 modules, dependency file, lock, source snapshots, hashes, commands and CI | Complete on Phase 2 branch; main intentionally unchanged |
| Project Presentation | `docs/phase2/phase2_presentation.pptx` | Editable deck with result table, charts, source notes and limitations; team must rehearse and confirm authorship |
| Viva Voce | `docs/phase2/viva_guide.md` | Preparation delivered; oral evaluation must be performed by the team |
| Project Report (LaTeX/IEEE) | `docs/phase2/phase2_report.tex` and `phase2_report.pdf` | IEEEtran source compiled to PDF; actual names/contributions and faculty-specific submission details must be confirmed |

## Evidence inventory

- Primary S-only CV: five common folds; three neural seeds; all configurations retained.
- Primary evaluation: S=550, C=184, historical T=184; no S+C refit.
- BN and NB: pgmpy estimation; exact inference checked against variable elimination; exported numeric bins, states and CPTs.
- Frozen models: 19 pooled models plus 16 independently fitted hospital models = 35.
- CV models are evaluated and audited per fold; final fitted models are the saved inference models.
- Probability diagnostics: ROC-AUC, trapezoidal PR-AUC, average precision, Brier, log loss, reliability bins and counts.
- Decision diagnostics: default and C-selected cutoffs, confusion matrices, F1, sensitivity, specificity, accuracy and precision.
- Uncertainty: 2,000 paired fixed-prediction bootstrap resamples for hybrid-minus-MLP differences, separately for all three seeds.
- Secondary analyses: four D-only hospital holdouts and Cleveland-only D CV.
- Data/source integrity: original CSV and historical AML results preserved; source hashes, code copies, source IDs and environment versions recorded.

## Final interpretation

The selected family is MLP. For seed 42, retrospective ROC-AUC is 0.9071,
PR-AUC 0.9192, F1 0.8411, and Brier 0.1236. Hybrid values are 0.9020,
0.9172, 0.8358, and 0.1252. The hybrid trades sensitivity for specificity
at its own C-selected cutoff. Paired intervals include zero. A completed hybrid
investigation is present; demonstrated overall improvement is not.

Remaining external requirements: faculty acceptance of the provisional graph,
merged-CSV provenance/patient linkage clarification, confirmed team authorship,
rehearsal, viva performance and any institution-specific report formatting rules.
