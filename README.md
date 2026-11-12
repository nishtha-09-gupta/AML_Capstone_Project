# AML Capstone Project — Heart Disease Classification

A reproducible Phase 1 machine learning project using the multi-center UCI Heart Disease dataset. The workflow covers data validation, exploratory data analysis, preprocessing, baseline classifiers, nested cross-validation, hyperparameter tuning, a provisional Bayesian Network, cutoff selection, and retrospective evaluation.

**Repository:** [AML_Capstone_Project](https://github.com/nishtha-09-gupta/AML_Capstone_Project)

> This is an academic classification project. It predicts recorded disease presence (`num > 0`), not future cardiovascular risk. The models are not clinically validated. Phase 1 and Phase 2 implementation and evidence are included in this repository; see the sections below.

## Phase 2 evaluation: start here

- [Overview and run instructions](docs/phase2/README.md)
- [Original protocol](docs/phase2/protocol.md) and [follow-up protocol](docs/phase2/hybrid_improvement_protocol.md)
- [Architectures and rationale](docs/phase2/literature_review.md) · [MLP implementation](phase2_models.py) · [hybrids](phase2_hybrids.py)
- [Results interpretation and limitations](docs/phase2/results_interpretation.md) · [saved follow-up results](artifacts/phase2/hybrid_followup_v1/RESULTS.md)
- [Current IEEE report PDF](docs/phase2/phase2_report_with_followup.pdf) and [LaTeX source](docs/phase2/phase2_report_with_followup.tex)
- [Current editable presentation](docs/phase2/phase2_presentation_with_followup.pptx) and [presentation PDF](docs/phase2/phase2_presentation_with_followup.pdf)
- [Verification and audit findings](docs/phase2/submission_audit.md) · [verified literature](docs/phase2/literature_review.md)
- [Team and proposed responsibility allocation](docs/phase2/contributions.md)
- [Dataset provenance, missingness, exclusions, and patient-linkage limitations](docs/dataset_provenance_and_limitations.md)

The stack's modest Brier gain is exploratory. The reported conditional interval
includes zero, so it does not establish statistical superiority. Preserve the
original negative hybrid findings and the follow-up selection caveats.

## Dataset

The supplied CSV contains **920 records and 16 columns** from Cleveland, Hungary, Switzerland, and VA Long Beach. Removing two exact duplicate records while ignoring the arbitrary row ID leaves **918 records**.

See [dataset provenance and limitations](docs/dataset_provenance_and_limitations.md) for the source record, reproducible feature and hospital missingness counts, exclusion rationale, and patient-linkage caveats.

| Binary target | Definition | Records |
| --- | --- | ---: |
| Negative | `num == 0` | 410 |
| Positive | `num > 0` | 508 |

The original target values range from 0 to 4. This project converts them into a binary disease-presence label and does not model severity. Deduplicated records are not necessarily verified unique patients; patient linkage is unavailable.

### Predictors and cleaning

| Type | Included variables |
| --- | --- |
| Numeric | `age`, `trestbps`, `chol`, `thalch`, `oldpeak` |
| Categorical | `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope` |

Excluded predictors are `id`, `dataset`, `ca`, `thal`, and the target `num`. Hospital labels are used only for grouping and evaluation. `ca` and `thal` are excluded because of extensive missingness.

Cleaning and preprocessing include:

- Schema, category, range, source-ID, and target validation.
- Exact-record deduplication, retaining the first occurrence.
- Conversion of zero blood-pressure and cholesterol entries to missing values.
- Numeric median imputation, missing-value indicators, and standardization for conventional classifiers.
- Categorical mode imputation and one-hot encoding, with unknown categories tolerated at prediction time.
- Training-only fitting of every learned transformation inside each cross-validation fit.

The Bayesian Network uses its own training-fitted imputation, discretization, and conditional probability tables.

## Implemented Models

| Model | Role |
| --- | --- |
| Logistic Regression | Linear classification baseline and tuned candidate |
| Random Forest | Ensemble baseline and tuned candidate |
| Histogram Gradient Boosting | Boosted-tree candidate |
| XGBoost | Gradient-boosted-tree candidate |
| Bayesian Network | Provisional discrete probabilistic classifier |

The original baseline workflow evaluates Logistic Regression and Random Forest through five-fold cross-validation. The extended workflow compares all five families using nested cross-validation.

The Bayesian Network specifies disease-to-feature dependencies and an additional `oldpeak` to `slope` dependency. It estimates smoothed conditional probability tables with `alpha=1` and supports inference with missing evidence. Its topology and independence assumptions remain provisional; arrow directions do not establish medical causation. Its single smoothing setting and unequal search budgets across families limit comparisons.

## Validation Workflow

```text
918 retained records
├── Development D: 734 records
│   ├── Selection S: 550 records
│   │   ├── Five outer folds for family comparison
│   │   ├── Three inner folds for hyperparameter selection
│   │   └── Final three-fold search and fitting on S
│   └── Cutoff C: 184 records
│       └── Select each frozen model's decision threshold
└── Retrospective evaluation T: 184 records
    └── Evaluate frozen models and decisions
```

The historical stratified split uses seed 42. The S/C split uses seed 43; outer, inner, and final-search cross-validation use seeds 44, 45, and 46 respectively.

Family selection uses the highest mean outer-fold F1 at probability cutoff 0.50. ROC-AUC breaks exact F1 ties, followed by the recorded fixed family order. Inner searches rank configurations by mean F1, then ROC-AUC, then canonical parameter JSON.

Each frozen model receives a separate cutoff selected on C by maximizing F1 over 0.00–1.00 in steps of 0.01. Ties choose the point nearest 0.50, then the smaller point. Predictions use `probability >= cutoff`.

Models are fitted on S only, with **no S+C refit**. Neither cutoff scores nor retrospective evaluation scores can replace the family selected on S. The winning family's CV score still has selection optimism.

**The historical holdout was previously inspected in the source project. Its results are retrospective, not a newly untouched test.**

## Results

Recorded run: `artifacts/runs/20261025T151903Z_ea0f799c/`.

Random Forest was selected by nested-CV F1. The remaining metrics below use each model's C-selected cutoff on the **184-record retrospective evaluation partition**.

| Model | Nested CV F1 | Cutoff | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8231 | 0.51 | 0.8315 | 0.8257 | 0.8824 | 0.8531 | 0.9125 |
| Random Forest | **0.8358** | 0.38 | 0.8043 | 0.7538 | 0.9608 | 0.8448 | 0.9029 |
| Histogram Gradient Boosting | 0.8104 | 0.37 | 0.8152 | 0.7698 | 0.9510 | 0.8509 | 0.8945 |
| XGBoost | 0.8308 | 0.34 | 0.8043 | 0.7463 | 0.9804 | 0.8475 | 0.9039 |
| Bayesian Network | 0.8179 | 0.54 | 0.7880 | 0.8058 | 0.8137 | 0.8098 | 0.8765 |

Logistic Regression's higher retrospective F1 and ROC-AUC do not change the predefined model-selection decision.

### Selected-model confusion matrix

Random Forest at cutoff **0.38**:

| Actual / Predicted | Negative | Positive |
| --- | ---: | ---: |
| Negative | TN: 50 | FP: 32 |
| Positive | FN: 4 | TP: 98 |

Specificity is **0.6098**. The cutoff is a statistical operating point selected on development data, not a medically established threshold.

### Hospital holdout checks

Each check uses development records only, holds out one hospital, and repeats family selection, model fitting, and cutoff selection independently using the remaining hospitals. Historical T records and pooled learned state are excluded.

| Held-out hospital | Locally selected model | Evaluation records | F1 | Always-positive F1 | Specificity | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cleveland | Logistic Regression | 248 | 0.7951 | 0.6260 | 0.7481 | 0.8612 |
| Hungary | Random Forest | 229 | 0.6948 | 0.5367 | 0.6207 | 0.8712 |
| Switzerland | Random Forest | 96 | 0.8712 | 0.9565 | 0.5000 | 0.7472 |
| VA Long Beach | Logistic Regression | 161 | 0.8561 | 0.8582 | 0.0500 | 0.7079 |

Switzerland and VA Long Beach do not outperform their same-row always-positive reference on F1. These findings and the low VA specificity constrain claims about transfer across hospitals.

## Repository Structure

```text
AML_Capstone_Project/
├── heart_disease_uci.csv       # Source dataset
├── data_loader.py             # Ingestion, validation and cleaning
├── eda.py                     # EDA and descriptive charts
├── modeling.py                # Original baseline preprocessing/models
├── train_baselines.py         # Two-model five-fold baseline workflow
├── advanced_modeling.py       # Extended modeling data-contract helpers
├── bayesian_network.py        # Provisional discrete BN
├── train.py                   # Nested selection, tuning and evaluation
├── experiment.py              # Complete experiment and artifact persistence
├── verify_run.py              # Saved-model and result verification
├── summarize_run.py           # Result tables and diagnostic charts
├── requirements.txt           # Runtime and test dependencies
├── config/
│   ├── phase1_protocol.json   # Protocol, grids and seeds
│   └── bn_graph.json          # BN structure and assumptions
├── docs/
│   ├── validation_protocol.md
│   └── bn_design.md
├── notebooks/
│   └── eda_walkthrough.ipynb
├── tests/                     # Data, baseline, BN and isolation tests
└── artifacts/
    ├── data_audit.json        # Descriptive audit
    └── runs/
        └── 20261025T151903Z_ea0f799c/
            ├── RESULTS.md
            ├── effective_protocol.json
            ├── manifest.json
            ├── verification.json
            ├── requirements-freeze.txt
            ├── roc_curves.png
            ├── selected_confusion_matrix.png
            ├── pooled/
            └── hospital_*/
```

Each experiment folder contains fitted models, exact partitions, nested out-of-fold predictions, complete search traces, cutoff curves/predictions, evaluation predictions, and metrics.

## Setup

```bash
git clone https://github.com/nishtha-09-gupta/AML_Capstone_Project.git
cd AML_Capstone_Project
python3 -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Saved model loading should use the package versions recorded in the run's `requirements-freeze.txt` and `environment.json`. The general requirements file allows version ranges and does not guarantee cross-version model compatibility.

On macOS, XGBoost requires an OpenMP runtime. If importing XGBoost reports a missing `libomp.dylib`, install a compatible runtime or supply its directory through `DYLD_LIBRARY_PATH`. The recorded run used the runtime bundled with its scikit-learn wheel.

## Run the Project

### Run tests

```bash
python -m pytest tests -q
```

Recorded verification: **48 tests passed**, with **39 subtests passed**.

### Generate EDA artifacts

```bash
python eda.py
```

This regenerates the descriptive audit and EDA outputs in `artifacts/`.

### Run the original baselines

```bash
python train_baselines.py --output artifacts/my_baseline_run
```

Choose a new output directory. The baseline runner does not overwrite existing results.

### Run the complete ML experiment

```bash
python train.py --run-experiment --allow-provisional-bn
```

Both flags are required. The run performs approximately **2,940 fits** across the pooled experiment and four hospital experiments, using one worker. It creates a new timestamped folder under `artifacts/runs/` and preserves earlier runs. Real-data training does not start on module import.

The root protocol retains historical draft questions. Each run stores an effective protocol declaring its provisional exploratory settings before fitting.

### Verify the recorded models and results

```bash
python verify_run.py artifacts/runs/20261025T151903Z_ea0f799c
```

The recorded verification reports **PASS for all 25 saved models**. Checks include model hashes, reproduced probabilities, metric reconstruction, cutoff selection, nested fold scores, family selection, and partition/hospital isolation. Verification writes `verification.json` in the selected run folder.

### Regenerate the result summary and charts

```bash
python summarize_run.py artifacts/runs/20261025T151903Z_ea0f799c
```

This writes `RESULTS.md`, ROC curves, and the selected-model confusion matrix from saved predictions, without retraining. Replace the run path in these commands to inspect a newly generated experiment.

## Reproducibility and Limitations

- Exact source IDs, split/fold memberships, seeds, code snapshots, source/model hashes, environment details, and dependency versions are saved with the run.
- Evaluation includes accuracy, precision, recall, specificity, F1, ROC-AUC, average precision, Brier score, log loss, and confusion matrices.
- Reliability bins and sex/age subgroup summaries are descriptive and do not establish calibration or fairness.
- The 2,000 paired row-bootstrap resamples condition on the fitted models and cutoffs. They do not capture training, family-selection, or patient-cluster uncertainty.
- The BN's structure and independence assumptions still need domain review. No causal interpretation is established.
- Dataset acquisition/merge history and patient linkage remain incomplete. Exact-record deduplication does not establish patient independence.
- No prospective clinical evaluation or clinical utility assessment is complete.
- The extended ML code was adapted from the team's existing HeartLens implementation. Individual contributions should be documented accurately.

## Data Attribution

The underlying data is the **UCI Heart Disease dataset**, credited to Andras Janosi, William Steinbrunn, Matthias Pfisterer, and Robert Detrano.

Dataset DOI: [10.24432/C52P4X](https://doi.org/10.24432/C52P4X).

Attribution to UCI does not by itself establish the exact acquisition and transformation history of the merged CSV supplied with this project.

## Phase 2 — Deep Learning Evaluation

The completed Phase 2 compares a pgmpy Bayesian Network, Naive Bayes, a small
regularized PyTorch MLP, and a BN-assisted MLP with cross-fitted probabilities.
It includes regularization and NB-input ablations, three neural seeds, calibration,
D-only hospital holdouts, Cleveland-only CV, saved models and result verification.

- [Setup, inference and reproduction](docs/phase2/README.md)
- [Results and limitations](artifacts/phase2/full_run/RESULTS.md)
- [Rubric evidence checklist](docs/phase2/rubric_checklist.md)
- [IEEE report](docs/phase2/phase2_report.pdf) and [LaTeX source](docs/phase2/phase2_report.tex)
- [Presentation](docs/phase2/phase2_presentation.pptx) and [viva guide](docs/phase2/viva_guide.md)

The standalone MLP was selected by the predeclared S-only criterion. The functioning
hybrid did not demonstrate added value over the MLP. All historical T results remain
retrospective; no clinical benefit or bonus-mark outcome is claimed.

### Hybrid improvement follow-up

A separate exploratory nested-CV study on S only adds residual BN correction and
a calibrated BN–MLP stack, with calibrated single-base controls. The stack has
the lowest mean Brier among all seven tested families, improving by 1.80% against
tuned MLP and 0.89% against calibrated BN. Every seed mean improves on both
controls, while conditional intervals include zero. Original T scores and
negative feature-hybrid findings remain preserved.

- [Follow-up methods, evidence and run instructions](docs/phase2/hybrid_followup_guide.md)
- [All follow-up results](artifacts/phase2/hybrid_followup_v1/RESULTS.md)
- [Updated report](docs/phase2/phase2_report_with_followup.pdf) and [LaTeX](docs/phase2/phase2_report_with_followup.tex)
- [Updated presentation](docs/phase2/phase2_presentation_with_followup.pptx)

The observed gain supports a stronger technical hybrid case; it does not establish
external superiority, clinical benefit or guaranteed bonus marks.
