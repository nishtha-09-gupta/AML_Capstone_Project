# Phase 2: running and reviewing the completed study

Use a separate Python environment; Phase 1's frozen models retain their own recorded dependencies.

```bash
python3 -m venv .venv-phase2
source .venv-phase2/bin/activate
python -m pip install -r requirements-phase2.txt
python -m pytest tests/test_phase2_baselines.py tests/test_phase2_models.py tests/test_phase2_workflow.py -q
python phase2_verify.py artifacts/phase2/full_run
```

Reproduce the full experiment in a **new** directory (never overwrite prior runs):

```bash
python phase2_train.py --output artifacts/phase2/my_run
python phase2_verify.py artifacts/phase2/my_run
python phase2_summarize.py artifacts/phase2/my_run
```

A short workflow check, unsuitable for performance conclusions:

```bash
python phase2_train.py --output /tmp/phase2-check --smoke
python phase2_verify.py /tmp/phase2-check
```

Batch inference accepts CSV rows with the eleven original eligible features:

```bash
python phase2_predict.py artifacts/phase2/full_run profiles.csv --model 'MLP__42' --output predictions.csv
```

Only load trusted joblib artifacts. Inference verifies the saved model hash,
uses the C-selected cutoff, and never trains. Scores describe recorded disease
presence in the study context, not medical advice. The existing Phase 1 web app
continues to show its Phase 1 frozen models; Phase 2 is reviewed through these
commands and its saved evidence.

## Reading order

1. `protocol.md`: fixed settings, partitions, hybrid rationale and restrictions.
2. `literature_review.md`: sources and design choices.
3. `../../artifacts/phase2/full_run/RESULTS.md`: measured results and limitations.
4. `rubric_checklist.md`: evidence for every evaluation criterion.
5. `phase2_report.tex` / `phase2_report.pdf`: IEEE-format report.
6. `phase2_presentation.pptx`: editable presentation; `viva_guide.md`: preparation.

The full run stores code snapshots, protocol, package versions, exact source IDs,
common CV folds, every seed's predictions, training histories and leakage audits,
BN bins/CPTs, final saved models/hashes, thresholds, metrics and verification.
CV evaluates fixed configurations. Phase 1's nested tuning scores are contextual
and have a different search budget; do not present them as identical procedures.

## Latest follow-up

Read [the hybrid follow-up guide](hybrid_followup_guide.md) for the improved
calibrated stack and its S-only nested-CV evidence. Use the report and presentation
with `with_followup` in their filenames for the combined account. The original
files remain available as the original experiment's record.
