# Verification record

The full experiment completed with five S folds and three neural seeds; it is not
a smoke run. `artifacts/phase2/full_run/verification.json` reports PASS and 35
saved models checked. Model probabilities, metrics, cutoffs, selection decisions,
CV coverage and hospital isolation were reconstructed. Fast BN inference also
agrees with pgmpy variable elimination and the original AML BN in focused tests.

Local command:

```bash
python -m pytest tests --ignore=tests/test_streamlit_app.py -q
```

Result: 60 tests passed, 39 subtests passed. The unchanged Phase 1 Streamlit UI
was excluded because it uses its separately recorded runtime and frozen models;
this record does not claim that the UI was revalidated in the Phase 2 environment.
The batch inference path was additionally checked on source feature rows against
saved evaluation predictions.

GitHub's dedicated Phase 2 workflow passed the implementation commit, including
unit checks and a full smoke-workflow verification. Smoke scores are never used
as project performance evidence.

The IEEEtran source compiled to PDF and every page was visually inspected.
The editable presentation was rendered slide by slide and checked for text fit,
source/result consistency, native tables/charts and package integrity. These
checks do not claim that the deck was opened in Microsoft PowerPoint.
