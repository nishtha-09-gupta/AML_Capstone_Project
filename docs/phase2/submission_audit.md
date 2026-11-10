# Phase 2 submission improvement audit

## Completed changes

- Inserted the five user-supplied author names consistently in the current report and slides. Institution/team/section come from the committed Phase 1 report.
- Added `team.json` and a roughly balanced **proposed** responsibility allocation in `contributions.md`, with Aanya leading implementation and integration. Actual completed contributions have not been independently confirmed and are not fabricated.
- Clarified modest exploratory Brier improvement, the zero-spanning conditional interval, lack of established statistical superiority, OOF-to-full-fit score shift, and post-selection uncertainty.
- Retained original feature-hybrid and residual negative results; explained zero initialization and why it does not guarantee generalization. Distinguished learned log-odds stacking from fixed probability averaging.
- Corrected the published Grinsztajn title and added verified medical-hybrid, cardiovascular-neural, stacking and medical-calibration references, with applicability distinctions.
- Added an early Phase 2 README entry point and a companion interpretation document. Frozen experimental narratives remain unchanged.
- Compiled the updated IEEEtran report, exported the current 15-slide presentation to PDF, and inspected every report page and every exported slide. Checked labels, margins, tables, scores and limitations. The three original chart workbooks were preserved byte-for-byte during editable-deck export. No native PowerPoint inspection is claimed.

## Main-to-Phase-2 comparison

Fetched `origin/main` and `origin/phase2-deep-learning` without switching, merging,
rebasing or resetting. The PDF's suggested Desktop audit checkout was absent;
the existing project checkout matched the verified remote Phase 2 branch.

At the audited base, main is `da62bdefc16004468de5d22e3d21dab815ea537c` and Phase 2
is `88dc99852ff59679a6e1d9ec00c2853a9d9ccf8d`. Relative to main, the only modified
pre-existing tracked files were README.md and .gitignore. There were 286 added
files and no deleted files. The additions are Phase 2 modules, tests, workflow,
documentation and experiment evidence. Existing Phase 1 application/model code,
source data and historical artifacts remain unchanged. This submission update
adds documentation, current deliverable revisions and a read-only verification
wrapper, with no architecture or training changes.

## Verification

```bash
git fetch origin main:refs/remotes/origin/main phase2-deep-learning:refs/remotes/origin/phase2-deep-learning
git branch -avv
git diff --stat origin/main...phase2-deep-learning
git diff --name-status origin/main...phase2-deep-learning
python tools/verify_submission.py --output /tmp/phase2-submission-audit
```

Use the Phase 2 environment. Choose a new output directory; the wrapper refuses
to overwrite an existing one. It runs `phase2_verify.py` and
`phase2_hybrid_verify.py` against temporary copies, then runs:

```bash
python -m pytest tests --ignore=tests/test_streamlit_app.py -q
```

Fresh result: **66 tests passed, 39 subtests passed**. Verification passed for
35 original saved models, 105 follow-up outer-fold models and 12 final models.
Hashes before/after confirmed **365 protected source/evidence files unchanged**.
Machine-readable details are in `submission_verification.json`. The unchanged
Phase 1 Streamlit UI was excluded; no new native UI validation is claimed.
Only small existing unit-test fits ran; no real-data performance experiment was
retrained or tuned to improve its score.

## Current deliverables

- `phase2_report_with_followup.tex` / `.pdf`: current report.
- `phase2_presentation_with_followup.pptx` / `.pdf`: current presentation, 15 slides.
- `literature_review.md`, `results_interpretation.md`, `viva_guide.md`: supporting material.
- `contributions.md`: proposed allocation for team confirmation.

Earlier Phase 1 and original Phase 2 reports/decks remain historical records;
use the `with_followup` files above for this submission.

## Remaining human confirmation

The team must confirm actual individual contributions before converting the
proposed allocation into completed-work statements. This audit does not invent
those facts. Faculty-specific acceptance of the provisional graph, patient-linkage
limitations, and presentation/viva performance remain external requirements.
No statistically significant superiority, external validation, novel algorithm
or guaranteed additional marks is asserted.
