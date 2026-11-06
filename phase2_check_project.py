"""Audit expected Phase 2 deliverables and immutable full-run evidence."""
import hashlib
import json
from pathlib import Path
from phase2_verify import verify
from experiment import write_json

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'artifacts/phase2/full_run'
REQUIRED=[
    'phase2_baselines.py','phase2_models.py','phase2_train.py','phase2_verify.py',
    'phase2_summarize.py','phase2_predict.py','requirements-phase2.txt',
    '.github/workflows/phase2.yml','tests/test_phase2_baselines.py',
    'tests/test_phase2_models.py','tests/test_phase2_workflow.py',
    'docs/phase2/protocol.md','docs/phase2/literature_review.md',
    'docs/phase2/README.md','docs/phase2/rubric_checklist.md',
    'docs/phase2/viva_guide.md','docs/phase2/phase2_report.tex',
    'docs/phase2/phase2_report.pdf','docs/phase2/phase2_presentation.pptx',
    'artifacts/phase2/full_run/RESULTS.md','artifacts/phase2/full_run/cv_by_seed.csv',
    'artifacts/phase2/full_run/evaluation_summary.csv',
    'artifacts/phase2/full_run/hospital_summary.csv',
    'artifacts/phase2/full_run/cleveland_summary.csv',
    'artifacts/phase2/full_run/paired_hybrid_minus_mlp.json',
    'artifacts/phase2/full_run/figures/discrimination.png',
    'artifacts/phase2/full_run/figures/calibration_ablations.png',
    'artifacts/phase2/full_run/figures/learning_curves.png',
]


def check():
    missing=[p for p in REQUIRED if not (ROOT/p).is_file() or not (ROOT/p).stat().st_size]
    if missing:
        raise ValueError('Missing deliverables: '+', '.join(missing))
    evidence=RUN/'evidence_sha256.json'
    if evidence.exists():
        for relative,digest in json.loads(evidence.read_text()).items():
            path=RUN/relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError('Changed run evidence: '+relative)
    verified=verify(RUN)
    result=dict(status='PASS',required_deliverables_checked=len(REQUIRED),
                saved_models_checked=verified['saved_models_checked'],
                hybrid_implemented=True,hybrid_improvement_demonstrated=False,
                human_requirements=['Confirm author names and contributions',
                    'Rehearse and perform presentation and viva',
                    'Faculty review of provisional graph and submission requirements'])
    write_json(ROOT/'docs/phase2/completion_check.json',result)
    return result


if __name__=='__main__':
    print(json.dumps(check(),indent=2))
