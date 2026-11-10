"""Verify copied evidence and unchanged baseline files without overwriting saved runs."""
from pathlib import Path
import argparse,sys
import hashlib,json,subprocess,tempfile,shutil
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',required=True,help='New directory for audit logs and JSON')
audit=Path(parser.parse_args().output).resolve();audit.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[1]
def snapshot():
 files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'artifacts').rglob('*') if p.is_file()}
 files['heart_disease_uci.csv']=hashlib.sha256((root/'heart_disease_uci.csv').read_bytes()).hexdigest()
 return files
before=snapshot()
with tempfile.TemporaryDirectory(dir=audit) as temporary:
 for dirname,script in [('full_run','phase2_verify.py'),('hybrid_followup_v1','phase2_hybrid_verify.py')]:
  target=Path(temporary)/dirname;shutil.copytree(root/'artifacts/phase2'/dirname,target)
  result=subprocess.run([sys.executable,script,str(target)],cwd=root,capture_output=True,text=True)
  (audit/(dirname+'-verification.log')).write_text(result.stdout+result.stderr)
  if result.returncode:raise RuntimeError(dirname+' verification failed')
  shutil.copy2(target/'verification.json',audit/(dirname+'-verification.json'))
 result=subprocess.run([sys.executable,'-m','pytest','tests','--ignore=tests/test_streamlit_app.py','-q'],cwd=root,capture_output=True,text=True)
 (audit/'tests.log').write_text(result.stdout+result.stderr)
 if result.returncode:raise RuntimeError('Tests failed')
after=snapshot();assert before==after,'Committed evidence changed'
modified=subprocess.check_output(['git','diff','--name-status','origin/main...HEAD'],cwd=root,text=True).splitlines()
mainfiles=subprocess.check_output(['git','ls-tree','-r','--name-only','origin/main'],cwd=root,text=True).splitlines()
changed=[n for n in mainfiles if subprocess.run(['git','diff','--quiet','origin/main','HEAD','--',n],cwd=root).returncode]
record=dict(main_commit=subprocess.check_output(['git','rev-parse','origin/main'],cwd=root,text=True).strip(),phase2_commit_before_submission_edits=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 changed_existing_main_files=changed,added_files=sum(x.startswith('A\t') for x in modified),deleted_files=[x[2:] for x in modified if x.startswith('D\t')],
 protected_files_checked=len(before),protected_sha256=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),protected_files_unchanged=True,
 full_run_verification=json.load(open(audit/'full_run-verification.json')),followup_verification=json.load(open(audit/'hybrid_followup_v1-verification.json')),tests=result.stdout.strip())
(audit/'submission_verification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
