import json,shutil,subprocess
from pathlib import Path
work=Path('/scratch/aparame/Dynamic_VLA');setup=work/'setup-video-clean-20261012'
oldsource=work/'third_party/dom100-ddp-bnfix-20261011';source=work/'third_party/dom100-ddp-video-clean-20261012'
old=work/'dom100-ddp-bnfix-20261011';family=work/'dom100-ddp-video-clean-20261012'
assert not source.exists()
shutil.copytree(str(oldsource),str(source),ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','.git','*.out','*.err','*.pyc'))
subprocess.check_call(['python3',str(setup/'patch_source.py'),str(source)])
report=json.loads(subprocess.check_output(['python3',str(setup/'freeze_clean_family.py'),str(old),str(family),str(source)],universal_newlines=True))
for name in ('dom100_distributed.sbatch','krv100_a100_8gpu.sbatch','krv100_h200_8gpu.sbatch'):
 f=source/'slurm'/name;t=f.read_text();t=t.replace('set -euo pipefail\n', 'set -euo pipefail\nexport DVLA_PROJECT_DIR='+str(source)+'\nexport DOM100_ROOT='+str(family)+'\n',1)
 if name=='dom100_distributed.sbatch':t=t.replace('exec bash scripts/run_dom100_distributed.sh','exec bash scripts/run_dom100_allocation_candidate.sh')
 f.write_text(t)
report['previous_jobs']=['17017544','17091380','17091381']
report['stage']='prepared_no_submission_yet'
(setup/'deployment.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
