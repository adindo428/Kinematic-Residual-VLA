from pathlib import Path
import sys,json,tarfile,hashlib
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'main8_palmetto_replicas_20261021'));import stage_and_submit as c
ROOT='/scratch/aparame/Dynamic_VLA/main8-h100-optimization-20261022'
SOURCE=HERE.parents[1]/'experiments/main8_h100_optimization_20261022'
c.HERE=HERE;c.ROOTS={'a100':ROOT}
def main():
 assert not (HERE/'submissions.json').exists(),'Recorded submissions; reconcile before retry'
 c.remote('test ! -e '+ROOT+' && mkdir -p '+ROOT+'/scripts '+ROOT+'/logs')
 with tarfile.open(HERE/'source.tar','w') as t:
  for p in SOURCE.iterdir():
   if p.is_file():t.add(p,arcname=p.name)
 c.remote('tar -xf - -C '+ROOT,(HERE/'source.tar').read_bytes())
 gate=ROOT+'/scripts/prepare.sbatch';gpu=ROOT+'/scripts/profile.sbatch'
 body=c.header('main8-h100-profile-prepare',ROOT,1)+f'''{c.PYTHON} -m unittest test_budget test_optimizer test_autocast
set +e
{c.PYTHON} train.py train --root {ROOT} --recipe pi05_baseline > production-forbidden.log 2>&1
status=$?
set -e
[[ $status -ne 0 ]]
{c.PYTHON} -c 'assert "diagnostic only" in open("production-forbidden.log").read()'
{c.PYTHON} prepare_diagnostic.py --root {ROOT}
'''
 gpu_body=c.header('main8-h100-runtime-profile',ROOT,1,'h100').replace('--constraint=gpu_h200','--constraint=gpu_h100').replace('--time=48:00:00','--time=04:00:00').replace('#SBATCH --requeue\n','').replace('#SBATCH --signal=B:USR1@1200\n','')
 gpu_body+=f'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled\n{c.PYTHON} sweep.py --root {ROOT}\n'
 specs={gate:body,gpu:gpu_body};preflight={}
 for file,text in specs.items():
  c.remote('cat > '+file,text.encode());result=c.remote('bash -n '+file+' && sbatch --test-only '+file+' 2>&1').decode()
  preflight[file]={'script_sha256':hashlib.sha256(text.encode()).hexdigest(),'scheduler_result':result}
 c.persist(HERE/'scheduler-preflights.json',preflight)
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('*.py')};c.persist(HERE/'source-manifest.json',{'files':hashes,'fingerprint':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()})
 cpu=c.submit('preparation',gate);c.submit('bounded_h100_profile',gpu,cpu)
if __name__=='__main__':main()
