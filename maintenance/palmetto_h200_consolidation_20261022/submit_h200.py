from pathlib import Path
import sys,json,tarfile,hashlib,shlex
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'main8_palmetto_replicas_20261021'))
import stage_and_submit as common
SOURCE=HERE.parents[1]/'experiments/main8_h200_consolidated_20261022'
ROOT='/scratch/aparame/Dynamic_VLA/main8-h200-only-20261022'
common.HERE=HERE;common.ROOTS={'a100':ROOT}
PYTHON=common.PYTHON
NAMES=('smolvla_baseline','smolvla_kr','domv100')

def main():
 assert not (HERE/'submissions.json').exists(),'Recorded submission; reconcile rather than retry'
 queue=common.remote('squeue -u aparame -h -o "%i|%j|%T|%R"').decode();assert 'main8-h200-only-' not in queue
 (HERE/'queue-before-h200.txt').write_text(queue)
 common.remote('test ! -e '+ROOT+' && mkdir -p '+ROOT+'/scripts '+ROOT+'/logs')
 with tarfile.open(HERE/'source.tar','w') as t:
  for p in SOURCE.iterdir():
   if p.is_file():t.add(p,arcname=p.name)
 common.remote('tar -xf - -C '+ROOT,(HERE/'source.tar').read_bytes())
 common.remote('cat > '+ROOT+'/replica-config.json',json.dumps({'mode':'canonical_h200_continuation','models':NAMES,'authorization':'Cancel all A100/slower Palmetto production and continue H200 from latest saved states;2026-10-22'}).encode())
 files={}
 gate=ROOT+'/scripts/prepare.sbatch'
 files[gate]=common.header('main8-h200-only-prepare',ROOT,1)+f'{PYTHON} -m unittest test_budget test_placements test_optimizer test_autocast\n{PYTHON} prepare_h200.py --root {ROOT}\n'
 for name in NAMES:
  python='/home/aparame/.conda/envs/dynamicvla/bin/python' if name=='domv100' else PYTHON
  body=common.header('main8-h200-only-'+name,ROOT,1,'h200')
  if name=='domv100':body+='export PYTHONPATH="$PWD:/scratch/aparame/Dynamic_VLA/main8-20k-10epochs-20261021-r2/delta-domv-f991015a"\n'
  body+=f'''export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_CONFIG_DIR={ROOT}/wandb-config
mkdir -p "$WANDB_CONFIG_DIR"
secret=/home/aparame/.config/kinematic-residual-vla/secrets.env
if [[ -f "$secret" && ! -L "$secret" && $(stat -c %a "$secret") == 600 ]]; then
 set -a
 source "$secret"
 set +a
fi
runner=
forward_stop() {{ if [[ -n "$runner" ]]; then kill -USR1 "$runner" || true; fi; }}
trap forward_stop USR1 TERM
{python} allocation.py --root {ROOT} --name {name} --placement h200 &
runner=$!
set +e
while true; do
 wait "$runner"; status=$?
 if ! kill -0 "$runner" 2>/dev/null; then break; fi
done
set -e
if [[ $status -eq 75 ]]; then scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
exit "$status"
'''
  files[ROOT+'/scripts/'+name+'.sbatch']=body
 preflight={}
 for path,body in files.items():
  common.remote('cat > '+shlex.quote(path),body.encode())
  result=common.remote('bash -n '+shlex.quote(path)+' && sbatch --test-only '+shlex.quote(path)+' 2>&1').decode()
  preflight[path]={'script_sha256':hashlib.sha256(body.encode()).hexdigest(),'scheduler_result':result}
 common.persist(HERE/'scheduler-preflights.json',preflight)
 source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('*.py')}
 common.persist(HERE/'source-manifest.json',{'files':source,'fingerprint':hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()})
 prep=common.submit('preparation',gate)
 for name in NAMES:common.submit(name,ROOT+'/scripts/'+name+'.sbatch',prep)
 state=json.loads((HERE/'submissions.json').read_text())
 candidates={name:{**state[name],'variant':name,'name':name,'world_size':2,'gpu':'H200','gpu_count':2,'family':ROOT,'run_directory':ROOT+'/runs/'+name} for name in NAMES}
 common.persist(HERE/'hardware_candidates.json',candidates)
 common.remote('mkdir -p '+ROOT+'/submissions && cat > '+ROOT+'/submissions/hardware_candidates.json',json.dumps(candidates).encode())

if __name__=='__main__':main()
