"""One-shot guarded staging and submission of the eight authorized replicas."""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import tarfile
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[1]/'experiments/main8_palmetto_replicas_20261021'
BASE='/scratch/aparame/Dynamic_VLA'
ROOTS={kind:BASE+'/main8-replicas-'+kind+'-20261021' for kind in ('a100','h200')}
PYTHON=BASE+'/cross-vla-krfm-five-epochs-20261020/env/bin/python'
SSH=['ssh','-F','/dev/null','-S','/private/tmp/vlcm-palmetto-k1vtw3i7/socket','-o','BatchMode=yes','-o','ConnectTimeout=15','aparame@slogin.palmetto.clemson.edu']
NAMES=('pi05_baseline','pi05_kr','vlash_baseline','vlash_kr')
ENV='export LD_LIBRARY_PATH=/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/lib64:/home/aparame/.conda/envs/dynamicvla/lib:${LD_LIBRARY_PATH:-}\nexport PYTHONPATH="$PWD:/scratch/aparame/Dynamic_VLA/cross-vla-krfm-20261019/upstream-vlash"\nexport OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1\n'

def persist(file,value):
    temp=file.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(file)

def remote(command,input=None,timeout=60):
    return subprocess.run(SSH+[command],input=input,capture_output=True,check=True,timeout=timeout).stdout

def header(name,root,nodes,gpu=None):
    text=f'''#!/usr/bin/env bash
#SBATCH --job-name={name}
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={32 if gpu else 8}
#SBATCH --mem={('224G' if gpu=='a100' else '256G') if gpu else '64G'}
#SBATCH --time={'48:00:00' if gpu else '02:00:00'}
#SBATCH --output={root}/logs/{name}-%j.out
#SBATCH --error={root}/logs/{name}-%j.err
'''
    if gpu:text+=f'#SBATCH --gpus-per-node={gpu}:2\n#SBATCH --constraint={"gpu_a100_80gb" if gpu=="a100" else "gpu_h200"}\n#SBATCH --signal=B:USR1@1200\n#SBATCH --requeue\n'
    return text+'set -euo pipefail\ncd '+root+'\n'+ENV

def scripts():
    files={}
    gate=ROOTS['a100']+'/scripts/prepare.sbatch'
    body=header('main8-replica-prepare',ROOTS['a100'],2)
    body+=f'''{PYTHON} -m unittest test_budget test_placements test_optimizer test_autocast > cpu-contracts.log 2>&1
set +e
{PYTHON} allocation.py --root {ROOTS['a100']} --name pi05_baseline --placement a100 --cpu-contract
status=$?
set -e
[[ $status -eq 75 ]]
{PYTHON} -c 'import json; assert json.load(open("two-node-contract.json"))["passed"]'
'''
    for kind,root in ROOTS.items():body+=f'cd {root}\n{PYTHON} prepare_replicas.py --root {root} --placement {kind}\n'
    files[gate]=body
    for kind,root in ROOTS.items():
        for name in NAMES:
            script=header('main8-rep-'+kind+'-'+name,root,2 if kind=='a100' else 1,kind)
            script+=f'''export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_CONFIG_DIR={root}/wandb-config
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
{PYTHON} allocation.py --root {root} --name {name} --placement {kind} &
runner=$!
set +e
while true; do
  wait "$runner"
  status=$?
  if ! kill -0 "$runner" 2>/dev/null; then break; fi
done
set -e
if [[ $status -eq 75 ]]; then scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
exit "$status"
'''
            files[root+'/scripts/'+name+'.sbatch']=script
    return files

def submit(key,script,dependency=None):
    statefile=HERE/'submissions.json';state=json.loads(statefile.read_text()) if statefile.exists() else {}
    if key in state:
        assert state[key].get('job_id'),'Uncertain submission; reconcile before retry'
        return state[key]['job_id']
    spec={'script':script,'script_sha256':json.loads((HERE/'scheduler-preflights.json').read_text())[script]['script_sha256'],'dependency':dependency,'intent_at':datetime.now(timezone.utc).isoformat(),'state':'intent'}
    state[key]=spec;persist(statefile,state)
    remote('cat > '+shlex.quote(ROOTS['a100']+'/submission-intent.json'),json.dumps(state).encode())
    args=['sbatch','--parsable']+(['--dependency=afterok:'+dependency] if dependency else [])+[script]
    result=remote(shlex.join(args)).decode().strip();spec['stdout']=result
    job=result.split(';')[0];assert job.isdigit(),'Uncertain scheduler result; reconcile: '+result
    spec.update(job_id=job,state='submitted');persist(statefile,state)
    for root in ROOTS.values():remote('cat > '+shlex.quote(root+'/submissions.json'),json.dumps(state).encode())
    print(json.dumps({key:spec}),flush=True);return job

def main():
    assert not (HERE/'submissions.json').exists(),'Existing state must be reconciled; never rerun launcher'
    queue=remote('squeue -u aparame -h -o "%i|%j|%T|%R"').decode()
    assert 'main8-rep-' not in queue and 'main8-replica-prepare' not in queue,'Replica jobs already exist'
    (HERE/'queue-before.txt').write_text(queue)
    for root in ROOTS.values():
        remote('test ! -e '+shlex.quote(root)+' && mkdir -p '+shlex.quote(root+'/scripts')+' '+shlex.quote(root+'/logs'))
    with tarfile.open(HERE/'source.tar','w') as archive:
        for p in sorted(SOURCE.iterdir()):
            if p.is_file():archive.add(p,arcname=p.name)
        archive.add(HERE/'source-checkpoint-hashes.json',arcname='source-checkpoint-hashes.json')
    data=(HERE/'source.tar').read_bytes()
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCE.iterdir()) if p.is_file()}
    persist(HERE/'upload-manifest.json',manifest)
    for kind,root in ROOTS.items():
        remote('tar -xf - -C '+shlex.quote(root),data)
        config={'mode':'separate_replicas','placement':kind,'canonical_assignments_unchanged':True,'authorization':'User explicitly selected separate replicas alongside existing jobs on 2026-10-21','origin_family':'/scratch/aparame/Dynamic_VLA/main8-20k-10epochs-20261021','models':list(NAMES)}
        remote('cat > '+shlex.quote(root+'/replica-config.json'),json.dumps(config).encode())
    specs=scripts();preflights={}
    for path,text in specs.items():
        remote('cat > '+shlex.quote(path),text.encode())
        result=remote('bash -n '+shlex.quote(path)+' && sbatch --test-only '+shlex.quote(path)+' 2>&1').decode()
        preflights[path]={'script_sha256':hashlib.sha256(text.encode()).hexdigest(),'scheduler_result':result}
    persist(HERE/'scheduler-preflights.json',preflights)
    gate=submit('preparation',ROOTS['a100']+'/scripts/prepare.sbatch')
    for kind,root in ROOTS.items():
        for name in NAMES:submit(kind+'/'+name,root+'/scripts/'+name+'.sbatch',gate)
    state=json.loads((HERE/'submissions.json').read_text())
    candidates={'replicas':True,'models':[{**state[kind+'/'+name],'name':name,'placement':kind,'root':root,'world_size':4 if kind=='a100' else 2,'gpu_count':4 if kind=='a100' else 2,'gpu_class':'A10080' if kind=='a100' else 'H200','run_directory':root+'/runs/'+name} for kind,root in ROOTS.items() for name in NAMES]}
    persist(HERE/'hardware_candidates.json',candidates)
    for root in ROOTS.values():remote('cat > '+shlex.quote(root+'/hardware_candidates.json'),json.dumps(candidates).encode())

if __name__=='__main__':main()
