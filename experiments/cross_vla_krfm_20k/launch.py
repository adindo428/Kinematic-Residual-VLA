"""Render launch scripts and submit individual jobs with durable receipts.

Preparation runs on CPUs. GPU profiles must complete for this exact source,
data and recipe before any production job can train. No old job is modified.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
from common import atomic, sha
from subset import require

PARENT = '/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020'
NAMES = ('pi05_baseline', 'pi05_kr', 'smolvla_baseline', 'smolvla_kr', 'vlash_baseline', 'vlash_kr')


def render(root, parent):
    root = Path(root)
    q = shlex.quote
    scripts = root / 'scripts'
    scripts.mkdir(exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    for command, names in [('prepare', ('dataset',)), ('profile', NAMES), ('train', NAMES)]:
        for name in names:
            key = f'{name}-{command}'
            gpu = 'h200' if name.startswith('pi05') else 'a100'
            memory = '256G' if gpu == 'h200' else '224G'
            header = f'''#!/bin/bash -l
#SBATCH --job-name=k20_{key}
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={8 if command == 'prepare' else 32}
#SBATCH --mem={'32G' if command == 'prepare' else memory}
#SBATCH --time={'04:00:00' if command == 'prepare' else '04:00:00' if command == 'profile' else '48:00:00'}
#SBATCH --output={root}/logs/{key}-%j.out
#SBATCH --error={root}/logs/{key}-%j.err
'''
            if command != 'prepare':
                header += f'#SBATCH --gpus={gpu}:2\n#SBATCH --constraint=gpu_{gpu}\n'
            if command == 'train':
                header += '#SBATCH --signal=B:USR1@1200\n#SBATCH --requeue\n'
            body = f'''set -euo pipefail
cd {q(str(root))}
export HF_HOME=/scratch/aparame/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH={q(str(Path(parent)/'upstream-vlash'))}
export LD_LIBRARY_PATH=/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/lib64:/home/aparame/.conda/envs/dynamicvla/lib:${{LD_LIBRARY_PATH:-}}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_START_METHOD=thread
export WANDB_INIT_TIMEOUT=45
'''
            if command == 'prepare':
                body += f'{q(str(Path(parent)/"env/bin/python"))} prepare.py --root {q(str(root))} --parent {q(str(parent))}\n'
            else:
                run = f'env/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 train.py {command} --root {q(str(root))} --recipe {name}'
                if command == 'profile':
                    body += run + f' --runtime {q(str(root/"runtimes"/(name+".json")))}\n'
                else:
                    out = q(str(root / 'runs' / name))
                    body += f'''mkdir -p {out}
rm -f {out}/stopped.json
runner=
forward_stop() {{
  if [[ -n "$runner" ]]; then pkill -USR1 -P "$runner" || true; fi
}}
trap forward_stop USR1 TERM
{run} &
runner=$!
set +e
while true; do
  wait "$runner"
  status=$?
  if ! kill -0 "$runner" 2>/dev/null; then break; fi
done
set -e
if [[ $status -eq 75 && -f {out}/stopped.json && ! -f {out}/completed.json ]]; then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
# torchrun may translate the worker's graceful exit code to 1. The stopped
# receipt is written only after every rank saves and closes successfully.
if [[ -f {out}/stopped.json && ! -f {out}/completed.json ]]; then
  env/bin/python -c 'import json; from pathlib import Path; p=Path("runs/{name}"); assert json.loads((p/"stopped.json").read_text())["progress"] == json.loads((p/"latest.json").read_text())["progress"] and (p/"latest.pt").is_file()'
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$status"
'''
            file = scripts / f'{key}.sbatch'
            text = header + body
            if file.exists():
                require(file.read_text() == text, f'Launch script changed: {file}')
            else:
                file.write_text(text)


def submit(root, name, command, dependency):
    require(command == 'prepare' or name in NAMES, 'Unknown recipe')
    key = f'{name}-{command}'
    script = root / 'scripts' / f'{key}.sbatch'
    state_path = root / 'submissions.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if key in state:
        require(state[key]['script_sha256'] == sha(script), 'Submitted script changed')
        require(state[key].get('job_id'), 'Previous submission outcome unknown; reconcile with scheduler')
        print(json.dumps(state[key]))
        return
    # Persist intent before contacting Slurm. An interrupted/unknown response
    # cannot cause the next invocation to silently launch a duplicate job.
    record = {'script': str(script), 'script_sha256': sha(script), 'dependency': dependency,
              'status': 'submission_intent', 'name': key}
    state[key] = record
    atomic(state_path, state)
    args = ['sbatch', '--parsable']
    if dependency:
        require(str(dependency).isdigit(), 'Invalid dependency job ID')
        args.append('--dependency=afterok:' + str(dependency))
    args.append(str(script))
    result = subprocess.run(args, capture_output=True, text=True)
    record.update(returncode=result.returncode, stdout=result.stdout.strip(), stderr=result.stderr.strip())
    job = result.stdout.strip().split(';')[0]
    if result.returncode == 0 and job.isdigit():
        record.update(job_id=job, status='submitted')
    else:
        record['status'] = 'unknown_or_failed_requires_reconciliation'
    atomic(state_path, state)
    require(record.get('job_id'), 'Submission failed or outcome unknown; inspect submissions.json before retrying')
    print(json.dumps(record), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['render', 'submit'])
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--parent', default=PARENT)
    p.add_argument('--command', choices=['prepare', 'profile', 'train'], default='prepare')
    p.add_argument('--recipe', default='dataset')
    p.add_argument('--dependency')
    a = p.parse_args()
    root = a.root.resolve()
    render(root, a.parent)
    if a.action == 'submit':
        submit(root, 'dataset' if a.command == 'prepare' else a.recipe, a.command, a.dependency)


if __name__ == '__main__':
    main()
