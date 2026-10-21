"""Write resource requests for the three approved four-GPU allocations."""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--site',required=True);p.add_argument('--recipe')
    a=p.parse_args();r=a.root.resolve();s=r/'scripts';s.mkdir(exist_ok=True);(r/'logs').mkdir(exist_ok=True)
    pal=a.site=='palmetto';delta=a.site=='delta'
    account='cuuser_yue6_vae_verification' if pal else 'bivv-delta-gpu' if delta else 'bivv-dtai-gh'
    partition='work1' if pal else 'gpuH200x8' if delta else 'ghx4'
    gpu='#SBATCH --gpus=a100:4\n#SBATCH --constraint=gpu_a100_40gb' if pal else '#SBATCH --gpus-per-node=4'
    key=a.recipe if pal else a.site
    if pal and key not in ('smolvla_baseline','smolvla_kr','domv100'):
        raise ValueError('Select a Palmetto model')
    header=f'''#!/usr/bin/env bash
#SBATCH --job-name=main8-20k-{key}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem={'448G' if pal else '512G' if delta else '400G'}
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
{gpu}
#SBATCH --output={r}/logs/allocation-%j.out
#SBATCH --error={r}/logs/allocation-%j.err
'''
    if pal:
        activation='''export LD_LIBRARY_PATH=/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/lib64:/home/aparame/.conda/envs/dynamicvla/lib:${LD_LIBRARY_PATH:-}
'''
        python='/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/env/bin/python'
    elif delta:
        activation='module load pytorch-conda/2.12\n';python=str(r/'env-delta-x86/bin/python')
    else:
        activation='export LD_LIBRARY_PATH=/sw/user/python/miniforge3-pytorch-2.12.0/lib:${LD_LIBRARY_PATH:-}\n';python=str(r/'env-deltaai-arm/bin/python')
    secrets='/home/aparame/.config/kinematic-residual-vla/secrets.env' if pal else '/u/aparameshwaran/.config/kinematic-residual-vla/'+('secrets-delta.env' if delta else 'secrets.env')
    body=f'''set -euo pipefail
cd {r}
{activation}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export GLOO_SOCKET_IFNAME=lo NCCL_DEBUG=WARN
secret={secrets}
if [[ -f "$secret" && ! -L "$secret" && $(stat -c %a "$secret") == 600 ]]; then
  set -a
  source "$secret"
  set +a
fi
export WANDB_CONFIG_DIR={r}/wandb-config
mkdir -p "$WANDB_CONFIG_DIR"
{python+' admit.py --site delta --root '+str(r) if delta else ''}
runner=
forward_stop() {{ if [[ -n "$runner" ]]; then kill -USR1 "$runner" || true; fi; }}
trap forward_stop USR1 TERM
{python} supervise.py --root {r} --site {a.site} {'--recipe '+a.recipe if pal else ''} &
runner=$!
set +e
while true; do
  wait "$runner"
  status=$?
  if ! kill -0 "$runner" 2>/dev/null; then break; fi
done
set -e
if [[ $status -eq 75 ]]; then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$status"
'''
    (s/(key+'-train.sbatch')).write_text(header+body)


if __name__=='__main__':main()
