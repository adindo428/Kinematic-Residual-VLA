# Copy to workstation/env.local.sh, edit paths, then source it from the repo root.
export KRVLA_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KRVLA_WORK="$HOME/krvla-workstation"
export POLICY_PYTHON="$HOME/miniconda3/envs/krvla-policy/bin/python"
export ISAACLAB_ROOT="$HOME/IsaacLab"
export ISAACLAB_ENV="env_isaaclab"
export ISAACLAB="$KRVLA_REPO/workstation/isaaclab.sh"
export STAGE="$KRVLA_WORK/official"
export ASSETS="$KRVLA_WORK/assets"
export RESULTS="$KRVLA_WORK/results/dom-rtxpro-01"
export CHECKPOINTS="$KRVLA_WORK/checkpoints"
export HF_HOME="$KRVLA_WORK/cache/huggingface"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
