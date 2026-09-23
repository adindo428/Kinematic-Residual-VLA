#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

BASELINE_CFG=${BASELINE_CFG:-configs/dynamicvla_dom_full_finetune_speed_ood_0p12_0p13.yaml}
RESIDUAL_CFG=${RESIDUAL_CFG:-configs/residual_dynamicvla_dom_full_finetune_speed_ood_0p12_0p13.yaml}
BASELINE_EXP=${BASELINE_EXP:-dynamicvla_dom_full_finetune_speed_ood_0p12_0p13}
RESIDUAL_EXP=${RESIDUAL_EXP:-residual_dynamicvla_dom_full_finetune_speed_ood_0p12_0p13}
BASELINE_CKPT=${BASELINE_CKPT:-}
RESIDUAL_CKPT=${RESIDUAL_CKPT:-}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
AUTO_RESUME=${AUTO_RESUME:-1}
RUN_PREFETCH=${RUN_PREFETCH:-1}
RUN_TRAIN=${RUN_TRAIN:-1}
RUN_OFFLINE_COMPARE=${RUN_OFFLINE_COMPARE:-1}
OFFLINE_COMPARE_OUTPUT=${OFFLINE_COMPARE_OUTPUT:-runs/proactive_vla/offline_compare/speed_ood_0p12_0p13_full_finetune_full_range_report.json}
PROGRESS_EVERY=${PROGRESS_EVERY:-100}

if command -v module >/dev/null 2>&1; then
  module load cuda/12.3.0 || true
  module load gcc/12.3.0 || true
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate dynamicvla
fi

mkdir -p runs/slurm_logs runs/proactive_vla/offline_compare

echo "[full-finetune-speed-ood] NPROC_PER_NODE=${NPROC_PER_NODE} AUTO_RESUME=${AUTO_RESUME} DVLA_BATCH_SIZE=${DVLA_BATCH_SIZE:-config} DVLA_GRAD_ACCUM_STEPS=${DVLA_GRAD_ACCUM_STEPS:-config}"

checkpoint_for_exp() {
  local exp_name=$1
  local explicit_ckpt=$2
  local ckpt_dir="runs/checkpoints/${exp_name}"

  if [[ -n "${explicit_ckpt}" ]]; then
    printf '%s\n' "${explicit_ckpt}"
    return 0
  fi
  if [[ "${AUTO_RESUME}" == "1" && -f "${ckpt_dir}/config.json" && -f "${ckpt_dir}/model.safetensors" ]]; then
    printf '%s\n' "${ckpt_dir}"
    return 0
  fi
  printf '\n'
}

run_torchrun_train() {
  local label=$1
  local cfg=$2
  local exp_name=$3
  local explicit_ckpt=$4
  local ckpt
  ckpt=$(checkpoint_for_exp "${exp_name}" "${explicit_ckpt}")

  local -a cmd=(torchrun --standalone --nnodes=1 "--nproc_per_node=${NPROC_PER_NODE}" run.py --cfg "${cfg}" --exp "${exp_name}")
  if [[ -n "${ckpt}" ]]; then
    echo "[full-finetune-speed-ood] Resuming ${label} from ${ckpt}"
    cmd+=(--ckpt "${ckpt}")
  else
    echo "[full-finetune-speed-ood] Starting ${label} from config checkpoint"
  fi
  "${cmd[@]}"
}

if [[ "${RUN_PREFETCH}" == "1" ]]; then
  echo "[full-finetune-speed-ood] Prefetching train split for baseline config"
  python run.py --cfg "${BASELINE_CFG}" --exp "${BASELINE_EXP}" --prefetch-only

  echo "[full-finetune-speed-ood] Prefetching train split for residual config"
  python run.py --cfg "${RESIDUAL_CFG}" --exp "${RESIDUAL_EXP}" --prefetch-only
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  run_torchrun_train "baseline full-parameter VLA" "${BASELINE_CFG}" "${BASELINE_EXP}" "${BASELINE_CKPT}"
  run_torchrun_train "kinematic residual full-parameter VLA" "${RESIDUAL_CFG}" "${RESIDUAL_EXP}" "${RESIDUAL_CKPT}"
else
  echo "[full-finetune-speed-ood] RUN_TRAIN=0; skipping GPU training."
fi

if [[ "${RUN_OFFLINE_COMPARE}" == "1" ]]; then
  echo "[full-finetune-speed-ood] Running post-training held-out offline comparison"
  python scripts/proactive_residual_offline_compare.py \
    --cfg "${RESIDUAL_CFG}" \
    --baseline-checkpoint "runs/checkpoints/${BASELINE_EXP}" \
    --residual-checkpoint "runs/checkpoints/${RESIDUAL_EXP}" \
    --output "${OFFLINE_COMPARE_OUTPUT}" \
    --max-batches 0 \
    --progress-every "${PROGRESS_EVERY}"
fi

echo "[full-finetune-speed-ood] Done. Report: ${OFFLINE_COMPARE_OUTPUT}"
