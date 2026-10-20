#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

BASELINE_CFG=${BASELINE_CFG:-configs/dynamicvla_dom_full_finetune_speed_ood_0p10_0p18.yaml}
RESIDUAL_CFG=${RESIDUAL_CFG:-configs/residual_dynamicvla_dom_full_finetune_speed_ood_0p10_0p18.yaml}
BASELINE_EXP=${BASELINE_EXP:-dynamicvla_dom_full_finetune_speed_ood_0p10_0p18}
RESIDUAL_EXP=${RESIDUAL_EXP:-residual_dynamicvla_dom_full_finetune_speed_ood_0p10_0p18}
BASELINE_CKPT=${BASELINE_CKPT:-}
RESIDUAL_CKPT=${RESIDUAL_CKPT:-}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
AUTO_RESUME=${AUTO_RESUME:-1}
RUN_PREFETCH=${RUN_PREFETCH:-1}
RUN_TRAIN=${RUN_TRAIN:-1}
RUN_OFFLINE_COMPARE=${RUN_OFFLINE_COMPARE:-0}
OFFLINE_COMPARE_OUTPUT=${OFFLINE_COMPARE_OUTPUT:-${DVLA_OUTPUT_ROOT:-/scratch/aparame/Dynamic_VLA/runs}/proactive_vla/offline_compare/speed_ood_0p10_0p18_full_finetune_report.json}
PROGRESS_EVERY=${PROGRESS_EVERY:-100}

source "${SCRIPT_DIR}/palmetto_env.sh"
mkdir -p "${DVLA_OUTPUT_ROOT}/proactive_vla/offline_compare"
python scripts/rebuild_dom_speed_split.py

echo "[full-finetune-speed-ood] NPROC_PER_NODE=${NPROC_PER_NODE} AUTO_RESUME=${AUTO_RESUME} DVLA_BATCH_SIZE=${DVLA_BATCH_SIZE:-config} DVLA_GRAD_ACCUM_STEPS=${DVLA_GRAD_ACCUM_STEPS:-config}"

checkpoint_for_exp() {
  local exp_name=$1
  local explicit_ckpt=$2
  local ckpt_dir="${DVLA_OUTPUT_ROOT}/checkpoints/${exp_name}"

  if [[ -n "${explicit_ckpt}" ]]; then
    if [[ ! -f "${explicit_ckpt}/training_state.pt" ]]; then
      echo "Resume checkpoint lacks training_state.pt: ${explicit_ckpt}" >&2
      return 2
    fi
    printf '%s\n' "${explicit_ckpt}"
    return 0
  fi
  if [[ "${AUTO_RESUME}" == "1" && -f "${ckpt_dir}/config.json" && -f "${ckpt_dir}/model.safetensors" && -f "${ckpt_dir}/training_state.pt" ]]; then
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
  python scripts/check_train_readiness.py --config "${BASELINE_CFG}"
  run_torchrun_train "baseline full-parameter VLA" "${BASELINE_CFG}" "${BASELINE_EXP}" "${BASELINE_CKPT}"
  run_torchrun_train "kinematic residual full-parameter VLA" "${RESIDUAL_CFG}" "${RESIDUAL_EXP}" "${RESIDUAL_CKPT}"
else
  echo "[full-finetune-speed-ood] RUN_TRAIN=0; skipping GPU training."
fi

if [[ "${RUN_OFFLINE_COMPARE}" == "1" ]]; then
  echo "[full-finetune-speed-ood] Running post-training held-out offline comparison"
  python scripts/proactive_residual_offline_compare.py \
    --cfg "${RESIDUAL_CFG}" \
    --baseline-checkpoint "${DVLA_OUTPUT_ROOT}/checkpoints/${BASELINE_EXP}" \
    --residual-checkpoint "${DVLA_OUTPUT_ROOT}/checkpoints/${RESIDUAL_EXP}" \
    --output "${OFFLINE_COMPARE_OUTPUT}" \
    --max-batches 0 \
    --progress-every "${PROGRESS_EVERY}"
fi

echo "[full-finetune-speed-ood] Done. Report: ${OFFLINE_COMPARE_OUTPUT}"
