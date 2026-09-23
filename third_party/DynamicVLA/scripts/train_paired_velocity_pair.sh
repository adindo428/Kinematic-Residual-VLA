#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

BASELINE_CFG=${BASELINE_CFG:-configs/dynamicvla_dom_paired_velocity.yaml}
RESIDUAL_CFG=${RESIDUAL_CFG:-configs/residual_dynamicvla_dom_paired_velocity.yaml}
BASELINE_EXP=${BASELINE_EXP:-dynamicvla_dom_paired_velocity}
RESIDUAL_EXP=${RESIDUAL_EXP:-residual_dynamicvla_dom_paired_velocity}
BASELINE_CKPT=${BASELINE_CKPT:-}
RESIDUAL_CKPT=${RESIDUAL_CKPT:-}
BASELINE_INIT_CKPT=${BASELINE_INIT_CKPT:-hzxie/dynamic-vla-DOM}
RESIDUAL_INIT_CKPT=${RESIDUAL_INIT_CKPT:-hzxie/dynamic-vla-DOM}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
AUTO_RESUME=${AUTO_RESUME:-0}
RESUME_RESIDUAL_ONLY=${RESUME_RESIDUAL_ONLY:-0}
RUN_PREFETCH=${RUN_PREFETCH:-1}
RUN_TRAIN=${RUN_TRAIN:-1}
RUN_OBSERVED_EVAL=${RUN_OBSERVED_EVAL:-1}
RUN_PAIRED_SWEEP=${RUN_PAIRED_SWEEP:-1}
OBSERVED_EVAL_OUTPUT=${OBSERVED_EVAL_OUTPUT:-runs/proactive_vla/offline_compare/paired_velocity_iid_ood_report.json}
OBSERVED_EVAL_CSV_OUTPUT=${OBSERVED_EVAL_CSV_OUTPUT:-runs/proactive_vla/offline_compare/paired_velocity_iid_ood_report.csv}
OBSERVED_EVAL_SPLITS=${OBSERVED_EVAL_SPLITS:-"iid_eval ood_val ood_test"}
OBSERVED_EVAL_BATCH_SIZE=${OBSERVED_EVAL_BATCH_SIZE:-8}
OBSERVED_EVAL_MAX_BATCHES=${OBSERVED_EVAL_MAX_BATCHES:-1000}
OBSERVED_EVAL_MAX_EPISODES=${OBSERVED_EVAL_MAX_EPISODES:-0}
OBSERVED_EVAL_NUM_WORKERS=${OBSERVED_EVAL_NUM_WORKERS:-}
PAIRED_SWEEP_OUTPUT=${PAIRED_SWEEP_OUTPUT:-runs/proactive_vla/paired_velocity_sweep/report.json}
DVLA_EVAL_TEST_DURING_TRAIN=${DVLA_EVAL_TEST_DURING_TRAIN:-0}
export DVLA_EVAL_TEST_DURING_TRAIN

if command -v module >/dev/null 2>&1; then
  module load cuda/12.3.0 || true
  module load gcc/12.3.0 || true
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate dynamicvla
fi

mkdir -p runs/slurm_logs runs/proactive_vla/paired_velocity_sweep runs/proactive_vla/offline_compare

if [[ "${RESUME_RESIDUAL_ONLY}" != "0" && "${RESUME_RESIDUAL_ONLY}" != "1" ]]; then
  echo "[paired-velocity] RESUME_RESIDUAL_ONLY must be 0 or 1, got: ${RESUME_RESIDUAL_ONLY}" >&2
  exit 2
fi

echo "[paired-velocity] NPROC_PER_NODE=${NPROC_PER_NODE} AUTO_RESUME=${AUTO_RESUME} RESUME_RESIDUAL_ONLY=${RESUME_RESIDUAL_ONLY} DVLA_BATCH_SIZE=${DVLA_BATCH_SIZE:-config} DVLA_GRAD_ACCUM_STEPS=${DVLA_GRAD_ACCUM_STEPS:-config}"
echo "[paired-velocity] Baseline init=${BASELINE_INIT_CKPT:-none} explicit_ckpt=${BASELINE_CKPT:-none}"
echo "[paired-velocity] Residual init=${RESIDUAL_INIT_CKPT:-none} explicit_ckpt=${RESIDUAL_CKPT:-none}"

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

checkpoint_for_eval() {
  local exp_name=$1
  local explicit_ckpt=$2
  if [[ -n "${explicit_ckpt}" ]]; then
    printf '%s\n' "${explicit_ckpt}"
  else
    printf 'runs/checkpoints/%s\n' "${exp_name}"
  fi
}

run_torchrun_train() {
  local label=$1
  local cfg=$2
  local exp_name=$3
  local explicit_ckpt=$4
  local init_ckpt=$5
  local ckpt
  ckpt=$(checkpoint_for_exp "${exp_name}" "${explicit_ckpt}")

  local -a cmd=(torchrun --standalone --nnodes=1 "--nproc_per_node=${NPROC_PER_NODE}" run.py --cfg "${cfg}" --exp "${exp_name}")
  if [[ -n "${ckpt}" ]]; then
    echo "[paired-velocity] Resuming ${label} from ${ckpt}"
    cmd+=(--ckpt "${ckpt}")
  elif [[ -n "${init_ckpt}" ]]; then
    echo "[paired-velocity] Initializing ${label} from ${init_ckpt}"
    cmd+=(--init-ckpt "${init_ckpt}")
  else
    echo "[paired-velocity] Starting ${label} from config initialization"
  fi
  "${cmd[@]}"
}

residual_resume_checkpoint() {
  local ckpt=${RESIDUAL_CKPT:-runs/checkpoints/${RESIDUAL_EXP}}
  local required_file

  for required_file in config.json model.safetensors training_state.pt; do
    if [[ ! -f "${ckpt}/${required_file}" ]]; then
      echo "[paired-velocity] Cannot resume residual-only training: missing ${ckpt}/${required_file}" >&2
      return 2
    fi
  done
  printf "%s\n" "${ckpt}"
}

if [[ "${RUN_PREFETCH}" == "1" ]]; then
  echo "[paired-velocity] Prefetching configured train/eval episodes"
  if [[ "${RESUME_RESIDUAL_ONLY}" == "1" ]]; then
    python run.py --cfg "${RESIDUAL_CFG}" --exp "${RESIDUAL_EXP}" --prefetch-only
  else
    python run.py --cfg "${BASELINE_CFG}" --exp "${BASELINE_EXP}" --prefetch-only
  fi
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  if [[ "${RESUME_RESIDUAL_ONLY}" == "1" ]]; then
    RESIDUAL_RESUME_CKPT=$(residual_resume_checkpoint)
    echo "[paired-velocity] Skipping baseline training and resuming only the residual VLA."
    run_torchrun_train "residual VLA" "${RESIDUAL_CFG}" "${RESIDUAL_EXP}" "${RESIDUAL_RESUME_CKPT}" ""
  else
    run_torchrun_train "baseline VLA" "${BASELINE_CFG}" "${BASELINE_EXP}" "${BASELINE_CKPT}" "${BASELINE_INIT_CKPT}"
    run_torchrun_train "residual VLA" "${RESIDUAL_CFG}" "${RESIDUAL_EXP}" "${RESIDUAL_CKPT}" "${RESIDUAL_INIT_CKPT}"
  fi
else
  echo "[paired-velocity] RUN_TRAIN=0; skipping training."
fi

BASELINE_EVAL_CKPT=$(checkpoint_for_eval "${BASELINE_EXP}" "${BASELINE_CKPT}")
RESIDUAL_EVAL_CKPT=$(checkpoint_for_eval "${RESIDUAL_EXP}" "${RESIDUAL_CKPT}")

if [[ "${RUN_OBSERVED_EVAL}" == "1" ]]; then
  echo "[paired-velocity] Running observed IID/OOD offline evaluation"
  read -r -a observed_splits <<< "${OBSERVED_EVAL_SPLITS}"
  observed_cmd=(
    python scripts/evaluate_baseline_residual_iid_ood.py
    --cfg "${RESIDUAL_CFG}"
    --baseline-checkpoint "${BASELINE_EVAL_CKPT}"
    --residual-checkpoint "${RESIDUAL_EVAL_CKPT}"
    --output "${OBSERVED_EVAL_OUTPUT}"
    --splits "${observed_splits[@]}"
    --batch-size "${OBSERVED_EVAL_BATCH_SIZE}"
    --max-batches "${OBSERVED_EVAL_MAX_BATCHES}"
    --max-episodes "${OBSERVED_EVAL_MAX_EPISODES}"
    --cached-only
    --shuffle-episodes
  )
  if [[ -n "${OBSERVED_EVAL_NUM_WORKERS}" ]]; then
    observed_cmd+=(--num-workers "${OBSERVED_EVAL_NUM_WORKERS}")
  fi
  if [[ -n "${OBSERVED_EVAL_CSV_OUTPUT}" ]]; then
    observed_cmd+=(--csv-output "${OBSERVED_EVAL_CSV_OUTPUT}")
  fi
  "${observed_cmd[@]}"
fi

if [[ "${RUN_PAIRED_SWEEP}" == "1" ]]; then
  echo "[paired-velocity] Running paired counterfactual velocity sweep (diagnostic only; not used for model selection)"
  python scripts/evaluate_residual_paired_velocity_sweep.py \
    --cfg "${RESIDUAL_CFG}" \
    --baseline-checkpoint "${BASELINE_EVAL_CKPT}" \
    --residual-checkpoint "${RESIDUAL_EVAL_CKPT}" \
    --output "${PAIRED_SWEEP_OUTPUT}" \
    --split train \
    --batch-size "${PAIRED_SWEEP_BATCH_SIZE:-8}" \
    --max-batches "${PAIRED_SWEEP_MAX_BATCHES:-500}" \
    --cached-only \
    --shuffle-episodes
fi

echo "[paired-velocity] Done. Observed eval report: ${OBSERVED_EVAL_OUTPUT}; paired sweep report: ${PAIRED_SWEEP_OUTPUT}"
