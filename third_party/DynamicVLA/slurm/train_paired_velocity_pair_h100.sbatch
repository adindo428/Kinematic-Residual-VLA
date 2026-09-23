#!/usr/bin/env bash
#SBATCH --job-name=dvla-paired-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
# Default allocation; scripts/submit_paired_velocity_pair.sh overrides this request.
#SBATCH --gres=gpu:h100:8
#SBATCH --time=72:00:00
#SBATCH --mem=512GB
#SBATCH --output=runs/slurm_logs/%x-%j.out
#SBATCH --error=runs/slurm_logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the DynamicVLA project root}"
mkdir -p runs/slurm_logs

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[paired-velocity] nvidia-smi is required for GPU auto-tuning." >&2
  exit 2
fi

NVIDIA_SMI_DEVICE_ARGS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
  NVIDIA_SMI_DEVICE_ARGS=(-i "${CUDA_VISIBLE_DEVICES}")
fi

mapfile -t ALLOCATED_GPU_NAMES < <(
  nvidia-smi "${NVIDIA_SMI_DEVICE_ARGS[@]}" --query-gpu=name --format=csv,noheader
)
mapfile -t ALLOCATED_GPU_MEMORY_MIB < <(
  nvidia-smi "${NVIDIA_SMI_DEVICE_ARGS[@]}" --query-gpu=memory.total --format=csv,noheader,nounits
)
ALLOCATED_GPU_COUNT=${#ALLOCATED_GPU_NAMES[@]}

if (( ALLOCATED_GPU_COUNT < 4 || ALLOCATED_GPU_COUNT > 8 )); then
  echo "[paired-velocity] Expected 4-8 allocated GPUs, found ${ALLOCATED_GPU_COUNT}." >&2
  exit 2
fi
if [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[0-9]+$ ]] && (( SLURM_GPUS_ON_NODE != ALLOCATED_GPU_COUNT )); then
  echo "[paired-velocity] Slurm reports ${SLURM_GPUS_ON_NODE} GPUs on the job node, but nvidia-smi exposed ${ALLOCATED_GPU_COUNT}." >&2
  exit 2
fi

case "${ALLOCATED_GPU_NAMES[0],,}" in
  *h200*) DETECTED_GPU_TYPE=h200 ;;
  *h100*) DETECTED_GPU_TYPE=h100 ;;
  *a100*) DETECTED_GPU_TYPE=a100 ;;
  *)
    echo "[paired-velocity] Unsupported GPU: ${ALLOCATED_GPU_NAMES[0]}. Expected A100, H100, or H200." >&2
    exit 2
    ;;
esac

for gpu_name in "${ALLOCATED_GPU_NAMES[@]}"; do
  case "${gpu_name,,}" in
    *"${DETECTED_GPU_TYPE}"*) ;;
    *)
      echo "[paired-velocity] Mixed GPU allocation is unsupported: ${ALLOCATED_GPU_NAMES[*]}" >&2
      exit 2
      ;;
  esac
done

if [[ -n "${DVLA_REQUESTED_GPU_TYPE:-}" ]]; then
  DVLA_REQUESTED_GPU_TYPE=${DVLA_REQUESTED_GPU_TYPE,,}
  if [[ "${DVLA_REQUESTED_GPU_TYPE}" != "${DETECTED_GPU_TYPE}" ]]; then
    echo "[paired-velocity] Requested ${DVLA_REQUESTED_GPU_TYPE}, but Slurm allocated ${DETECTED_GPU_TYPE}." >&2
    exit 2
  fi
fi
if [[ -n "${DVLA_REQUESTED_GPU_COUNT:-}" ]]; then
  if [[ ! "${DVLA_REQUESTED_GPU_COUNT}" =~ ^[0-9]+$ ]]; then
    echo "[paired-velocity] DVLA_REQUESTED_GPU_COUNT must be an integer." >&2
    exit 2
  fi
  if (( DVLA_REQUESTED_GPU_COUNT != ALLOCATED_GPU_COUNT )); then
    echo "[paired-velocity] Requested ${DVLA_REQUESTED_GPU_COUNT} GPUs, but Slurm exposed ${ALLOCATED_GPU_COUNT}." >&2
    exit 2
  fi
fi

MIN_GPU_MEMORY_MIB=0
for gpu_memory_mib in "${ALLOCATED_GPU_MEMORY_MIB[@]}"; do
  gpu_memory_mib=${gpu_memory_mib//[[:space:]]/}
  if [[ ! "${gpu_memory_mib}" =~ ^[0-9]+$ ]]; then
    echo "[paired-velocity] Could not parse GPU memory value: ${gpu_memory_mib}" >&2
    exit 2
  fi
  if (( MIN_GPU_MEMORY_MIB == 0 || gpu_memory_mib < MIN_GPU_MEMORY_MIB )); then
    MIN_GPU_MEMORY_MIB=${gpu_memory_mib}
  fi
done

case "${DETECTED_GPU_TYPE}" in
  h200)
    AUTO_BATCH_SIZE=64
    ;;
  h100)
    AUTO_BATCH_SIZE=32
    ;;
  a100)
    if (( MIN_GPU_MEMORY_MIB >= 70000 )); then
      AUTO_BATCH_SIZE=32
    else
      AUTO_BATCH_SIZE=16
    fi
    ;;
esac

export NPROC_PER_NODE=${NPROC_PER_NODE:-${ALLOCATED_GPU_COUNT}}
if [[ ! "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || (( NPROC_PER_NODE < 1 || NPROC_PER_NODE > ALLOCATED_GPU_COUNT )); then
  echo "[paired-velocity] NPROC_PER_NODE must be between 1 and ${ALLOCATED_GPU_COUNT}, got ${NPROC_PER_NODE}." >&2
  exit 2
fi

export DVLA_TARGET_GLOBAL_BATCH_SIZE=${DVLA_TARGET_GLOBAL_BATCH_SIZE:-1024}
export DVLA_BATCH_SIZE=${DVLA_BATCH_SIZE:-${AUTO_BATCH_SIZE}}
if [[ ! "${DVLA_TARGET_GLOBAL_BATCH_SIZE}" =~ ^[0-9]+$ ]] || (( DVLA_TARGET_GLOBAL_BATCH_SIZE < 1 )); then
  echo "[paired-velocity] DVLA_TARGET_GLOBAL_BATCH_SIZE must be a positive integer." >&2
  exit 2
fi
if [[ ! "${DVLA_BATCH_SIZE}" =~ ^[0-9]+$ ]] || (( DVLA_BATCH_SIZE < 1 )); then
  echo "[paired-velocity] DVLA_BATCH_SIZE must be a positive integer." >&2
  exit 2
fi

if [[ -z "${DVLA_GRAD_ACCUM_STEPS:-}" ]]; then
  GLOBAL_MICROBATCH_SIZE=$((NPROC_PER_NODE * DVLA_BATCH_SIZE))
  DVLA_GRAD_ACCUM_STEPS=$(((DVLA_TARGET_GLOBAL_BATCH_SIZE + GLOBAL_MICROBATCH_SIZE / 2) / GLOBAL_MICROBATCH_SIZE))
  if (( DVLA_GRAD_ACCUM_STEPS < 1 )); then
    DVLA_GRAD_ACCUM_STEPS=1
  fi
fi
export DVLA_GRAD_ACCUM_STEPS
if [[ ! "${DVLA_GRAD_ACCUM_STEPS}" =~ ^[0-9]+$ ]] || (( DVLA_GRAD_ACCUM_STEPS < 1 )); then
  echo "[paired-velocity] DVLA_GRAD_ACCUM_STEPS must be a positive integer." >&2
  exit 2
fi

EFFECTIVE_GLOBAL_BATCH_SIZE=$((NPROC_PER_NODE * DVLA_BATCH_SIZE * DVLA_GRAD_ACCUM_STEPS))
echo "[paired-velocity] Auto-tuned hardware: type=${DETECTED_GPU_TYPE} count=${ALLOCATED_GPU_COUNT} min_memory_mib=${MIN_GPU_MEMORY_MIB}"
echo "[paired-velocity] Training geometry: processes=${NPROC_PER_NODE} batch_per_gpu=${DVLA_BATCH_SIZE} grad_accum=${DVLA_GRAD_ACCUM_STEPS} effective_global_batch=${EFFECTIVE_GLOBAL_BATCH_SIZE} target=${DVLA_TARGET_GLOBAL_BATCH_SIZE}"

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-4}
export DVLA_N_WORKERS=${DVLA_N_WORKERS:-2}
export DVLA_DATALOADER_PREFETCH_FACTOR=${DVLA_DATALOADER_PREFETCH_FACTOR:-2}
export DVLA_PERSISTENT_WORKERS=${DVLA_PERSISTENT_WORKERS:-true}

export AUTO_RESUME=${AUTO_RESUME:-0}
export RESUME_RESIDUAL_ONLY=${RESUME_RESIDUAL_ONLY:-0}
export BASELINE_INIT_CKPT=${BASELINE_INIT_CKPT:-hzxie/dynamic-vla-DOM}
export RESIDUAL_INIT_CKPT=${RESIDUAL_INIT_CKPT:-hzxie/dynamic-vla-DOM}
export DVLA_EVAL_SPLITS=${DVLA_EVAL_SPLITS:-"iid_eval ood_val"}
export DVLA_EVAL_FREQ_EPOCH=${DVLA_EVAL_FREQ_EPOCH:-1}
export DVLA_EVAL_MAX_BATCHES=${DVLA_EVAL_MAX_BATCHES:-1000}
export DVLA_EVAL_PRIMARY_METRIC=${DVLA_EVAL_PRIMARY_METRIC:-action_mse}
export DVLA_EVAL_PRIMARY_SPLIT=${DVLA_EVAL_PRIMARY_SPLIT:-ood_val}
export DVLA_RESIDUAL_LAMBDA_TYPE=${DVLA_RESIDUAL_LAMBDA_TYPE:-constant}
export RUN_PREFETCH=${RUN_PREFETCH:-0}
export RUN_TRAIN=${RUN_TRAIN:-1}
export RUN_OBSERVED_EVAL=${RUN_OBSERVED_EVAL:-1}
export RUN_PAIRED_SWEEP=${RUN_PAIRED_SWEEP:-1}
export OBSERVED_EVAL_SPLITS=${OBSERVED_EVAL_SPLITS:-"iid_eval ood_val ood_test"}
export OBSERVED_EVAL_MAX_BATCHES=${OBSERVED_EVAL_MAX_BATCHES:-1000}

bash scripts/train_paired_velocity_pair.sh
