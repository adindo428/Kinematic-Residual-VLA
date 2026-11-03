#!/usr/bin/env bash
set -euo pipefail
source scripts/dom100_env.sh
variant=${1:-krv100}
gpu_count=${DOM100_DDP_GPUS:-${SLURM_GPUS_ON_NODE:-4}}
case "${gpu_count}" in 4|8) ;; *) echo "Request four or eight GPUs on one node." >&2; exit 2 ;; esac
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
continuation="${DOM100_ROOT}/runs/${variant}/continuation-${SLURM_JOB_ID:-local}.json"
rm -f "${continuation}"
child=
forward_stop() {
    if [[ -n "${child}" ]]; then
        python - "${child}" <<'PY'
import os,signal,sys,psutil
try:
    children=psutil.Process(int(sys.argv[1])).children()
    for process in children:
        if any('dom100_distributed.py' in item for item in process.cmdline()):
            os.kill(process.pid,signal.SIGUSR1)
except psutil.Error:
    pass
PY
    fi
}
trap forward_stop USR1 TERM
resume_args=()
if [[ "${DOM100_RESUME_STOPPED:-0}" == 1 ]]; then resume_args+=(--resume-stopped); fi
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${gpu_count}" \
    scripts/dom100_distributed.py --family-root "${DOM100_ROOT}" --variant "${variant}" \
    --microbatch "${DOM100_DDP_MICROBATCH:-44}" --workers "${DOM100_DDP_WORKERS:-1}" "${resume_args[@]}" &
child=$!
set +e
wait "${child}"
status=$?
while kill -0 "${child}" 2>/dev/null; do wait "${child}"; status=$?; done
if (( status > 128 )); then wait "${child}"; status=$?; fi
set -e
# torchrun reports a child exit of 75 as its own exit of 1. The worker writes
# its orderly-continuation record after the synchronized checkpoint.
if [[ -f "${continuation}" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]] && scontrol requeue "${SLURM_JOB_ID}"; then exit 0; fi
    echo "Checkpoint saved. Resubmit this file to resume." >&2
    exit 75
fi
exit "${status}"
