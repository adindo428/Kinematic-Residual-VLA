#!/usr/bin/env bash
set -euo pipefail
mode=${1:?mode required}
export DOM100_NODE_MODE="$mode"
export DOM100_NODE_COUNT=${SLURM_JOB_NUM_NODES:?allocation required}
export DOM100_NODE_MASTER
DOM100_NODE_MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export DOM100_NODE_PORT=$((18000 + SLURM_JOB_ID % 10000))
export GLOO_SOCKET_IFNAME=hsn0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
# Keep the site NCCL/OFI environment; only enable its communication diagnostics.
export NCCL_DEBUG=INFO
child=
forward_stop() {
    if [[ "$mode" == train && -n "$child" ]]; then
        srun --overlap --nodes="$DOM100_NODE_COUNT" --ntasks="$DOM100_NODE_COUNT" --ntasks-per-node=1 --cpus-per-task=1 \
            python "$DOM100_ROOT/stop_ranks.py" || true
    fi
}
trap forward_stop USR1 TERM
srun --overlap --nodes="$DOM100_NODE_COUNT" --ntasks="$DOM100_NODE_COUNT" --ntasks-per-node=1 --cpus-per-task=32 \
    bash "$DOM100_ROOT/node_torchrun.sh" &
child=$!
set +e
wait "$child";status=$?
while kill -0 "$child" 2>/dev/null; do wait "$child";status=$?;done
if (( status > 128 )); then wait "$child";status=$?;fi
set -e
continuation="$DOM100_ROOT/runs/krvp100/continuation-$SLURM_JOB_ID.json"
if [[ "$mode" == train && -f "$continuation" ]]; then
    if scontrol requeue "$SLURM_JOB_ID"; then exit 0;fi
    exit 75
fi
exit "$status"
