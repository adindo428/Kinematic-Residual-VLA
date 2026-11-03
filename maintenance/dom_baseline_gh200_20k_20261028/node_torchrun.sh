#!/usr/bin/env bash
set -euo pipefail
ip -o -4 addr show hsn0 >/dev/null
if [[ "$DOM100_NODE_MODE" == benchmark ]]; then
    entry="$DOM100_ROOT/native_benchmark.py"
    args=(--family-root "$DOM100_ROOT" --runtime-file "$DOM100_RUNTIME_FILE")
else
    entry="$DOM100_ROOT/entry.py"
    args=(--family-root "$DOM100_ROOT" --runtime-file "$DOM100_RUNTIME_FILE" --command "$DOM100_NODE_MODE")
fi
exec python -m torch.distributed.run --nnodes="$DOM100_NODE_COUNT" --nproc-per-node=4 \
    --node-rank="$SLURM_PROCID" --master-addr="$DOM100_NODE_MASTER" --master-port="$DOM100_NODE_PORT" \
    "$entry" "${args[@]}"
