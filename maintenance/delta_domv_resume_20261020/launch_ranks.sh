#!/usr/bin/env bash
set -euo pipefail
mode=${1:?mode required}
HERE=/work/nvme/bivv/aparameshwaran/dynamicvla/setup-delta-domv-resume-20261020
export DOM100_NODE_MODE="$mode"
world=${DOM100_DIAGNOSTIC_WORLD:-8}
runtime=${DOM100_DIAGNOSTIC_RUNTIME:-$HERE/runtime.json}
if [[ "$mode" == benchmark ]]; then
 exec python -m torch.distributed.run --standalone --nproc-per-node="$world" "${DOM100_BENCHMARK_DRIVER:-$HERE/native_benchmark.py}" --family-root "$DOM100_ROOT" --runtime-file "$runtime"
fi
if [[ "$mode" == smoke ]]; then
 exec python -m torch.distributed.run --standalone --nproc-per-node="$world" "$HERE/entry.py" --command smoke --diagnostic-world "$world" --runtime-file "$runtime"
fi
child=
stop_workers() { python "$HERE/stop_ranks.py" || true; }
trap stop_workers USR1 TERM
python -m torch.distributed.run --standalone --nproc-per-node=8 "$HERE/entry.py" --command train &
child=$!
set +e
wait "$child";status=$?
while kill -0 "$child" 2>/dev/null; do wait "$child";status=$?;done
if (( status > 128 )); then wait "$child";status=$?;fi
set -e
if [[ -f "$HERE/continuation-$SLURM_JOB_ID.json" ]]; then
 python "$HERE/prepare.py" --checkpointed
 exit 0
fi
exit "$status"
