#!/usr/bin/env bash
set -euo pipefail
python=$1
root=$2
name=$3
cd "$root"
world=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["world_size"])' "$root/recipes/$name.json")
runner=
forward_stop() {
  if [[ -n "$runner" ]]; then pkill -USR1 -P "$runner" || true; fi
}
trap forward_stop USR1 TERM
run_stage() {
  local stage=$1
  local status=0
  local runtime=()
  if [[ "$stage" == profile ]]; then runtime=(--runtime "$root/runtimes/$name.json"); fi
  "$python" -m torch.distributed.run --standalone --nproc_per_node="$world" train.py "$stage" --root "$root" --recipe "$name" "${runtime[@]}" &
  runner=$!
  set +e
  while true; do
    wait "$runner"
    status=$?
    if ! kill -0 "$runner" 2>/dev/null; then break; fi
  done
  set -e
  runner=
  return "$status"
}
if [[ ! -f "$root/profiles/$name.json" ]]; then run_stage profile; fi
rm -f "$root/runs/$name/stopped.json"
run_stage train
