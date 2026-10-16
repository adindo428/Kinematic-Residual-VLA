#!/usr/bin/env bash
# SSD-only released DOM baseline, with the authors' 4.5 / 2.2.1 software stack.
set -euo pipefail
paper_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$paper_repo/workstation/env.local.sh"
paper_runner="$KRVLA_REPO/evaluation/official_dom/scripts/local.py"
paper_action="${1:-help}"
if (($#)); then shift; fi
case "$paper_action" in
  fetch)
    exec "$POLICY_PYTHON" "$paper_runner" fetch --stage "$STAGE" --stack paper45 "$@" ;;
  prepare)
    exec "$POLICY_PYTHON" "$paper_runner" prepare --stage "$STAGE" --assets "$ASSETS" \
      --root "$RESULTS" --stack paper45 --gpu-label 'two RTX A6000 48GB (separate simulation/inference)' "$@" ;;
  pilot|full)
    exec "$POLICY_PYTHON" "$paper_runner" run --stage "$STAGE" --assets "$ASSETS" \
      --root "$RESULTS" --isaaclab "$ISAACLAB" --policy-python "$POLICY_PYTHON" \
      --sim-gpu "$SIM_GPU" --policy-gpu "$POLICY_GPU" --mode "$paper_action" \
      --kit-args="$KRVLA_KIT_ARGS" "$@" ;;
  report) exec "$POLICY_PYTHON" "$paper_runner" report --root "$RESULTS" "$@" ;;
  status)
    "$POLICY_PYTHON" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['RESULTS'])
status_path = root/'suite_status.json'
status = json.loads(status_path.read_text()) if status_path.exists() else {'state':'not started'}
print('Suite:', status['state'])
total = 0
for folder in sorted((root/'full').glob('*')):
    path = folder/'trial_results.jsonl'
    count = 0
    if path.exists():
        for line in path.read_text().splitlines():
            try: json.loads(line)
            except json.JSONDecodeError: continue  # A record may currently be writing.
            count += 1
    total += count
    print(f'{folder.name}: {count}/200 trials logged')
print(f'Total: {total}/1800 trials logged (final accounting is verified by the runner)')
print('Results:', root)
PY
    ;;
  *) echo 'Usage: bash workstation/dom-paper45.sh {fetch|prepare|pilot|full|report|status} [extra arguments]'; exit 2 ;;
esac
