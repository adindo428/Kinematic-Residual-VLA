#!/usr/bin/env bash
# Authenticate locally, then reuse that connection for an SSD-only DOM transfer.
set -euo pipefail
palmetto_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$palmetto_repo/workstation/env.local.sh"
export PALMETTO_DOM_USER="${PALMETTO_DOM_USER:-aparame}"
export PALMETTO_DOM_SOURCE="${PALMETTO_DOM_SOURCE:-/scratch/$PALMETTO_DOM_USER/.cache/huggingface/lerobot/hzxie/DOM}"
export PALMETTO_DOM_HOST=hpcdtn01.rcd.clemson.edu
palmetto_socket="$KRVLA_WORK/ssh/palmetto"
palmetto_hosts="$KRVLA_WORK/ssh/known_hosts"
case "${1:-status}" in
  connect)
    if [[ ! -s "$palmetto_hosts" ]]; then
      echo 'Missing verified Clemson DTN host key; setup must finish first.' >&2
      exit 1
    fi
    if ssh -S "$palmetto_socket" -O check "$PALMETTO_DOM_USER@$PALMETTO_DOM_HOST" 2>/dev/null; then
      echo 'Palmetto transfer connection is already available.'
      exit 0
    fi
    # Password/Duo prompts stay in the user's terminal; no password is stored.
    ssh -M -N -f -S "$palmetto_socket" -o ControlPersist=24h \
      -o UserKnownHostsFile="$palmetto_hosts" -o StrictHostKeyChecking=yes \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
      "$PALMETTO_DOM_USER@$PALMETTO_DOM_HOST"
    echo 'Authenticated Palmetto connection ready for dataset transfer.'
    ;;
  run)
    exec "$POLICY_PYTHON" -u "$palmetto_repo/workstation/transfer_dom_palmetto.py"
    ;;
  status)
    "$POLICY_PYTHON" - <<'PY'
import json,os
from pathlib import Path
path=Path(os.environ['KRVLA_WORK'])/'training/dom-full/palmetto_transfer_status.json'
print(path.read_text() if path.exists() else 'Palmetto transfer not queued yet.')
PY
    ;;
  *) echo 'Usage: bash workstation/palmetto-dom.sh {connect|status}' >&2; exit 2 ;;
esac
