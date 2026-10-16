#!/usr/bin/env bash
set -euo pipefail
dom_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$dom_repo/workstation/env.local.sh"
export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=30 HF_HUB_DISABLE_XET=1
exec "$POLICY_PYTHON" -u "$dom_repo/workstation/download_dom_dataset.py" \
  --dataset-root "$HF_HOME/lerobot/hzxie/DOM" \
  --state-root "$KRVLA_WORK/training/dom-full" \
  --revision bbf6baa15f7253746d271a6e958e037afa537b1e --workers 4 --cameras training --direct-get "$@"
