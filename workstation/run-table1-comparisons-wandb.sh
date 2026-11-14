#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo/workstation/env.local.sh"
export NETRC="$WANDB_CONFIG_DIR/netrc"
exec "$KRVLA_WORK/envs/eval20k/bin/python" -u "$repo/workstation/publish_table1_comparisons.py" "$@"
