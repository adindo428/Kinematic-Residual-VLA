#!/usr/bin/env bash
# Keep project credentials and W&B configuration on the SSD.
set -euo pipefail
wandb_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$wandb_repo/workstation/env.local.sh"
export NETRC="$WANDB_CONFIG_DIR/netrc"
umask 077
mkdir -p "$WANDB_CONFIG_DIR"
chmod 700 "$WANDB_CONFIG_DIR"
case "${1:-check}" in
  login) exec "$POLICY_PYTHON" -m wandb login --relogin ;;
  check) exec "$POLICY_PYTHON" "$wandb_repo/workstation/check_training_wandb.py" ;;
  *) echo 'Usage: bash workstation/wandb-ssd.sh {login|check}' >&2; exit 2 ;;
esac
