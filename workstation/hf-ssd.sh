#!/usr/bin/env bash
set -euo pipefail
hf_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$hf_repo/workstation/env.local.sh"
umask 077
exec "$POLICY_PYTHON" -c 'from huggingface_hub import login; login(add_to_git_credential=False)'
