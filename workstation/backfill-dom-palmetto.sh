#!/usr/bin/env bash
set -euo pipefail
backfill_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$backfill_repo/workstation/env.local.sh"
exec "$POLICY_PYTHON" -u "$backfill_repo/workstation/backfill_dom_palmetto.py"
