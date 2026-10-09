#!/usr/bin/env bash
# Run inside a dedicated Python 3.10 policy environment; never the Isaac environment.
set -euo pipefail
workstation_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workstation_python="${POLICY_PYTHON:-python}"
"$workstation_python" - <<'PY'
import platform, sys
if platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise SystemExit('This installer targets Linux x86_64 NVIDIA workstations.')
if sys.version_info[:2] != (3, 10):
    raise SystemExit('Activate a dedicated Python 3.10 policy environment first.')
if 'isaac' in sys.prefix.lower():
    raise SystemExit('Do not install policy packages into the Isaac environment.')
PY
"$workstation_python" -m pip install --upgrade pip
"$workstation_python" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"$workstation_python" -m pip install -r "$workstation_repo/evaluation/official_dom/requirements-policy.txt" -c "$workstation_repo/evaluation/official_dom/requirements-constraints.txt"
"$workstation_python" -m pip check
"$workstation_python" "$workstation_repo/workstation/preflight.py" --output "${KRVLA_WORK:-$HOME/krvla-workstation}/checks/policy.json"
