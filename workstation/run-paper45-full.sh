#!/usr/bin/env bash
# Run the complete verified benchmark and save a durable completion/failure status.
set -euo pipefail
paper_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$paper_repo/workstation/env.local.sh"
"$POLICY_PYTHON" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['RESULTS'])
assert json.loads((root/'pilot_verified.json').read_text())['verified']
for name in ('policy', 'checkpoint', 'simulator'):
    assert json.loads((Path(os.environ['KRVLA_WORK'])/'checks'/f'{name}.json').read_text())['validated']
PY
paper_status() {
  DOM_SUITE_STATE="$1" DOM_SUITE_EXIT="$2" "$POLICY_PYTHON" - <<'PY'
import datetime, json, os
from pathlib import Path
root = Path(os.environ['RESULTS'])
path = root/'suite_status.json'
data = json.loads(path.read_text()) if path.exists() else {}
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
data.setdefault('started_at_utc', now)
data.update(state=os.environ['DOM_SUITE_STATE'],
            exit_code=None if os.environ['DOM_SUITE_STATE']=='running' else int(os.environ['DOM_SUITE_EXIT']),
            updated_at_utc=now, expected_trials=1800, results=str(root))
temporary = path.with_suffix('.json.tmp')
temporary.write_text(json.dumps(data, indent=2)+'\n')
temporary.replace(path)
PY
}
paper_status running 0
trap 'paper_exit=$?; if ((paper_exit == 0)); then paper_status complete 0; else paper_status failed "$paper_exit"; fi' EXIT
bash "$paper_repo/workstation/dom-paper45.sh" full --resume-verified
