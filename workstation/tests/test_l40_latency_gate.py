"""No scored deployment is allowed when either latency percentile fails."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l40_latency_gate import check_dummy


@pytest.mark.parametrize('p50,p95,passed', [(300., 320., True), (340., 320., False), (300., 400., False)])
def test_a6000_limits_are_enforced_before_scoring(tmp_path, p50, p95, passed):
    plan = {'models': [{'key': 'pi'}], 'a6000_reference': {
        'dummy': {'p50_ms': 300., 'p95_ms': 330.}, 'scope': 'same architecture'}}
    profile = {'paired_trials': [{'models': {'pi': {'single': {'p50_ms': p50, 'p95_ms': p95}}}}]}
    if passed:
        check_dummy(plan, profile, tmp_path)
    else:
        with pytest.raises(RuntimeError): check_dummy(plan, profile, tmp_path)
    receipt = json.loads((tmp_path/'l40_dummy_qualification.json').read_text())
    assert receipt['passed'] is passed and receipt['scored'] is False
