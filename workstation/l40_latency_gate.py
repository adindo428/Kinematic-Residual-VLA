"""Fail closed before scoring if a one-L40 deployment exceeds A6000 limits."""
import json
from pathlib import Path


def check_dummy(plan, profile, output):
    if 'a6000_reference' not in plan: return
    reference = plan['a6000_reference']['dummy']
    actual = profile['paired_trials'][-1]['models'][plan['models'][0]['key']]['single']
    passed = all(actual[k] <= reference[k]/.9 for k in ['p50_ms', 'p95_ms'])
    receipt = {'passed': passed, 'scored': False, 'a6000': reference, 'l40': actual,
               'minimum_retention': .9, 'comparison_scope': plan['a6000_reference']['scope']}
    (Path(output)/'l40_dummy_qualification.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if not passed: raise RuntimeError('L40 policy latency exceeds A6000 limits; no scored trials permitted')


def check_pilots(plan, output):
    if 'a6000_reference' not in plan: return
    from monitor_table1_latency import log_samples, coverage
    from capture_table1_runtime import stats
    rows = []
    for scene, reference in plan['a6000_reference']['cases'].items():
        folders = [Path(j['folder']) for j in plan['jobs'] if not j['scored'] and j['scene'] == scene]
        samples = [r for f in folders for r in log_samples(f/'client.log')]
        actual = stats([r['ms'] for r in samples])
        stale = [r['stale'] for r in samples if r['stale'] is not None]
        expired = sum(v >= 20 for v in stale)/len(stale) if stale else 1.
        passed = len(samples) >= 30 and all(actual[k] <= reference[k]/.9 for k in ['p50_ms', 'p95_ms'])
        passed = passed and expired <= reference['expired_chunk_fraction']+.02
        rows.append({'scene': scene, 'reference': reference, 'l40': actual,
                     'expired_chunk_fraction': expired, 'passed': passed})
    receipt = {'passed': all(r['passed'] for r in rows), 'scored': False, 'cases': rows,
               'minimum_retention': .9, 'simulator': '4.5.0', 'isaaclab': '2.2.1'}
    (Path(output)/'l40_streaming_qualification.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if not receipt['passed']: raise RuntimeError('One-L40 simulation/inference pilots exceed A6000 timing limits; scoring held')
    (Path(output)/'l40_scoring_authorized.json').write_text(json.dumps(receipt, indent=2)+'\n')
