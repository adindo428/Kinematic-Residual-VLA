"""Queue the explicitly authorized epoch-3.208 KR snapshot after VLASH."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
LOW = REPO/'scratch/dom45/evaluations/low-clutter-full-table1-20261102'
ROOT = LOW/'pi05-kr-epoch003-partial-next-20261111'
PARENT = LOW/'vlash-grasp-gating-20261111'
RELEASE = REPO/'scratch/dom45/releases/kinematic-residual-vla-checkpoints-20261108'
RELATIVE = 'checkpoints/20k-continuation/pi05_kr/epoch-003-partial-next-u000301262/checkpoint.pt'
KEY = 'pi05_kr_epoch3_partial4'
os.environ['DOM_RESIDENT_ADAPTER_ROOT'] = str(ROOT)
import run_resident_adapter_table1 as runner


def prepare():
    low = runner.low
    if (ROOT/'plan.json').exists(): return low.read(ROOT/'plan.json')
    import torch
    ROOT.mkdir(parents=True, exist_ok=True)
    path = RELEASE/'staging'/RELATIVE; stat = path.stat()
    checksum = low.read(RELEASE/'hash_cache.private.json')[RELATIVE]
    assert (stat.st_size, stat.st_mtime_ns) == (checksum['bytes'], checksum['mtime_ns'])
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    progress, identity = saved['progress'], saved['identity']; del saved
    assert progress == {'updates': 301262, 'samples': 9640384, 'epoch': 3, 'cursor': 500000}
    catalog = low.read(LOW/'catalog.json'); entry = dict(catalog['models']['pi05_kr'])
    assert entry['identity'] == identity
    entry.update(checkpoint=str(path), weight_path=str(path), checkpoint_sha256=checksum['sha256'],
        sha256=checksum['sha256'], checkpoint_bytes=stat.st_size, weight_bytes=stat.st_size,
        checkpoint_mtime_ns=stat.st_mtime_ns, weight_mtime_ns=stat.st_mtime_ns,
        progress=progress, label='pi0.5 + KR (epoch 3 + partial 4; 3.208 epochs)',
        selection='Retained snapshot at 301262 updates; not exact epoch-3 validation-best weights',
        client_kind='resident_adapter', gate_mode='constant', user_authorized_latest_partial=True)
    spec = {'key': KEY, 'entry': entry, 'catalog': {k: catalog[k] for k in ['official_source', 'vlash_source']},
            'corrected_prompt_contract': False, 'grasp_gate': False}
    protocol = low.read(LOW/'protocol.json'); tests = {t['name']: t for t in protocol['tests']}
    for test in tests.values(): assert low.sha(test['path']) == test['config_sha256']
    jobs = []
    lists = {dim: [s for s in (LOW/'test_lists'/(dim+'.txt')).read_text().splitlines() if s] for dim in low.DIMS}
    for i in range(2):
        scene = lists['CR'][0]; key = f'pilot-{i}'
        jobs.append(dict(id=key, model=KEY, scene=scene, scene_sha256=tests[Path(scene).stem]['config_sha256'],
            repeats=1, repeat_offset=0, folder=str(ROOT/'blocks'/key), dimension='CR', scored=False))
    for pass_index, offset in [(1, 0), (2, 10)]:
        for dim in low.DIMS:
            for scene in lists[dim]:
                key = f'pass{pass_index}-{dim}-{Path(scene).stem}'
                jobs.append(dict(id=key, model=KEY, scene=scene, scene_sha256=tests[Path(scene).stem]['config_sha256'],
                    repeats=10, repeat_offset=offset, folder=str(ROOT/'blocks'/key), dimension=dim,
                    scored=True, pass_index=pass_index))
    for job in jobs:
        folder = Path(job['folder']); folder.mkdir(parents=True, exist_ok=True)
        (folder/'case.txt').write_text(job['scene']+'\n')
    files = [REPO/'workstation'/f for f in ['queue_pi05_kr_epoch3.py', 'run_resident_adapter_table1.py',
        'resident_adapter_table1.py', 'resident_adapter_table1_client.py', 'adapter_grasp_gating.py',
        'resident_low_clutter_server.py', 'resident_table1_server.py', 'resident_table1_worker.py',
        'smolvla_table1_client.py']]
    files += [p for p in Path(entry['code_root']).rglob('*') if p.is_file() and p.suffix in ['.py', '.json']]
    plan = dict(models=[spec], catalog=spec['catalog'], jobs=jobs,
        files_sha256={str(p): low.sha(p) for p in files}, protocol_sha256=low.sha(LOW/'protocol.json'),
        new_scored_trials=1800, success_history={KEY: []}, first_pass_repeats=10, remaining_repeats=10,
        automatic_second_pass_authorized=True, minimum_latency_retention=.9, created_utc=runner.stamp())
    low.write(ROOT/'plan.json', plan)
    low.write(ROOT/'existing.json', {'records': {KEY: []}, 'sources': []})
    low.write(ROOT/'checkpoint_policy.json', {'minimum_completed_epochs': 10, 'held_models': {},
        'authorized_exceptions': {KEY: {'user_authorized': True, 'checkpoint_sha256': checksum['sha256'],
        'reason': 'User explicitly requested the retained epoch-3.208 checkpoint on 2026-11-11'}}})
    low.write(ROOT/'request.json', {'enabled': True, 'user_authorized': True, 'models': [KEY],
        'wait_for_completion': str(PARENT/'completion.json'), 'notify_on_completion': True,
        'checkpoint': entry, 'gate_mode': 'constant', 'target_per_model': 1800})
    low.write(ROOT/'queue_status.json', {'state': 'waiting_for_vlash_comparison', 'updated_utc': runner.stamp()})
    text = (LOW/'table1_comparison.md').read_text()
    if 'Queued π0.5+KR follow-up:' not in text:
        with (LOW/'table1_comparison.md').open('a') as f:
            f.write('\nQueued π0.5+KR follow-up: epoch 3 + partial epoch 4 (3.208 epochs), original ungated reconstruction; '
                    '1,800 low-clutter trials after the current VLASH comparison. Exact epoch-3 weights were not found.\n')
    return plan


def parent_records(validate=False):
    previous = runner.low.read(PARENT/'plan.json')
    old_root, old_models = runner.ROOT, list(runner.MODELS)
    runner.ROOT = PARENT
    runner.MODELS[:] = [m['key'] for m in previous['models']]
    try:
        return runner.collect(previous, validate=validate)
    finally:
        runner.ROOT = old_root
        runner.MODELS[:] = old_models


def first_pass_ready(records, tests):
    expected = {'vlash_baseline_fixed', 'vlash_kr_fixed', 'vlash_kr_gated'}
    if set(records) != expected:
        return False
    # Require all 90 scenes, repeats 1–10, rather than an arbitrary 900 trials.
    scenes = {t['name'] for t in tests}
    return all(len(rr) == 900 and {r['env_name'] for r in rr} == scenes
               and all({r['trial'] for r in rr if r['env_name'] == scene} == set(range(1, 11))
                       for scene in scenes) for rr in records.values())


def wait_and_run(plan):
    low = runner.low
    completion = PARENT/'completion.json'
    tests = low.read(LOW/'protocol.json')['tests']
    while True:
        request = low.read(ROOT/'request.json')
        first_pass = request.get('handoff_after_trials_per_model') == 900
        if first_pass:
            records = parent_records()
            if first_pass_ready(records, tests):
                # The on-disk server schedule excludes deferred pass-two jobs.
                # The already loaded client cannot begin a second-pass trial.
                parent_plan = low.read(PARENT/'plan.json')
                assert parent_plan.get('deferred_jobs')
                assert not any(j.get('pass_index') == 2 for j in parent_plan['jobs'])
                assert first_pass_ready(parent_records(validate=True), tests)
                subprocess.run(['systemctl', '--user', 'stop',
                                'krvla-vlash-gated-comparison-20261111.service'], check=True)
                # Restore the original schedule for a later second-pass resume.
                parent_plan['jobs'] += parent_plan.pop('deferred_jobs')
                parent_plan['second_pass_deferred_for'] = KEY
                low.write(PARENT/'plan.json', parent_plan)
                low.write(PARENT/'first_pass_handoff.json', {
                    'verified': True, 'trials_saved_by_model': {k: len(v) for k, v in records.items()},
                    'remaining_trials_per_model': 900, 'next_model': KEY, 'updated_utc': runner.stamp()})
                low.write(PARENT/'execution/status.json', {
                    'stage': 'deferred_after_first_pass', 'updated_utc': runner.stamp(), 'next_model': KEY})
                break
        if completion.exists() and low.read(completion).get('complete'): break
        low.write(ROOT/'queue_status.json', {
            'state': 'waiting_for_vlash_first_900' if first_pass else 'waiting_for_vlash_comparison',
            'handoff_trials_per_model': 900 if first_pass else 1800,
            'verified_block_trials_by_model': {k: len(v) for k, v in records.items()} if first_pass else {},
            'updated_utc': runner.stamp()})
        time.sleep(30)
    with (LOW/'queue.lock').open('a') as lock:
        # Parent writes completion before its final cleanup; wait for its lock.
        fcntl.flock(lock, fcntl.LOCK_EX)
        if first_pass:
            assert low.read(PARENT/'first_pass_handoff.json')['verified']
            assert first_pass_ready(parent_records(validate=True), tests)
        else:
            assert low.read(completion)['trials_saved_by_model'] == {
                'vlash_baseline_fixed': 1800, 'vlash_kr_fixed': 1800, 'vlash_kr_gated': 1800}
        for path, digest in plan['files_sha256'].items(): assert low.sha(path) == digest, path
        held = low.read(PARENT/'held_records.json')
        previous = low.read(PARENT/'plan.json'); runner.ROOT = PARENT
        runner.MODELS[:] = [m['key'] for m in previous['models']]
        try: held.update(runner.collect(previous, validate=True))
        finally: runner.ROOT = ROOT
        low.write(ROOT/'held_records.json', held)
        catalog = low.read(LOW/'catalog.json'); entry = plan['models'][0]['entry']
        catalog['models'][KEY] = entry
        if KEY not in catalog['order']: catalog['order'].append(KEY)
        catalog.update(enabled_models=[KEY], queue_order=[KEY], active_resident_adapter_root=str(ROOT))
        low.write(LOW/'catalog.json', catalog)
        result = LOW/'results'/KEY; result.mkdir(parents=True, exist_ok=True)
        low.write(result/'protocol.json', {**low.read(LOW/'protocol.json'), 'checkpoint': entry})
        low.write(ROOT/'execution/status.json', {'stage': 'preparing', 'updated_utc': runner.stamp()})
        low.write(ROOT/'queue_status.json', {'state': 'starting', 'updated_utc': runner.stamp()})
    runner.MODELS[:] = [KEY]
    runner.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(); plan = prepare()
    print(json.dumps({'queued_model': KEY, 'progress': plan['models'][0]['entry']['progress'],
                      'dependency': str(PARENT/'completion.json')}), flush=True)
    if not args.prepare_only: wait_and_run(plan)
