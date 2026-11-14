#!/usr/bin/env python3
"""Repair and queue native DOM+velocity after both current SmolVLA evaluations."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import importlib.util
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from evaluation_training_policy import require_eligible

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / 'scratch/dom45'
ORIGINAL = WORK / 'evaluations/multivla20k-20261029'
ROOT = WORK / 'evaluations/domv-after-smolvla-20261102'
PREDECESSOR = WORK / 'evaluations/smolvla-runtime-repair-20261102'
PREDECESSOR_SERVICE = 'krvla-smolvla-runtime-repair-20261102.service'
SMOL_MODELS = ['smolvla_baseline', 'smolvla_kr']
MODEL = 'domv100'
spec = importlib.util.spec_from_file_location('frozen_domv_suite', ORIGINAL / 'run_local.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
suite.ROOT = ROOT


def prepare():
    catalog = suite.read(ORIGINAL / 'catalog.json')
    entry = catalog['models'][MODEL]
    catalog['models'] = {MODEL: entry}
    checkpoint = Path(entry['checkpoint'])
    assert suite.sha256(checkpoint) == entry['checkpoint_sha256']
    pointer = suite.read(checkpoint.parent / 'latest.json')
    completed = suite.read(checkpoint.parent / 'completed.json')
    assert pointer['identity'] == entry['identity']
    assert pointer['progress'] == completed['progress'] == entry['progress']
    assert entry['progress']['epoch'] == 10 and entry['progress']['updates'] == 286838
    config = suite.read(Path(entry['code_root']) / 'parents/domv100/config.json')
    assert config['condition_on_object_velocity'] and not config['enable_residual_flow']
    assert config['n_obs_steps'] == 2 and config['chunk_size'] == config['n_action_steps'] == 20
    entry['training_wandb_run'] = 'bwx4b5i1'
    catalog['dependency'] = {'results': str(PREDECESSOR), 'service': PREDECESSOR_SERVICE,
                             'models': SMOL_MODELS, 'required_state': 'both terminal; service stopped; GPUs free'}
    catalog['inference_fix'] = ('Use the tested libzmq polling wait and batched RPC instruction with the frozen native DOM policy. '
                                'Preserve two-frame [-2,0] inputs, current object pose/velocity, trained normalization, '
                                'BF16, 10 inference steps and native latency scaling exactly once. Residual flow remains disabled.')
    for path in (Path(__file__), REPO / 'workstation/domv_table1_client.py', REPO / 'workstation/smolvla_table1_client.py'):
        catalog['files_sha256'][str(path)] = suite.sha256(path)
    root = ROOT / 'results' / MODEL
    shutil.copytree(ORIGINAL / 'results' / MODEL / 'test_lists', root / 'test_lists', dirs_exist_ok=True)
    protocol = suite.read(ORIGINAL / 'results/dom100/protocol.json')
    protocol.update(checkpoint=entry, method_label='DOM + velocity', policy_source=catalog['dom_source'],
                    evaluation_code_sha256=catalog['files_sha256'], inference_fix=catalog['inference_fix'],
                    velocity_source='Current simulator target-object pose and velocity only; no future state',
                    residual_flow=False, observation_indices=[-2, 0])
    suite.write(root / 'protocol.json', protocol)
    if not (root / 'suite_status.json').exists():
        suite.write(root / 'suite_status.json', {'state': 'waiting_for_smolvla_evaluations'})
    directory = ROOT / 'workstation'
    directory.mkdir(parents=True, exist_ok=True)
    link = directory / 'multivla_table1_client.py'
    target = REPO / 'workstation/domv_table1_client.py'
    if not link.exists():
        link.symlink_to(target)
    assert link.resolve() == target
    suite.write(ROOT / 'catalog.json', catalog)
    return catalog


def dependency_ready():
    for name in SMOL_MODELS:
        path = PREDECESSOR / 'results' / name / 'suite_status.json'
        if not path.exists():
            return False, 'Waiting for ' + name
        state = suite.read(path)
        if state.get('state') not in ('complete', 'failed'):
            return False, 'Waiting for ' + name + ' to finish'
        if state['state'] == 'complete':
            summary = suite.read(path.parent / 'summary.json')
            assert state['verified_trials'] == summary['verified_trials'] == 360 and summary['complete']
    overall = PREDECESSOR / 'status.json'
    if not overall.exists() or suite.read(overall).get('state') not in ('complete', 'finished_with_failures'):
        return False, 'Waiting for the SmolVLA controller to finish'
    active = subprocess.check_output(['systemctl', '--user', 'show', PREDECESSOR_SERVICE, '--property=ActiveState', '--value'], text=True).strip()
    if active in ('activating', 'active', 'deactivating', 'reloading'):
        return False, 'Waiting for the SmolVLA service to release its processes'
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        return False, 'Waiting for the compute GPUs to be free'
    return True, 'Both SmolVLA evaluations have finished and GPUs are free'


def publish(catalog):
    rows = suite.read(PREDECESSOR / 'comparison.json')['rows']
    root = ROOT / 'results' / MODEL
    values = {}
    count = 0
    for i, dim in enumerate(suite.DIMS):
        folder = root / 'full' / f'{i:02d}_{dim}'
        if (folder / 'verification.json').exists():
            verified = suite.verify_folder(folder, root, root / 'test_lists' / f'{i:02d}_{dim}.txt', 20)
            assert verified['trials'] == 40
            values[dim] = verified['success_percent']
            count += 40
    if (root / 'summary.json').exists() and count == 360:
        values.update(suite.read(root / 'summary.json')['observed'])
    index = next(i for i, row in enumerate(rows) if row['Method'] == 'DOM + velocity')
    rows[index] = {'Method': 'DOM + velocity', **values}
    fields = ['Method', *suite.DIMS, 'Success', 'Time']
    lines = (PREDECESSOR / 'table1_comparison.md').read_text().splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith('| Method |'))
    last = first + 2
    while last < len(lines) and lines[last].startswith('|'):
        last += 1
    table = [lines[first], lines[first + 1]]
    for row in rows:
        table.append('| ' + row['Method'] + ' | ' + ' | '.join(f'{row[k]:.2f}' if k in row else 'pending' for k in fields[1:]) + ' |')
    lines[first:last] = table
    text = '\n'.join(lines)
    text += ('\n\nDOM + velocity uses the completed epoch-10 checkpoint (286,838 updates) from W&B bwx4b5i1, '
             'with native velocity conditioning and residual flow disabled. Its RPC waits between observations to avoid starving '
             'the inference thread. Trained normalization, two-frame inputs, simulator timing and latency scaling are retained. '
             'Nine pilot trials must verify before the 360 scored trials.\n')
    (ROOT / 'table1_comparison.md').write_text(text)
    ide = WORK / 'results/krv100-best-epoch003-task-table1-20261025/table1_task_subset.md'
    temporary = ide.with_suffix('.md.tmp')
    temporary.write_text(text)
    temporary.replace(ide)
    with (ROOT / 'table1_comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    suite.write(ROOT / 'comparison.json', {'rows': rows, 'progress': {'model': MODEL, 'verified_trials': count,
                                                                  **suite.read(root / 'suite_status.json')}})
    try:
        import wandb
        report = ROOT / 'reports' / ROOT.name
        report.mkdir(parents=True, exist_ok=True)
        for filename in ('status.json', 'catalog.json', 'comparison.json', 'table1_comparison.md', 'table1_comparison.csv'):
            shutil.copy2(ROOT / filename, report / filename)
        for filename in ('summary.json', 'suite_status.json', 'pilot/policy_probe.json'):
            source = root / filename
            if source.exists():
                shutil.copy2(source, report / f'domv100-{source.name}')
        api = wandb.Api(timeout=30)
        for rid in ('2qh0ijb1', 'bwx4b5i1'):
            run = api.run('aparame-clemson-university/Kinematic-Residual-VLA/' + rid)
            for path in report.iterdir():
                run.upload_file(str(path), root=str(ROOT))
        suite.write(ROOT / 'wandb_upload.json', {'state': 'uploaded', 'updated_utc': datetime.now(timezone.utc).isoformat()})
    except Exception as error:
        suite.write(ROOT / 'wandb_upload.json', {'state': 'retry_next_phase', 'error': str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    try:
        require_eligible(MODEL)
    except ValueError as error:
        suite.status('excluded_by_training_policy', model=MODEL, reason=str(error))
        print(str(error), flush=True)
        return
    with (ROOT / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        catalog = prepare()
        suite.status('waiting_for_smolvla_evaluations', dependency=str(PREDECESSOR), model=MODEL)
        if args.prepare_only:
            print(catalog['models'][MODEL])
            return
        while True:
            ready, reason = dependency_ready()
            if ready:
                break
            suite.status('waiting_for_smolvla_evaluations', dependency=str(PREDECESSOR), model=MODEL, reason=reason)
            time.sleep(30)
        # No GPU allocation or model probe is permitted before this gate.
        for path, expected in {**catalog['files_sha256'], **catalog['asset_sha256']}.items():
            assert suite.sha256(path) == expected, 'Frozen dependency changed: ' + path
        root = ROOT / 'results' / MODEL
        try:
            suite.status('evaluating', model=MODEL, phase='pilot')
            suite.write(root / 'suite_status.json', {'state': 'pilot'})
            publish(catalog)
            suite.phase(catalog, MODEL, 'pilot', 0)
            for i, dim in enumerate(suite.DIMS):
                suite.status('evaluating', model=MODEL, phase='full', dimension=dim)
                suite.write(root / 'suite_status.json', {'state': 'running', 'dimension': dim})
                suite.phase(catalog, MODEL, 'full', i)
                publish(catalog)
            suite.summarize(catalog, MODEL)
            suite.write(root / 'suite_status.json', {'state': 'complete', 'verified_trials': 360})
            suite.status('complete', model=MODEL, verified_trials=360)
        except Exception as error:
            suite.write(root / 'suite_status.json', {'state': 'failed', 'error': str(error),
                                                    'traceback': traceback.format_exc(), 'infrastructure_failure': True})
            suite.status('failed', model=MODEL, error=str(error))
        publish(catalog)


if __name__ == '__main__':
    main()
