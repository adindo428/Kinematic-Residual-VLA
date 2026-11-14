#!/usr/bin/env python3
"""Queue completed SmolVLA ±KR snapshots after fresh KRV100 epoch-10 evaluation."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / 'scratch/dom45'
ORIGINAL = WORK / 'evaluations/multivla20k-20261029'
ROOT = WORK / 'evaluations/smolvla-after-krv100-20261101'
PREDECESSOR = WORK / 'evaluations/krv100-fresh20k-palmetto-table1-20261101'
PREDECESSOR_SERVICE = 'krvla-fresh20k-palmetto-table1-20261101.service'
ORDER = ['smolvla_baseline', 'smolvla_kr']
RUNS = {'smolvla_baseline': 'qabxg0en', 'smolvla_kr': '8npmrzpw'}
spec = importlib.util.spec_from_file_location('frozen_local_suite', ORIGINAL / 'run_local.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
suite.ROOT = ROOT


def prepare():
    catalog = suite.read(ORIGINAL / 'catalog.json')
    catalog['models'] = {name: catalog['models'][name] for name in ORDER}
    catalog['dependency'] = {'results': str(PREDECESSOR), 'service': PREDECESSOR_SERVICE,
                             'required_checkpoint_selection': 'latest', 'required_epoch': 10,
                             'required_state': 'complete', 'required_verified_trials': 360}
    catalog['inference_fix'] = ('Batch the RPC instruction and wait in libzmq poll between observations instead of busy-spinning. '
                                'This releases the Python interpreter for the inference worker. Weights, precision, 10 denoising steps, '
                                'image resolution, simulator timing, latency scaling and stale-action handling are unchanged.')
    for path in (Path(__file__), REPO / 'workstation/smolvla_table1_client.py'):
        catalog['files_sha256'][str(path)] = suite.sha256(path)
    for name in ORDER:
        entry = catalog['models'][name]
        checkpoint = Path(entry['checkpoint'])
        assert suite.sha256(checkpoint) == entry['checkpoint_sha256']
        pointer = suite.read(checkpoint.parent / 'latest.json')
        completed = suite.read(checkpoint.parent / 'completed.json')
        assert pointer['progress'] == completed['progress'] == entry['progress']
        assert entry['progress']['epoch'] == 10 and entry['progress']['updates'] == 396050
        assert pointer['identity'] == entry['identity']
        entry['training_wandb_run'] = RUNS[name]
        entry['training_completion_wandb_check'] = {'state': 'finished', 'optimizer_updates': 396050,
                                                   'samples': 25347200, 'subset_epoch': 10,
                                                   'evidence': 'W&B API checked in the scheduling session; final progress matches local metadata'}
        root = ROOT / 'results' / name
        shutil.copytree(ORIGINAL / 'results' / name / 'test_lists', root / 'test_lists', dirs_exist_ok=True)
        protocol = suite.read(ORIGINAL / 'results/dom100/protocol.json')
        protocol.update(checkpoint=entry, method_label=entry['label'],
                        policy_source=entry['code_root'], evaluation_code_sha256=catalog['files_sha256'],
                        inference_fix=catalog['inference_fix'])
        suite.write(root / 'protocol.json', protocol)
        if not (root / 'suite_status.json').exists():
            suite.write(root / 'suite_status.json', {'state': 'waiting_for_krv100_epoch10'})
    directory = ROOT / 'workstation'
    directory.mkdir(parents=True, exist_ok=True)
    link = directory / 'multivla_table1_client.py'
    target = REPO / 'workstation/smolvla_table1_client.py'
    if not link.exists():
        link.symlink_to(target)
    assert link.resolve() == target
    suite.write(ROOT / 'catalog.json', catalog)
    return catalog


def dependency_ready():
    pointer = PREDECESSOR / 'results/latest/suite_status.json'
    overall = PREDECESSOR / 'status.json'
    if not pointer.exists() or not overall.exists():
        return False, 'KRV100 evaluation status is not available'
    selected = suite.read(pointer)
    if selected.get('state') != 'complete' or selected.get('verified_trials') != 360:
        return False, 'Waiting for 360 verified KRV100 epoch-10 trials'
    if suite.read(overall).get('state') not in ('complete', 'finished_with_failures'):
        return False, 'Waiting for the KRV100 evaluation controller to finish'
    active = subprocess.check_output(['systemctl', '--user', 'show', PREDECESSOR_SERVICE, '--property=ActiveState', '--value'], text=True).strip()
    if active in ('activating', 'active', 'deactivating', 'reloading'):
        return False, 'Waiting for the KRV100 service to release its processes'
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        return False, 'Waiting for the compute GPUs to be free'
    summary = suite.read(PREDECESSOR / 'results/latest/summary.json')
    assert summary['complete'] and summary['verified_trials'] == 360
    assert summary['checkpoint']['progress']['epoch'] == 10
    return True, 'KRV100 epoch-10 evaluation completed and GPUs are free'


def publish(catalog):
    rows = suite.read(PREDECESSOR / 'comparison.json')['rows']
    progress = []
    for name in ORDER:
        root = ROOT / 'results' / name
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
        index = next(i for i, row in enumerate(rows) if row['Method'] == catalog['models'][name]['label'])
        rows[index] = {'Method': catalog['models'][name]['label'], **values}
        progress.append({'model': name, 'verified_trials': count, **suite.read(root / 'suite_status.json')})
    fields = ['Method', *suite.DIMS, 'Success', 'Time']
    text = (PREDECESSOR / 'table1_comparison.md').read_text()
    lines = text.splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith('| Method |'))
    last = first + 2
    while last < len(lines) and lines[last].startswith('|'):
        last += 1
    table = [lines[first], lines[first + 1]]
    for row in rows:
        table.append('| ' + row['Method'] + ' | ' + ' | '.join(f'{row[k]:.2f}' if k in row else 'pending' for k in fields[1:]) + ' |')
    lines[first:last] = table
    text = '\n'.join(lines).replace('Other transferred models use the previously supplied latest snapshots; their failed pilots still require repair.',
                                    'SmolVLA and SmolVLA + KR use the completed epoch-10 snapshots (396,050 updates); their current local evaluation state is recorded in the SmolVLA queue. Remaining transferred-model pilots still require repair.')
    text += ('\n\nSmolVLA instruction batching is corrected at the RPC boundary. The RPC waits in libzmq poll between observations '
             'to release Python execution time to the inference worker. Trained model weights, normalization, physical KR reconstruction, '
             '10 denoising steps, image resolution, simulator timing and stale-action handling are retained. '
             'Each model must pass nine separate pilot trials before its 360 scored trials.\n')
    (ROOT / 'table1_comparison.md').write_text(text)
    ide = WORK / 'results/krv100-best-epoch003-task-table1-20261025/table1_task_subset.md'
    temporary = ide.with_suffix('.md.tmp')
    temporary.write_text(text)
    temporary.replace(ide)
    with (ROOT / 'table1_comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    suite.write(ROOT / 'comparison.json', {'rows': rows, 'progress': progress})
    try:
        import wandb
        report = ROOT / 'reports' / ROOT.name
        report.mkdir(parents=True, exist_ok=True)
        for filename in ('status.json', 'catalog.json', 'comparison.json', 'table1_comparison.md', 'table1_comparison.csv'):
            shutil.copy2(ROOT / filename, report / filename)
        for name in ORDER:
            for filename in ('summary.json', 'suite_status.json', 'pilot/policy_probe.json'):
                source = ROOT / 'results' / name / filename
                if source.exists():
                    shutil.copy2(source, report / f'{name}-{source.name}')
        api = wandb.Api(timeout=30)
        for rid in ('2qh0ijb1', *RUNS.values()):
            run = api.run('aparame-clemson-university/Kinematic-Residual-VLA/' + rid)
            for path in report.iterdir():
                run.upload_file(str(path), root=str(ROOT))
        suite.write(ROOT / 'wandb_upload.json', {'state': 'uploaded', 'updated_utc': datetime.now(timezone.utc).isoformat()})
    except Exception as error:
        suite.write(ROOT / 'wandb_upload.json', {'state': 'retry_next_phase', 'error': str(error)})


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--results-root', type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.results_root.resolve()
    suite.ROOT = ROOT
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        catalog = prepare()
        suite.status('waiting_for_krv100_epoch10', order=ORDER, dependency=str(PREDECESSOR),
                     reason='360 verified epoch-10 trials, stopped predecessor service and free compute GPUs required')
        if args.prepare_only:
            print(json.dumps({'models': catalog['models'], 'dependency': catalog['dependency'], 'results': str(ROOT)}, indent=2))
            return
        while True:
            ready, reason = dependency_ready()
            if ready:
                break
            suite.status('waiting_for_krv100_epoch10', order=ORDER, dependency=str(PREDECESSOR), reason=reason)
            time.sleep(30)
        # Hash verification and all GPU use happen after the dependency gate.
        for path, expected in {**catalog['files_sha256'], **catalog['asset_sha256']}.items():
            assert suite.sha256(path) == expected, 'Frozen dependency changed: ' + path
        # Check both repaired pipelines before spending hours on scored trials.
        passed = []
        for name in ORDER:
            root = ROOT / 'results' / name
            try:
                suite.status('evaluating', model=name, phase='pilot')
                suite.write(root / 'suite_status.json', {'state': 'pilot'})
                publish(catalog)
                suite.phase(catalog, name, 'pilot', 0)
                suite.write(root / 'suite_status.json', {'state': 'pilot_passed', 'verified_pilot_trials': 9})
                passed.append(name)
            except Exception as error:
                suite.write(root / 'suite_status.json', {'state': 'failed', 'error': str(error),
                                                        'traceback': traceback.format_exc(), 'infrastructure_failure': True})
            publish(catalog)
        for name in passed:
            root = ROOT / 'results' / name
            try:
                for i, dim in enumerate(suite.DIMS):
                    suite.status('evaluating', model=name, phase='full', dimension=dim)
                    suite.write(root / 'suite_status.json', {'state': 'running', 'dimension': dim})
                    suite.phase(catalog, name, 'full', i)
                    publish(catalog)
                suite.summarize(catalog, name)
                suite.write(root / 'suite_status.json', {'state': 'complete', 'verified_trials': 360})
            except Exception as error:
                suite.write(root / 'suite_status.json', {'state': 'failed', 'error': str(error),
                                                        'traceback': traceback.format_exc(), 'infrastructure_failure': True})
            publish(catalog)
        failed = [name for name in ORDER if suite.read(ROOT / 'results' / name / 'suite_status.json')['state'] != 'complete']
        suite.status('complete' if not failed else 'finished_with_failures', failed_models=failed)
        publish(catalog)


if __name__ == '__main__':
    main()
