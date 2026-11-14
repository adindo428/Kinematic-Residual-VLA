#!/usr/bin/env python3
"""Queue only completed VLASH+KR and PI05 baseline after the existing local queue."""
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
from evaluation_training_policy import comparison_note, filter_comparison_rows, require_eligible

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / 'scratch/dom45'
ORIGINAL = WORK / 'evaluations/multivla20k-20261029'
ROOT = WORK / 'evaluations/completed-vlas-after-current-20261102'
TRANSFERS = WORK / 'training/palmetto-updated-20261102'
PREDECESSOR = WORK / 'evaluations/smolvla-runtime-repair-20261102'
PREDECESSOR_SERVICE = 'krvla-smolvla-runtime-repair-20261102.service'
ORDER = ['vlash_kr', 'pi05_baseline']
RUNS = {'vlash_kr': '9a9e2f23', 'pi05_baseline': '3dab4a55'}
spec = importlib.util.spec_from_file_location('completed_vla_suite', ORIGINAL / 'run_local.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
suite.ROOT = ROOT


def transfers_ready():
    for name in ORDER:
        path = TRANSFERS / name / 'transfer_receipt.json'
        if not path.exists():
            return False, 'Waiting for complete, verified transfer: ' + name
        if not suite.read(path)['evaluation_eligible']:
            return False, 'Transferred checkpoint is incomplete: ' + name
    return True, 'Both completed model transfers are verified'


def dependency_ready():
    for name in ['smolvla_baseline', 'smolvla_kr']:
        path = PREDECESSOR / 'results' / name / 'suite_status.json'
        if not path.exists() or suite.read(path).get('state') not in ('complete', 'failed'):
            return False, 'Waiting for ' + name + ' to finish'
    path = PREDECESSOR / 'status.json'
    if not path.exists() or suite.read(path).get('state') not in ('complete', 'finished_with_failures'):
        return False, 'Waiting for both SmolVLA evaluations to finish'
    active = subprocess.check_output(['systemctl', '--user', 'show', PREDECESSOR_SERVICE, '--property=ActiveState', '--value'], text=True).strip()
    if active in ('activating', 'active', 'deactivating', 'reloading'):
        return False, 'Waiting for the existing evaluation service to release its processes'
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        return False, 'Waiting for the compute GPUs to be free'
    return True, 'Predecessor evaluations have finished and GPUs are free'


def prepare():
    catalog = suite.read(ORIGINAL / 'catalog.json')
    catalog['models'] = {name: catalog['models'][name] for name in ORDER}
    catalog['held_models'] = ['vlash_baseline', 'pi05_kr']
    catalog['dependency'] = {'results': str(PREDECESSOR), 'service': PREDECESSOR_SERVICE,
                             'order': ['smolvla_baseline', 'smolvla_kr', *ORDER],
                             'domv100': 'cancelled by user; exceeds prior full-data epoch limit'}
    catalog['runtime_repairs'] = ['Batched instruction', 'Waiting RPC poll', 'VLASH internal tokenizer ID resolved to pinned local metadata']
    for path in (Path(__file__), REPO / 'workstation/completed_vla_table1_client.py', REPO / 'workstation/smolvla_table1_client.py',
                 REPO / 'workstation/evaluation_training_policy.py'):
        catalog['files_sha256'][str(path)] = suite.sha256(path)
    for name in ORDER:
        receipt = suite.read(TRANSFERS / name / 'transfer_receipt.json')
        assert receipt['evaluation_eligible'] and receipt['complete_epoch10']
        entry = catalog['models'][name]
        require_eligible(name, entry)
        assert receipt['identity'] == entry['identity']
        checkpoint = Path(receipt['checkpoint'])
        stat = checkpoint.stat()
        assert stat.st_size == receipt['bytes'] and stat.st_mtime_ns == receipt['mtime_ns']
        entry.update(checkpoint=str(checkpoint), checkpoint_bytes=stat.st_size, checkpoint_mtime_ns=stat.st_mtime_ns,
                     checkpoint_sha256=receipt['sha256'], sha256=receipt['sha256'], progress=receipt['progress'],
                     training_wandb_run=RUNS[name], transfer_receipt=str(TRANSFERS / name / 'transfer_receipt.json'),
                     selection='completed epoch-10 latest snapshot; validation-best selection unverified')
        root = ROOT / 'results' / name
        shutil.copytree(ORIGINAL / 'results' / name / 'test_lists', root / 'test_lists', dirs_exist_ok=True)
        protocol = suite.read(ORIGINAL / 'results/dom100/protocol.json')
        protocol.update(checkpoint=entry, method_label=entry['label'], policy_source=entry['code_root'],
                        evaluation_code_sha256=catalog['files_sha256'], runtime_repairs=catalog['runtime_repairs'])
        suite.write(root / 'protocol.json', protocol)
        if not (root / 'suite_status.json').exists():
            suite.write(root / 'suite_status.json', {'state': 'queued_after_current_evaluations'})
    directory = ROOT / 'workstation'
    directory.mkdir(parents=True, exist_ok=True)
    link = directory / 'multivla_table1_client.py'
    target = REPO / 'workstation/completed_vla_table1_client.py'
    if not link.exists():
        link.symlink_to(target)
    assert link.resolve() == target
    suite.write(ROOT / 'catalog.json', catalog)
    return catalog


def publish(catalog):
    rows = filter_comparison_rows(suite.read(PREDECESSOR / 'comparison.json')['rows'])
    progress = []
    for name in ORDER:
        root = ROOT / 'results' / name
        values, count = {}, 0
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
    lines = (PREDECESSOR / 'table1_comparison.md').read_text().splitlines()
    lines[0] = '# Table 1: 20k fine-tuning comparison with less than one prior full-data epoch'
    first = next(i for i, line in enumerate(lines) if line.startswith('| Method |'))
    last = first + 2
    while last < len(lines) and lines[last].startswith('|'):
        last += 1
    table = [lines[first], lines[first + 1]]
    for row in rows:
        table.append('| ' + row['Method'] + ' | ' + ' | '.join(f'{row[k]:.2f}' if k in row else 'pending' for k in fields[1:]) + ' |')
    lines = lines[:first] + table
    text = '\n'.join(lines) + ('\n\nSuccess is a percentage; Time includes scored successes and failures; pilot trials are excluded. '
                             'VLASH + KR and pi0.5 baseline use the newly transferred completed epoch-10 checkpoints. '
                             'VLASH baseline and pi0.5 + KR are held pending final training checkpoints at the user\'s request. '
                             'Each scored suite requires a new strict GPU probe and nine verified pilot trials.\n')
    text += '\n' + comparison_note() + '\n'
    (ROOT / 'table1_comparison.md').write_text(text)
    ide = WORK / 'results/krv100-best-epoch003-task-table1-20261025/table1_task_subset.md'
    temp = ide.with_suffix('.md.tmp')
    temp.write_text(text)
    temp.replace(ide)
    with (ROOT / 'table1_comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    suite.write(ROOT / 'comparison.json', {'rows': rows, 'progress': progress, 'held_models': catalog['held_models']})
    try:
        import wandb
        report = ROOT / 'reports' / ROOT.name
        report.mkdir(parents=True, exist_ok=True)
        for name in ('status.json', 'catalog.json', 'comparison.json', 'table1_comparison.md', 'table1_comparison.csv'):
            shutil.copy2(ROOT / name, report / name)
        for name in ORDER:
            for filename in ('summary.json', 'suite_status.json', 'pilot/policy_probe.json'):
                path = ROOT / 'results' / name / filename
                if path.exists():
                    shutil.copy2(path, report / f'{name}-{path.name}')
        api = wandb.Api(timeout=30)
        for rid in ('2qh0ijb1', *RUNS.values()):
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
    with (ROOT / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            ready, reason = transfers_ready()
            if ready:
                break
            suite.status('waiting_for_checkpoint_transfers', order=ORDER, reason=reason,
                         held_models=['vlash_baseline', 'pi05_kr'])
            if args.prepare_only:
                print(reason)
                return
            time.sleep(30)
        catalog = prepare()
        suite.status('queued_after_current_evaluations', order=ORDER, dependency=str(PREDECESSOR))
        if args.prepare_only:
            print({name: entry['progress'] for name, entry in catalog['models'].items()})
            return
        while True:
            ready, reason = dependency_ready()
            if ready:
                break
            suite.status('queued_after_current_evaluations', order=ORDER, dependency=str(PREDECESSOR), reason=reason)
            time.sleep(30)
        for path, expected in {**catalog['files_sha256'], **catalog['asset_sha256']}.items():
            assert suite.sha256(path) == expected, 'Frozen dependency changed: ' + path
        passed = []
        for name in ORDER:
            root = ROOT / 'results' / name
            try:
                require_eligible(name, catalog['models'][name])
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
                    require_eligible(name, catalog['models'][name])
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
