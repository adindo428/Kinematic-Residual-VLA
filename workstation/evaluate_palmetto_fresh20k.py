#!/usr/bin/env python3
"""Evaluate transferred fresh KRV100 best/latest on the fixed local task suite."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / 'scratch/dom45'
TRANSFER = WORK / 'training/krv100-fresh20k-h200-20261029'
TRAIN = TRANSFER / 'family/runs/krv100'
SOURCE = TRANSFER / 'source'
RPC_SOURCE = WORK / 'training/krv100-full-20261010/source'
ROOT = WORK / 'evaluations/krv100-fresh20k-palmetto-table1-20261101'
ORIGINAL = WORK / 'evaluations/multivla20k-20261029'
IDE_TABLE = WORK / 'results/krv100-best-epoch003-task-table1-20261025/table1_task_subset.md'
DIMS = ['CR', 'DA', 'LS', 'VU', 'SR', 'MP', 'VG', 'MG', 'DR']
ORDER = ['best', 'latest']
sys.path.insert(0, str(REPO / 'evaluation/official_dom/scripts'))
from report import read_records, verify_folder
from run_suite import stop_group


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def status(state, **details):
    write(ROOT / 'status.json', {'state': state, 'updated_utc': datetime.now(timezone.utc).isoformat(), **details})


def prepare():
    receipt = read(TRANSFER / 'download_move_receipt.json')
    assert receipt['sha256_verified'] and receipt['downloads_copy_deleted']
    hashes = {item['path']: item['sha256'] for item in receipt['files']}
    family = TRANSFER / 'family'
    manifest = read(family / 'manifest.json')
    assert manifest['fingerprint'] == digest({k: v for k, v in manifest.items() if k != 'fingerprint'})
    artifact_sources = {}
    for relative, expected in manifest['artifacts'].items():
        artifact = family / relative
        if relative == 'initialization/config.json' and not artifact.exists():
            artifact = WORK / 'training/fresh20k-20261025/family' / relative
        assert sha256(artifact) == expected, relative
        artifact_sources[relative] = str(artifact)
    source_hashes = {str(p.relative_to(SOURCE)): sha256(p) for sub in ('core', 'utils', 'policies', 'proactive_vla', 'scripts')
                     for p in sorted((SOURCE / sub).rglob('*.py'))}
    assert digest(source_hashes) == manifest['source_sha256'], 'Transferred training source digest mismatch'
    provenance = read(TRAIN / 'provenance.json')
    assert provenance['wandb_id'] == 'nj3zd8q7' and provenance['variant'] == 'krv100'
    assert provenance['manifest'] == manifest['fingerprint']
    local_manifest = read(WORK / 'training/fresh20k-20261025/family/manifest.json')
    assert manifest['split_sha256'] == local_manifest['split_sha256'], 'Unexpected training split'
    assert manifest['weights_sha256'] == local_manifest['weights_sha256'], 'Unexpected released initialization'
    completed = read(TRAIN / 'training_complete.json')
    assert completed['epoch'] == 10 and completed['updates'] == 68380
    template = ORIGINAL / 'results/dom100'
    catalog = {'models': {}, 'asset_sha256': read(ORIGINAL / 'catalog.json')['asset_sha256'],
               'files_sha256': read(ORIGINAL / 'catalog.json')['files_sha256'],
               'training_source_sha256': manifest['source_sha256'], 'training_manifest': manifest['fingerprint'],
               'verified_training_artifact_sources': artifact_sources,
               'wandb_run': 'aparame-clemson-university/Kinematic-Residual-VLA/nj3zd8q7',
               'policy_source': str(SOURCE), 'rpc_source': str(RPC_SOURCE), 'downloads_receipt': str(TRANSFER / 'download_move_receipt.json')}
    for path in [*SOURCE.rglob('*.py'), Path(__file__), REPO / 'workstation/inference_fresh20k_checkpoint.py',
                 REPO / 'workstation/krvla_table1_probe.py', RPC_SOURCE / 'scripts/inference.py']:
        catalog['files_sha256'][str(path)] = sha256(path)
    validation = {int(p.stem.split('-')[-1]): read(p)['h20/action_mse'] for p in TRAIN.glob('validation-epoch-*.json')}
    for selection in ORDER:
        pointer = read(TRAIN / (selection + '.json'))
        assert pointer['fingerprint'] == provenance['fingerprint']
        generation = TRAIN / 'generations' / pointer['generation']
        root = ROOT / 'results' / selection
        checkpoint = root / 'checkpoint'
        checkpoint.mkdir(parents=True, exist_ok=True)
        for filename in ('model.safetensors', 'config.json', 'provenance.json'):
            original = generation / filename
            assert sha256(original) == hashes[str(original.relative_to(TRANSFER))], 'Transferred checkpoint changed'
            dest = checkpoint / filename
            if not dest.exists():
                os.link(original, dest)
            assert sha256(dest) == sha256(original)
        config = read(checkpoint / 'config.json')
        assert config['wandb_id'] == 'nj3zd8q7' and config['experiment_fingerprint'] == provenance['fingerprint']
        assert config['epoch'] == pointer['progress']['epoch']
        assert config['enable_residual_flow'] and config['condition_on_object_velocity']
        if selection == 'best':
            assert validation[config['epoch']] == min(validation.values()), 'Best pointer is not validation minimum'
        label = f"KRV100 · fresh 20k ({'validation-best' if selection == 'best' else 'latest'}, epoch {config['epoch']})"
        entry = {'label': label, 'selection': selection, 'checkpoint': str(checkpoint), 'generation': str(generation),
                 'checkpoint_sha256': sha256(checkpoint / 'model.safetensors'), 'config_sha256': sha256(checkpoint / 'config.json'),
                 'progress': pointer['progress'], 'validation_action_mse': validation[config['epoch']], 'wandb_id': 'nj3zd8q7'}
        catalog['models'][selection] = entry
        shutil.copytree(template / 'test_lists', root / 'test_lists', dirs_exist_ok=True)
        protocol = read(template / 'protocol.json')
        protocol.update(checkpoint=entry, method_label=label, checkpoint_kind='locally_trained',
                        policy_source=str(SOURCE), policy_source_sha256=manifest['source_sha256'],
                        evaluation_code_sha256=catalog['files_sha256'],
                        velocity_source='Current simulator target-object pose and linear velocity; no future observations')
        write(root / 'protocol.json', protocol)
        if not (root / 'suite_status.json').exists():
            write(root / 'suite_status.json', {'state': 'queued'})
    write(ROOT / 'catalog.json', catalog)
    return catalog


def validate(catalog):
    for path, expected in {**catalog['files_sha256'], **catalog['asset_sha256']}.items():
        assert sha256(path) == expected, 'Frozen dependency changed: ' + path
    for entry in catalog['models'].values():
        assert sha256(Path(entry['checkpoint']) / 'model.safetensors') == entry['checkpoint_sha256']
        assert sha256(Path(entry['checkpoint']) / 'config.json') == entry['config_sha256']


def phase(catalog, selection, mode, index=0):
    root = ROOT / 'results' / selection
    folder = root / 'pilot' if mode == 'pilot' else root / 'full' / f'{index:02d}_{DIMS[index]}'
    listing = root / 'test_lists' / ('pilot.txt' if mode == 'pilot' else f'{index:02d}_{DIMS[index]}.txt')
    repeats = 1 if mode == 'pilot' else 20
    if folder.exists():
        if (folder / 'verification.json').exists():
            return verify_folder(folder, root, listing, repeats)
        raise FileExistsError('Partial results retained for diagnosis: ' + str(folder))
    if mode != 'pilot':
        assert read(root / 'pilot_verified.json')['verified']
    for test in read(root / 'protocol.json')['tests']:
        assert sha256(test['path']) == test['config_sha256']
    validate(catalog)
    folder.mkdir(parents=True)
    entry = catalog['models'][selection]
    env = dict(os.environ, DOM_OFFICIAL_SOURCE=str(WORK / 'official/source'), DOM_TRIAL_RESULTS=str(folder / 'trial_results.jsonl'),
               DOM_RECORD_RUNTIME='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    client_env = dict(env, CUDA_VISIBLE_DEVICES='1')
    server_env = dict(env, CUDA_VISIBLE_DEVICES='0')
    if mode == 'pilot':
        probe = [str(WORK / 'envs/train/bin/python'), '-u', str(REPO / 'workstation/krvla_table1_probe.py'),
                 '--source', str(SOURCE), '--checkpoint', entry['checkpoint'], '--output', str(root / 'policy_probe.json')]
        with (folder / 'probe.log').open('w') as log:
            subprocess.run(probe, env=client_env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
    client = [str(WORK / 'envs/train/bin/python'), '-u', str(REPO / 'workstation/inference_fresh20k_checkpoint.py'),
              '--source', str(SOURCE), '--rpc-source', str(RPC_SOURCE), '-p', entry['checkpoint'],
              '-a', f'KRV100_fresh20k_{selection}', '-r', 'euler', '-d', '-s', '--img_port', '27500', '--act_port', '27501',
              '-o', str(folder / 'actions')]
    assets = WORK / 'assets'
    server = [str(REPO / 'workstation/isaaclab-paper45.sh'), '-p', str(REPO / 'workstation/krvla_table1_server.py'),
              '--headless', '--enable_cameras', '--device', 'cuda:0', '--num_envs', '1', '--scene_dir', str(assets / 'scenes'),
              '--object_dir', str(assets / 'objects'), '--env_cfg', str(listing), '--output_dir', str(folder / 'videos'),
              '-n', str(repeats), '--save', '--img_port', '27500', '--act_port', '27501', '--kit_args', os.environ['KRVLA_KIT_ARGS']]
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True)
    assert len(gpu.splitlines()) == 2 and all('RTX A6000' in line for line in gpu.splitlines())
    write(folder / 'runtime.json', {'gpu': gpu, 'sim_gpu': 0, 'policy_gpu': 1, 'isaac_sim': '4.5.0', 'isaac_lab': '2.2.1',
                                  'repeats': repeats, 'streaming': True})
    write(folder / 'commands.json', {'server': server, 'client': client})
    processes = []
    try:
        with (folder / 'server.log').open('w') as slog, (folder / 'client.log').open('w') as clog:
            processes.append(subprocess.Popen(server, env=server_env, stdout=slog, stderr=subprocess.STDOUT, start_new_session=True))
            processes.append(subprocess.Popen(client, env=client_env, stdout=clog, stderr=subprocess.STDOUT, start_new_session=True))
            changed, size = time.monotonic(), -1
            while processes[1].poll() is None:
                if processes[0].poll() is not None:
                    raise RuntimeError('Simulator exited; see server.log')
                path = folder / 'trial_results.jsonl'
                new = path.stat().st_size if path.exists() else 0
                if new != size:
                    changed, size = time.monotonic(), new
                if time.monotonic() - changed > 1200:
                    raise RuntimeError('No completed trial for 20 minutes')
                time.sleep(5)
            if processes[1].returncode:
                raise RuntimeError('Policy exited with an error; see client.log')
        verified = verify_folder(folder, root, listing, repeats)
        if mode == 'pilot':
            write(root / 'pilot_verified.json', verified)
        return verified
    except BaseException as error:
        write(folder / 'failure.json', {'error': str(error), 'infrastructure_failure': True})
        raise
    finally:
        for process in reversed(processes):
            stop_group(process)


def publish(catalog, upload=True):
    previous = read(ORIGINAL / 'comparison.json')
    rows = previous['rows']
    fields = ['Method', *DIMS, 'Success', 'Time']
    dom_latest = WORK / 'evaluations/dom-latest-table1-20261031/results/dom100/summary.json'
    if dom_latest.exists() and read(dom_latest).get('complete'):
        rows.append({'Method': 'DOM · fresh 20k (latest, epoch 10)', **read(dom_latest)['observed']})
    progress = []
    for selection in ORDER:
        root = ROOT / 'results' / selection
        entry = catalog['models'][selection]
        values, records = {}, []
        for i, dim in enumerate(DIMS):
            folder = root / 'full' / f'{i:02d}_{dim}'
            if (folder / 'verification.json').exists():
                verified = verify_folder(folder, root, root / 'test_lists' / f'{i:02d}_{dim}.txt', 20)
                assert verified['trials'] == 40
                values[dim] = verified['success_percent']
                records.extend(read_records(folder))
        if len(records) == 360:
            values.update(Success=100 * statistics.mean(r['success'] for r in records), Time=statistics.mean(r['task_time_s'] for r in records))
            write(root / 'summary.json', {'complete': True, 'verified_trials': 360, 'observed': values,
                                          'checkpoint': entry, 'protocol': read(root / 'protocol.json')})
        rows.append({'Method': entry['label'], **values})
        progress.append({'selection': selection, 'verified_trials': len(records), **read(root / 'suite_status.json')})
    lines = ['# Table 1: task-aligned VLA comparison', '',
             'Same 18 custom configurations × 20 repeats = 360 scored trials per checkpoint, plus nine separate pilot trials. Local Isaac Sim 4.5.0 / Isaac Lab 2.2.1, two RTX A6000. This is not the official 1,800-trial Table 1.', '',
             '| ' + ' | '.join(fields) + ' |', '|' + '|'.join(['---'] + ['---:'] * 11) + '|']
    for row in rows:
        lines.append('| ' + row['Method'] + ' | ' + ' | '.join(f'{row[k]:.2f}' if k in row else 'pending' for k in fields[1:]) + ' |')
    lines += ['', 'Success is a percentage; Time is mean simulated seconds across all scored successes and failures. Pilots are excluded. Pending results are never scored as zero.', '',
              'Fresh KRV100 comes from Palmetto run fresh20k-palmetto-krv100-s42 (nj3zd8q7): released initialization, exact 20k split, seed 42. Validation-best is epoch 5; latest is epoch 10. Both are evaluated locally on the same fixed task subset. Checkpoint identities and progress are recorded in catalog.json.', '',
              'The two older KRV100 rows belong to the previous run with two preceding full-data fine-tuning epochs. Other transferred models use the previously supplied latest snapshots; their failed pilots still require repair. Their validation-best selection is not asserted.', '',
              'Task families overlap the 20k subset, while VG, MG and DR include generalization/stress conditions. This suite is not entirely in distribution. Training budgets and prior exposure differ.', '']
    content = '\n'.join(lines)
    (ROOT / 'table1_comparison.md').write_text(content)
    temp = IDE_TABLE.with_suffix('.md.tmp')
    temp.write_text(content)
    temp.replace(IDE_TABLE)
    with (ROOT / 'table1_comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write(ROOT / 'comparison.json', {'rows': rows, 'progress': progress})
    if not upload:
        return
    try:
        import wandb
        report = ROOT / 'reports/fresh20k-palmetto-krv100'
        report.mkdir(parents=True, exist_ok=True)
        paths = [ROOT / name for name in ('table1_comparison.md', 'table1_comparison.csv', 'comparison.json', 'catalog.json', 'status.json')]
        for selection in ORDER:
            for name in ('summary.json', 'policy_probe.json', 'suite_status.json'):
                path = ROOT / 'results' / selection / name
                if path.exists():
                    target = report / f'{selection}-{name}'
                    shutil.copy2(path, target)
                    paths.append(target)
        for path in paths:
            target = report / path.name
            if path != target:
                shutil.copy2(path, target)
        api = wandb.Api(timeout=30)
        runs = ['aparame-clemson-university/Kinematic-Residual-VLA/' + run for run in ('nj3zd8q7', '2qh0ijb1')]
        for run_path in runs:
            run = api.run(run_path)
            for path in report.iterdir():
                run.upload_file(str(path), root=str(ROOT))
        write(ROOT / 'wandb_upload.json', {'state': 'uploaded', 'runs': runs, 'updated_utc': datetime.now(timezone.utc).isoformat()})
    except Exception as error:
        write(ROOT / 'wandb_upload.json', {'state': 'retry_next_phase', 'error': str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status('preparing')
        catalog = prepare()
        validate(catalog)
        status('prepared' if args.prepare_only else 'waiting_for_free_gpus', selections=ORDER)
        publish(catalog, upload=not args.prepare_only)
        if args.prepare_only:
            print(json.dumps({'models': catalog['models'], 'results': str(ROOT)}, indent=2))
            return
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            time.sleep(30)
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Evaluation service stopped')
        signal.signal(signal.SIGTERM, interrupted)
        for selection in ORDER:
            root = ROOT / 'results' / selection
            try:
                status('evaluating', selection=selection, phase='pilot')
                write(root / 'suite_status.json', {'state': 'pilot'})
                publish(catalog)
                phase(catalog, selection, 'pilot')
                for i, dim in enumerate(DIMS):
                    status('evaluating', selection=selection, phase='full', dimension=dim)
                    write(root / 'suite_status.json', {'state': 'running', 'dimension': dim})
                    phase(catalog, selection, 'full', i)
                    publish(catalog)
                write(root / 'suite_status.json', {'state': 'complete', 'verified_trials': 360})
            except Exception as error:
                write(root / 'suite_status.json', {'state': 'failed', 'error': str(error), 'traceback': traceback.format_exc(), 'infrastructure_failure': True})
            publish(catalog)
        failed = [name for name in ORDER if read(ROOT / 'results' / name / 'suite_status.json')['state'] != 'complete']
        status('complete' if not failed else 'finished_with_failures', failed_selections=failed)
        publish(catalog)


if __name__ == '__main__':
    main()
