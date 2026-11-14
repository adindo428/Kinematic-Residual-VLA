#!/usr/bin/env python3
"""Run the frozen 20-case / 100-trial procedural collection-supported evaluation."""
import argparse
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / 'scratch/dom45'
ROOT = WORK / 'evaluations/collection-matched20k-20261102'
OLD = WORK / 'evaluations/multivla20k-20261029'
RPC = WORK / 'training/krv100-full-20261010/source'
KRV = WORK / 'training/krv100-fresh20k-h200-20261029/source'
ORDER = ['krv100_best', 'krv100_latest', 'smolvla_baseline', 'smolvla_kr', 'vlash_kr', 'pi05_baseline', 'dom100_best', 'dom100_latest']
TASKS = ['pick', 'place', 'long_horizon']
sys.path.insert(0, str(REPO / 'evaluation/official_dom/scripts'))
from report import verify_folder, read_records
from run_suite import stop_group
from evaluation_training_policy import require_eligible, comparison_note


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def status(state, **details):
    write(ROOT / 'status.json', {'state': state, 'updated_utc': datetime.now(timezone.utc).isoformat(), **details})


def prepare():
    if (ROOT / 'catalog.json').exists():
        return read(ROOT / 'catalog.json')
    protocol = read(ROOT / 'protocol.json')
    assert len(protocol['tests']) == 20 and protocol['repeats'] == 5 and protocol['scored_trials'] == 100
    catalog = {key: value for key, value in read(OLD / 'catalog.json').items()
               if key in ('python', 'expected_gpu', 'official_source', 'dom_source', 'vlash_source', 'assets')}
    entries = {}
    for directory in ['smolvla-runtime-repair-20261102', 'completed-vlas-after-current-20261102']:
        for name, entry in read(WORK / 'evaluations' / directory / 'catalog.json')['models'].items():
            require_eligible(name, entry)
            entries[name] = {**entry, 'client_kind': 'smol' if name.startswith('smolvla') else 'completed'}
    for name, directory, selection, kr in [
        ('dom100_best', 'multivla20k-20261029', 'dom100', False),
        ('dom100_latest', 'dom-latest-table1-20261031', 'dom100', False),
        ('krv100_best', 'krv100-fresh20k-palmetto-table1-20261101', 'best', True),
        ('krv100_latest', 'krv100-fresh20k-palmetto-table1-20261101', 'latest', True)]:
        require_eligible('krv100_fresh' if kr else 'dom100')
        entry = read(WORK / 'evaluations' / directory / 'catalog.json')['models'][selection]
        entries[name] = {**entry, 'client_kind': 'native', 'kr': kr, 'source': str(KRV if kr else RPC)}
    files = set()
    for filename in ['run_collection_matched_eval.py', 'collection_matched_native_probe.py', 'inference_krvla.py',
                     'inference_fresh20k_checkpoint.py', 'krvla_policy_runtime.py', 'collection_matched_server.py',
                     'smolvla_table1_client.py', 'completed_vla_table1_client.py', 'evaluation_training_policy.py']:
        files.add(REPO / 'workstation' / filename)
    files.update((REPO / 'evaluation/official_dom/scripts').glob('*.py'))
    for entry in entries.values():
        checkpoint = Path(entry['checkpoint'])
        path = checkpoint / 'model.safetensors' if checkpoint.is_dir() else checkpoint
        assert sha(path) == entry['checkpoint_sha256'], 'Checkpoint changed: ' + str(path)
        stat = path.stat()
        entry.update(weight_path=str(path), weight_bytes=stat.st_size, weight_mtime_ns=stat.st_mtime_ns)
        if checkpoint.is_dir():
            files.add(checkpoint / 'config.json')
        if entry['client_kind'] != 'native':
            files.update(p for p in Path(entry['code_root']).rglob('*') if p.suffix in ('.py', '.json') and p.is_file())
        else:
            for sub in ('policies', 'utils', 'core', 'proactive_vla'):
                files.update((Path(entry['source']) / sub).rglob('*.py'))
    files.add(RPC / 'scripts/inference.py')
    files.add(Path(catalog['official_source']) / 'scripts/inference.py')
    files.add(OLD / 'workstation/multivla_table1_client.py')
    files.add(OLD / 'workstation/rpc_geometry.py')
    catalog.update(models=entries, order=ORDER, held_models=['vlash_baseline', 'pi05_kr'],
                   protocol_sha256=sha(ROOT / 'protocol.json'), files_sha256={str(p): sha(p) for p in sorted(files)})
    for name, entry in entries.items():
        root = ROOT / 'results' / name
        shutil.copytree(ROOT / 'test_lists', root / 'test_lists', dirs_exist_ok=True)
        write(root / 'protocol.json', {**protocol, 'checkpoint': entry})
        write(root / 'suite_status.json', {'state': 'queued', 'verified_trials': 0})
    write(ROOT / 'catalog.json', catalog)
    return catalog


def validate(catalog):
    assert sha(ROOT / 'protocol.json') == catalog['protocol_sha256']
    protocol = read(ROOT / 'protocol.json')
    assert sha(ROOT / 'selection.json') == protocol['selection_sha256']
    for test in protocol['tests']:
        assert sha(test['path']) == test['config_sha256'], 'Frozen scenario changed'
    for path, expected in {**catalog['files_sha256'], **protocol['asset_sha256'], **protocol['files_sha256']}.items():
        assert sha(path) == expected, 'Frozen dependency changed: ' + path
    for name, entry in catalog['models'].items():
        path = Path(entry['weight_path'])
        assert path.stat().st_size == entry['weight_bytes'] and path.stat().st_mtime_ns == entry['weight_mtime_ns']
        if entry['client_kind'] != 'native':
            require_eligible(name, entry)


def client_command(catalog, name, folder):
    entry = catalog['models'][name]
    port = '27600'
    if entry['client_kind'] == 'native':
        script = 'inference_fresh20k_checkpoint.py' if entry['kr'] else 'inference_krvla.py'
        command = [str(WORK / 'envs/train/bin/python'), '-u', str(REPO / 'workstation' / script),
            '--source', entry['source'], '-p', entry['checkpoint'], '-a', name, '-r', 'euler', '-d', '-s',
            '--img_port', port, '--act_port', '27601', '-o', str(folder / 'actions')]
        if entry['kr']:
            command += ['--rpc-source', str(RPC)]
        return command
    script = 'smolvla_table1_client.py' if entry['client_kind'] == 'smol' else 'completed_vla_table1_client.py'
    return [entry.get('python', catalog['python']), '-u', str(REPO / 'workstation' / script),
            '--catalog', str(ROOT / 'catalog.json'), '--model', name, '--output', str(folder), '--port', port]


def phase(catalog, name, mode):
    root = ROOT / 'results' / name
    folder = root / mode
    listing = root / 'test_lists' / ('pilot.txt' if mode == 'pilot' else 'all.txt')
    repeats = 1 if mode == 'pilot' else 5
    if folder.exists():
        if (folder / 'verification.json').exists():
            return verify_folder(folder, root, listing, repeats)
        raise FileExistsError('Partial phase retained; investigate before retry: ' + str(folder))
    if mode == 'full':
        assert read(root / 'pilot/verification.json')['verified']
    validate(catalog)
    folder.mkdir(parents=True)
    env = dict(os.environ, DOM_OFFICIAL_SOURCE=catalog['official_source'], DOM_TRIAL_RESULTS=str(folder / 'trial_results.jsonl'),
               DOM_RECORD_RUNTIME='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    client_env, server_env = dict(env, CUDA_VISIBLE_DEVICES='1'), dict(env, CUDA_VISIBLE_DEVICES='0')
    client = client_command(catalog, name, folder)
    if mode == 'pilot':
        entry = catalog['models'][name]
        if entry['client_kind'] == 'native':
            probe = [str(WORK / 'envs/train/bin/python'), '-u', str(REPO / 'workstation/collection_matched_native_probe.py'),
                '--source', entry['source'], '--checkpoint', entry['checkpoint'], '--output', str(folder / 'policy_probe.json')]
            if entry['kr']:
                probe.append('--kr')
        else:
            probe = [*client, '--probe']
        with (folder / 'probe.log').open('w') as stream:
            subprocess.run(probe, env=client_env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=1200)
    server = [str(REPO / 'workstation/isaaclab-paper45.sh'), '-p', str(REPO / 'workstation/collection_matched_server.py'),
        '--headless', '--enable_cameras', '--device', 'cuda:0', '--num_envs', '1',
        '--scene_dir', str(WORK / 'assets/full-scene-release'), '--object_dir', str(WORK / 'assets/objects'),
        '--env_cfg', str(listing), '--output_dir', str(folder / 'videos'), '-n', str(repeats), '--save',
        '--img_port', '27600', '--act_port', '27601', '--kit_args', os.environ['KRVLA_KIT_ARGS']]
    write(folder / 'commands.json', {'server': server, 'client': client})
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True)
    assert len(gpu.splitlines()) == 2 and all('RTX A6000' in line for line in gpu.splitlines())
    write(folder / 'runtime.json', {'gpu': gpu, 'sim_gpu': 0, 'policy_gpu': 1, 'repeats': repeats,
        'streaming': True, 'isaac_sim': '4.5.0', 'isaac_lab': '2.2.1'})
    processes = []
    try:
        with (folder / 'server.log').open('w') as slog, (folder / 'client.log').open('w') as clog:
            processes.append(subprocess.Popen(server, env=server_env, stdout=slog, stderr=subprocess.STDOUT, start_new_session=True))
            processes.append(subprocess.Popen(client, env=client_env, stdout=clog, stderr=subprocess.STDOUT, start_new_session=True))
            changed, size = time.monotonic(), -1
            while processes[1].poll() is None:
                if processes[0].poll() is not None:
                    raise RuntimeError('Simulator exited; inspect server.log')
                path = folder / 'trial_results.jsonl'
                new = path.stat().st_size if path.exists() else 0
                if new != size:
                    changed, size = time.monotonic(), new
                    status('evaluating', model=name, phase=mode, completed_records=len(path.read_text().splitlines()) if path.exists() else 0)
                if time.monotonic()-changed > 1200:
                    raise RuntimeError('No completed trial for 20 minutes')
                time.sleep(5)
            if processes[1].returncode:
                raise RuntimeError('Policy exited with an error; inspect client.log')
        return verify_folder(folder, root, listing, repeats)
    except BaseException as error:
        write(folder / 'failure.json', {'error': str(error), 'infrastructure_failure': True})
        raise
    finally:
        for process in reversed(processes):
            stop_group(process)


def summarize(catalog, name):
    root = ROOT / 'results' / name
    verified = verify_folder(root / 'full', root, root / 'test_lists/all.txt', 5)
    assert verified['trials'] == 100
    records = read_records(root / 'full')
    task_of = {t['name']: t['task'] for t in read(ROOT / 'protocol.json')['tests']}
    observed = {}
    for task, count in [('pick', 45), ('place', 45), ('long_horizon', 10)]:
        values = [r for r in records if task_of[r['env_name']] == task]
        assert len(values) == count
        observed[task] = 100 * statistics.mean(r['success'] for r in values)
    observed.update(Success=verified['success_percent'], Time=verified['mean_task_time_s'])
    write(root / 'summary.json', {'complete': True, 'verified_trials': 100, 'observed': observed,
        'checkpoint': catalog['models'][name], 'protocol_sha256': catalog['protocol_sha256']})
    write(root / 'suite_status.json', {'state': 'complete', 'verified_trials': 100})


def publish(catalog, upload=False):
    rows = []
    for name in catalog['order']:
        path = ROOT / 'results' / name / 'summary.json'
        rows.append({'Method': catalog['models'][name]['label'], **(read(path)['observed'] if path.exists() else {})})
    fields = ['Method', *TASKS, 'Success', 'Time']
    text = ('# Collection-supported 20k evaluation\n\n'
        '20 frozen procedural scenarios × 5 repeats = 100 scored trials per checkpoint: '
        '45 pick, 45 place and 10 long-horizon trials. Three pilot trials are excluded. '
        'These are new collection-generator cases, not the official paper Table 1 and not exact episode replays. '
        'Scenario selection is frozen before any policy results. The earlier 360-trial stress suite remains separate.\n\n')
    text += '| Method | Pick (%) | Place (%) | Long horizon (%) | Success (%) | Mean time (s) |\n'
    text += '|---|---:|---:|---:|---:|---:|\n'
    for row in rows:
        text += '| ' + row['Method'] + ' | ' + ' | '.join(f'{row[key]:.2f}' if key in row else 'pending' for key in fields[1:]) + ' |\n'
    text += '\n' + comparison_note() + '\n\n'
    text += ('House membership, all distractor/container identities and exact poses cannot be recovered from the retained episode metadata. '
             'Task, principal assets, rigid counts, motion profiles and camera setup are supported by the 20k data; room/geometry sampling uses the collection generator. '
             'Pick/object and place/container distractor allocation is a documented design assumption. '
             'Repeated runs share frozen geometry; this is a small deadline-budget evaluation.\n')
    report = ROOT / 'table1_comparison.md'
    report.write_text(text)
    with (ROOT / 'table1_comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    write(ROOT / 'comparison.json', {'rows': rows, 'scored_trials_per_checkpoint': 100})
    ide = WORK / 'results/krv100-best-epoch003-task-table1-20261025/table1_task_subset.md'
    archive = ROOT / 'previous_stress_table.md'
    if not archive.exists():
        shutil.copy2(ide, archive)
    ide.write_text(text)
    if upload:
        # Only result tables; detailed provenance, paths, manifests and raw trials stay local.
        try:
            import wandb
            run = wandb.Api(timeout=30).run('aparame-clemson-university/Kinematic-Residual-VLA/2qh0ijb1')
            directory = ROOT / 'reports/collection-supported-100'
            directory.mkdir(parents=True, exist_ok=True)
            numeric = '| Method | Pick (%) | Place (%) | Long horizon (%) | Success (%) | Mean time (s) |\n|---|---:|---:|---:|---:|---:|\n'
            for row in rows:
                numeric += '| ' + row['Method'] + ' | ' + ' | '.join(f'{row[key]:.2f}' if key in row else 'pending' for key in fields[1:]) + ' |\n'
            (directory / 'results.md').write_text('100-trial collection-supported procedural evaluation; separate from paper stress tests.\n\n' + numeric)
            shutil.copy2(ROOT / 'table1_comparison.csv', directory / 'results.csv')
            for path in directory.iterdir():
                run.upload_file(str(path), root=str(ROOT))
            write(ROOT / 'wandb_upload.json', {'state': 'uploaded_results_only', 'updated_utc': datetime.now(timezone.utc).isoformat()})
        except Exception as error:
            write(ROOT / 'wandb_upload.json', {'state': 'retry_next_checkpoint', 'error': str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        catalog = prepare()
        validate(catalog)
        for name in catalog['order']:
            root = ROOT / 'results' / name
            (root / 'test_lists/all.txt').write_text(''.join(t['path'] + '\n' for t in read(ROOT / 'protocol.json')['tests']))
        publish(catalog)
        if args.prepare_only:
            status('prepared', order=catalog['order'], scored_trials_per_checkpoint=100)
            return
        for name in catalog['order']:
            root = ROOT / 'results' / name
            if (root / 'summary.json').exists():
                continue
            try:
                for mode in ['pilot', 'full']:
                    status('evaluating', model=name, phase=mode)
                    write(root / 'suite_status.json', {'state': mode, 'verified_trials': 0})
                    phase(catalog, name, mode)
                summarize(catalog, name)
                publish(catalog, upload=True)
            except BaseException as error:
                write(root / 'suite_status.json', {'state': 'failed', 'error': str(error), 'traceback': traceback.format_exc()})
                status('failed', model=name, error=str(error))
                publish(catalog)
                raise
        status('complete', checkpoints=len(catalog['order']), scored_trials_per_checkpoint=100)
        publish(catalog, upload=True)


if __name__ == '__main__':
    main()
