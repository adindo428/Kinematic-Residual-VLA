#!/usr/bin/env python3
"""Run a shared training-prompt-matched approximate IID evaluation."""
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
ROOT = WORK / 'evaluations/prompt-matched20k-20261102'
PREVIOUS = WORK / 'evaluations/collection-matched20k-20261102'
OLD = WORK / 'evaluations/multivla20k-20261029'
RPC = WORK / 'training/krv100-full-20261010/source'
KRV = WORK / 'training/krv100-fresh20k-h200-20261029/source'
ORDER = ['krv100_best', 'krv100_latest', 'smolvla_baseline', 'smolvla_kr', 'vlash_kr', 'pi05_baseline', 'dom100_best', 'dom100_latest']
TASKS = ['pick', 'place', 'long_horizon']
sys.path.insert(0, str(REPO / 'evaluation/official_dom/scripts'))
from report import verify_folder as verify_trial_folder, read_records
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
    assert len(protocol['tests']) == 20 and protocol['scored_trials'] == 100
    assert all(t['condition'] == 'approx_iid' for t in protocol['tests'])
    selection = read(ROOT / 'selection.json')
    cases = {c['name']:c for c in selection['cases']}
    witnesses = {c['design_witness'] for c in selection['cases']}
    cameras = {}
    for line in (WORK / 'cache/huggingface/lerobot/hzxie/DOM/meta/camera.jsonl').open():
        row = json.loads(line)
        if row['episode_index'] in witnesses:
            cameras[row['episode_index']] = row['cameras']
    for test in protocol['tests']:
        cfg = read(test['path']); case = cases[test['name']]
        evidence = read(ROOT / 'case_evidence' / (test['name']+'.json'))
        assert len(evidence['prompts']) == 5
        for row in evidence['prompts']:
            witness = case['training_witnesses'][str(row['training_witness_episode'])]
            assert row['text'] == cfg['evaluation_prompts'][row['trial']-1]
            for key in ['objects','containers']:
                description = row['metadata'][key][0]
                assert description in witness[key] and description in cfg['instruction'][key]
        assert Path(cfg['scene']['object']['spawn']['usd_path']).stem == case['target_asset']
        assert Path(cfg['scene']['container']['spawn']['usd_path']).stem == case['container_asset']
        for camera in cameras[case['design_witness']]:
            actual = cfg['scene'][camera['name']]
            assert actual['width'] == camera['width'] and actual['height'] == camera['height']
            assert actual['offset']['convention'] == camera['offset']['convention']
            for field in ['pos','rot']:
                assert all(abs(a-b)<1e-6 for a,b in zip(actual['offset'][field],camera['offset'][field]))
            assert abs(actual['spawn']['focal_length']-camera['focal'])<1e-6
    # Reuse the prior complete SHA256 verification only when file identity,
    # size and nanosecond mtime still match. Never copy the large weights.
    catalog = read(PREVIOUS / 'catalog.json')
    for entry in catalog['models'].values():
        stat = Path(entry['weight_path']).stat()
        assert stat.st_size == entry['weight_bytes'] and stat.st_mtime_ns == entry['weight_mtime_ns']
    for path, expected in catalog['files_sha256'].items():
        assert sha(path) == expected, 'Prior verified inference source changed: ' + path
    catalog['checkpoint_verification_reused_from'] = str(PREVIOUS / 'catalog.json')
    catalog['checkpoint_verification_receipt_sha256'] = sha(PREVIOUS / 'catalog.json')
    catalog['protocol_sha256'] = sha(ROOT / 'protocol.json')
    for name in ['run_prompt_matched_eval.py', 'prompt_matched_server.py', 'prepare_prompt_matched_eval.py',
                 'run-prompt-matched-eval.sh']:
        path = REPO / 'workstation' / name
        catalog['files_sha256'][str(path)] = sha(path)
    for path in [ROOT / 'distribution_evidence.json', *(ROOT / 'case_evidence').glob('*.json')]:
        catalog['files_sha256'][str(path)] = sha(path)
    for entry_name, entry in catalog['models'].items():
        root = ROOT / 'results' / entry_name
        shutil.copytree(ROOT / 'test_lists', root / 'test_lists', dirs_exist_ok=True)
        write(root / 'protocol.json', {**protocol, 'checkpoint': entry})
        write(root / 'suite_status.json', {'state': 'queued', 'verified_trials': 0})
    write(ROOT / 'catalog.json', catalog)
    return catalog


def verify_folder(folder, root, listing, repeats):
    verified = verify_trial_folder(folder, root, listing, repeats)
    trace = [json.loads(line) for line in (Path(folder) / 'prompt_trace.jsonl').read_text().splitlines()]
    tests = {t['name']: read(t['path']) for t in read(Path(root) / 'protocol.json')['tests']}
    names = [Path(line).stem for line in Path(listing).read_text().splitlines() if line]
    expected = {(name, i+1) for name in names for i in range(repeats)}
    assert len(trace) == len(expected) and {(r['env_name'],r['trial']) for r in trace} == expected
    for row in trace:
        cfg = tests[row['env_name']]
        assert row['text'] == cfg['evaluation_prompts'][row['trial']-1]
        assert row['seed'] == cfg['seed']
    verified['matched_prompts_verified'] = True
    write(Path(folder) / 'verification.json', verified)
    return verified


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
               DOM_RECORD_RUNTIME='1', DOM_EXPECTED_SIM='4.5.0', DOM_EXPECTED_LAB='2.2.1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
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
    server = [str(REPO / 'workstation/isaaclab-paper45.sh'), '-p', str(REPO / 'workstation/prompt_matched_server.py'),
        '--headless', '--enable_cameras', '--device', 'cuda:0', '--num_envs', '1', '--tolerance', '0.07',
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
    text = ('# Training-prompt-matched 20k evaluation (approximately IID)\n\n'
        '20 frozen procedural scenarios × 5 repeats = 100 scored trials per checkpoint: '
        '45 pick, 45 place and 10 long-horizon trials. Three pilot trials are excluded. '
        'These are approximately IID procedural cases matched to training prompts and initial-state support; house membership remains unverified. '
        'Every method receives the same five frozen prompts per case and the same reset seeds. Selection uses no policy success results. Earlier stress suites remain separate. Velocity-only OOD cases are prepared separately and are NOT queued.\n\n')
    text += '| Method | Pick (%) | Place (%) | Long horizon (%) | Success (%) | Mean time (s) |\n'
    text += '|---|---:|---:|---:|---:|---:|\n'
    for row in rows:
        text += '| ' + row['Method'] + ' | ' + ' | '.join(f'{row[key]:.2f}' if key in row else 'pending' for key in fields[1:]) + ' |\n'
    text += '\n' + comparison_note() + '\n\n'
    text += ('House membership, all distractor/container identities and exact poses cannot be recovered from the retained episode metadata. '
             'Task, principal assets, rigid counts, motion profiles and camera setup are supported by the 20k data; room/geometry sampling uses the collection generator. '
             'Entity counts are inferred from training relational prompts and total rigid counts. The case evidence records training witnesses, supported description pairs, and nearest initial-state distances. '
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
            directory = ROOT / 'reports/prompt-matched-iid100'
            directory.mkdir(parents=True, exist_ok=True)
            numeric = '| Method | Pick (%) | Place (%) | Long horizon (%) | Success (%) | Mean time (s) |\n|---|---:|---:|---:|---:|---:|\n'
            for row in rows:
                numeric += '| ' + row['Method'] + ' | ' + ' | '.join(f'{row[key]:.2f}' if key in row else 'pending' for key in fields[1:]) + ' |\n'
            (directory / 'results.md').write_text('100-trial training-prompt-matched approximate IID evaluation; separate from velocity shifts and paper stress tests.\n\n' + numeric)
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
