#!/usr/bin/env python3
"""Resume the frozen Table 1 suite with two independent evaluator lanes."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

import run_low_clutter_table1 as original
from render_table1_live_status import render
from table1_latency_control import ModelHeld

ROOT = original.ROOT
REPO = original.REPO
WORK = original.WORK
STOP = threading.Event()
GATE = None


def lane_status(lane, **details):
    original.write(ROOT / 'parallel_lanes' / f'{lane}.json', {
        'lane': lane, 'updated_utc': datetime.now(timezone.utc).isoformat(), **details})


def commands(catalog, name, folder, listing, repeats, lane):
    port = 27600 + lane * 10
    client = original.runtime.client_command(catalog, name, folder)
    if '--img_port' in client:
        client[client.index('--img_port') + 1] = str(port)
        client[client.index('--act_port') + 1] = str(port + 1)
    else:
        client[client.index('--port') + 1] = str(port)
    server = [str(REPO / 'workstation/isaaclab-paper45.sh'), '-p',
        str(REPO / 'workstation/low_clutter_table1_server.py'), '--headless', '--enable_cameras',
        '--device', 'cuda:0', '--num_envs', '1', '--tolerance', '0.07',
        '--scene_dir', str(WORK / 'assets/scenes'), '--object_dir', str(WORK / 'assets/objects'),
        '--env_cfg', str(listing), '--output_dir', str(folder / 'videos'), '-n', str(repeats), '--save',
        '--img_port', str(port), '--act_port', str(port + 1), '--kit_args',
        os.environ['KRVLA_KIT_ARGS'] + ' --/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false']
    return server, client


def probe(command, env, folder):
    process = None
    try:
        with (folder / 'probe.log').open('w') as stream:
            process = subprocess.Popen(command, env=env, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
            started = time.monotonic()
            while process.poll() is None:
                if STOP.wait(1):
                    raise InterruptedError('Parallel queue stopped during checkpoint probe')
                if time.monotonic() - started > 1200:
                    raise TimeoutError('Checkpoint probe exceeded 20 minutes')
            if process.returncode:
                raise RuntimeError('Checkpoint probe failed: ' + str(folder / 'probe.log'))
    finally:
        if process is not None:
            original.stop_group(process)


def phase(catalog, name, dimension, lane):
    root = ROOT / 'results' / name
    folder = root / 'pilot' if dimension is None else root / 'full' / f'{original.DIMS.index(dimension):02d}_{dimension}'
    if (folder / 'verification.json').exists() or GATE is None:
        return _phase(catalog, name, dimension, lane)
    GATE.enter(lane, name, dimension)
    try:
        return _phase(catalog, name, dimension, lane)
    finally:
        GATE.leave(lane)


def _phase(catalog, name, dimension, lane):
    root = ROOT / 'results' / name
    pilot = dimension is None
    folder = root / 'pilot' if pilot else root / 'full' / f'{original.DIMS.index(dimension):02d}_{dimension}'
    listing = root / 'test_lists' / ('pilot.txt' if pilot else dimension + '.txt')
    repeats = 1 if pilot else 20
    if folder.exists():
        if (folder / 'verification.json').exists():
            return original.verify(folder, root, listing, repeats)
        raise FileExistsError('Partial output retained; inspect before retry: ' + str(folder))
    if not pilot:
        assert original.read(root / 'pilot/verification.json')['matched_prompts_verified']
    original.validate(catalog)
    folder.mkdir(parents=True)
    target = 3 if pilot else 200
    def progress(count, state='evaluating'):
        details = {'state': state, 'model': name, 'phase': 'pilot' if pilot else 'full',
            'dimension': dimension, 'completed_records_in_phase': count, 'phase_trial_target': target,
            'checkpoint_trial_target': 1800, 'image_port': 27600 + lane * 10}
        lane_status(lane, **details)
        original.write(root / 'suite_status.json', details)
    progress(0, 'probing' if pilot else 'evaluating')
    env = dict(os.environ, DOM_OFFICIAL_SOURCE=catalog['official_source'],
        DOM_TRIAL_RESULTS=str(folder / 'trial_results.jsonl'), DOM_EXPECTED_SIM='4.5.0',
        DOM_EXPECTED_LAB='2.2.1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    client_env = dict(env, CUDA_VISIBLE_DEVICES='1')
    server_env = dict(env, CUDA_VISIBLE_DEVICES='0')
    server, client = commands(catalog, name, folder, listing, repeats, lane)
    original.write(folder / 'commands.json', {'server': server, 'client': client})
    original.write(folder / 'runtime.json', {'simulation_gpu': 0, 'inference_gpu': 1,
        'parallel_lanes': GATE.limit if GATE else 2, 'execution_mode': GATE.mode if GATE else 'parallel_shared',
        'latency_guard_version': 1, 'lane': lane, 'repeats': repeats, 'streaming': True,
        'scheduling_receipt': str(ROOT / 'parallel_execution.json')})
    processes = []
    try:
        if pilot:
            entry = catalog['models'][name]
            if entry['client_kind'] == 'native':
                command = [str(WORK / 'envs/train/bin/python'), '-u',
                    str(REPO / 'workstation/collection_matched_native_probe.py'),
                    '--source', entry['source'], '--checkpoint', entry['checkpoint'],
                    '--output', str(folder / 'policy_probe.json')]
                if entry['kr']:
                    command += ['--kr']
            else:
                command = [*client, '--probe']
            probe(command, client_env, folder)
        progress(0)
        with (folder / 'server.log').open('w') as slog, (folder / 'client.log').open('w') as clog:
            processes.append(subprocess.Popen(server, env=server_env, stdout=slog,
                stderr=subprocess.STDOUT, start_new_session=True))
            processes.append(subprocess.Popen(client, env=client_env, stdout=clog,
                stderr=subprocess.STDOUT, start_new_session=True))
            changed = time.monotonic()
            size = -1
            while processes[1].poll() is None:
                if STOP.is_set():
                    raise InterruptedError('Parallel queue stopped')
                if processes[0].poll() is not None:
                    raise RuntimeError('Simulator exited: ' + str(folder / 'server.log'))
                path = folder / 'trial_results.jsonl'
                new = path.stat().st_size if path.exists() else 0
                if new != size:
                    changed = time.monotonic()
                    size = new
                    progress(len(path.read_text().splitlines()) if path.exists() else 0)
                if time.monotonic() - changed > 1200:
                    raise TimeoutError('No completed trial for 20 minutes: ' + str(folder))
                STOP.wait(2)
            if processes[1].returncode:
                raise RuntimeError('Policy exited: ' + str(folder / 'client.log'))
        verified = original.verify(folder, root, listing, repeats)
        progress(verified['trials'], 'verified')
        return verified
    except BaseException as error:
        original.write(folder / 'failure.json', {'error': str(error), 'infrastructure_failure': True})
        raise
    finally:
        for process in reversed(processes):
            original.stop_group(process)


def worker(catalog, pending, lane, failures):
    while not STOP.is_set():
        try:
            name = pending.get_nowait()
        except queue.Empty:
            lane_status(lane, state='idle')
            return
        try:
            priority = catalog.get('priority_checkpoint')
            while priority and name != priority and not (ROOT / 'results' / priority / 'summary.json').exists():
                lane_status(lane, state='waiting_for_priority_checkpoint', model=name,
                    phase='waiting', priority_checkpoint=priority)
                if STOP.wait(1):
                    raise InterruptedError('Stopped while waiting for priority checkpoint')
            entry = catalog['models'][name]
            if name == 'old_krv100':
                request = original.read(ROOT / 'historical_checkpoint_request.json')
                assert request['explicit_user_request'] and request['model'] == name
                assert entry['historical_user_exception'] and not entry['primary_20k_cohort']
                assert entry['checkpoint_sha256'] == request['checkpoint_sha256']
                assert entry['config_sha256'] == request['config_sha256'] == original.sha(Path(entry['checkpoint'])/'config.json')
                assert entry['checkpoint'] == request['checkpoint']
            else:
                original.require_eligible('dom100' if name.startswith('dom100') else
                    'krv100_fresh' if name.startswith('krv100') else name,
                    entry if entry['client_kind'] != 'native' else None)
            phase(catalog, name, None, lane)
            for dimension in original.DIMS:
                if STOP.is_set():
                    raise InterruptedError('Parallel queue stopped')
                phase(catalog, name, dimension, lane)
            original.summarize(catalog, name)
            lane_status(lane, state='checkpoint_complete', model=name)
        except ModelHeld as error:
            detail = {'state': 'held_for_latency_review', 'model': name, 'reason': str(error)}
            original.write(ROOT / 'results' / name / 'suite_status.json', detail)
            lane_status(lane, **detail)
        except BaseException as error:
            detail = {'state': 'failed', 'model': name, 'error': str(error), 'traceback': traceback.format_exc()}
            original.write(ROOT / 'results' / name / 'suite_status.json', detail)
            lane_status(lane, **detail)
            failures.put(detail)
            STOP.set()
            return
        finally:
            pending.task_done()


def snapshot(state='evaluating'):
    lanes = [original.read(ROOT / 'parallel_lanes' / f'{i}.json') for i in range(2)]
    active = [r for r in lanes if r['state'] in ['evaluating', 'probing', 'verified']]
    primary = active[0] if active else {}
    original.status(state, **{k: v for k, v in primary.items() if k not in ['state', 'updated_utc']},
        execution_mode=GATE.mode if GATE else 'parallel_shared', concurrency=max(GATE.limit,len(active)) if GATE else 2,
        requested_concurrency=GATE.limit if GATE else 2,
        latency_guard_version=1, active_evaluations=active, lanes=lanes)


def main():
    global GATE
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        catalog = original.prepare()
        original.ORDER = list(catalog['order'])
        original.validate(catalog)
        for port in [27600, 27601, 27610, 27611]:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', port))
        assert original.read(WORK / 'evaluations/parallel-capability-20261103/assessment.json')['complete']
        original.write(ROOT / 'parallel_resume_status.before.json', original.read(ROOT / 'status.json'))
        previous_execution = original.read(ROOT / 'parallel_execution.json') if (ROOT / 'parallel_execution.json').exists() else {}
        original.write(ROOT / 'parallel_execution.json', {'started_utc': previous_execution.get('started_utc', datetime.now(timezone.utc).isoformat()),
            'resumed_utc': datetime.now(timezone.utc).isoformat(), 'latency_guard_version': 1,
            'user_authorized': True, 'concurrency': 2, 'simulation_gpu': 0, 'inference_gpu': 1,
            'ports': [[27600, 27601], [27610, 27611]], 'protocol_sha256': catalog['protocol_sha256'],
            'runner_sha256': original.sha(__file__),
            'benchmark_assessment_sha256': original.sha(WORK / 'evaluations/parallel-capability-20261103/assessment.json'),
            'preserve_verified_phases': True, 'trials_per_checkpoint': 1800,
            'runtime_note': 'Parallel mode begins after DOM-best SR. Benchmark showed ~2x throughput, +8.4% chunk p95; accuracy equivalence is not established.'})
        pending = queue.Queue()
        for name in catalog.get('queue_order', catalog['order']):
            if name not in catalog.get('enabled_models', catalog['models']):
                continue
            if not (ROOT / 'results' / name / 'summary.json').exists():
                pending.put(name)
        for i in range(2):
            lane_status(i, state='starting')
        failures = queue.Queue()
        from table1_latency_control import LatencyGate
        GATE = LatencyGate(catalog, STOP, lane_status)
        threads = [threading.Thread(target=worker, args=(catalog, pending, i, failures), name=f'eval-lane-{i}') for i in range(2)]
        def interrupted(signum, frame):
            STOP.set()
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        original.publish(catalog, upload=False)
        previous = None
        for thread in threads:
            thread.start()
        try:
            while any(t.is_alive() for t in threads):
                GATE.check_request()
                snapshot('stopping' if STOP.is_set() else 'evaluating')
                signature = tuple(sorted(str(p) for p in [
                    *(ROOT / 'results').glob('*/full/*/verification.json'),
                    *(ROOT / 'results').glob('*/summary.json')]))
                if signature != previous:
                    original.publish(catalog, upload=False)
                    subprocess.run([sys.executable, str(REPO / 'workstation/analyze_low_clutter_pairs.py'),
                        '--root', str(ROOT)], check=True)
                    previous = signature
                render(ROOT, not STOP.is_set())
                time.sleep(5)
        finally:
            STOP.set()
            for thread in threads:
                thread.join()
        if not failures.empty():
            error = failures.get()
            snapshot('failed')
            raise RuntimeError(error['error'])
        complete = all((ROOT / 'results' / n / 'summary.json').exists() for n in catalog['order'])
        held = any(original.read(ROOT / 'results' / n / 'suite_status.json').get('state') == 'held_for_latency_review' for n in catalog['order'])
        snapshot('complete' if complete else 'held_for_latency_review' if held else 'interrupted')
        original.publish(catalog, upload=False)
        render(ROOT, False)
        if complete:
            subprocess.run([sys.executable, str(REPO / 'workstation/analyze_low_clutter_pairs.py'),
                '--root', str(ROOT)], check=True)


if __name__ == '__main__':
    main()
