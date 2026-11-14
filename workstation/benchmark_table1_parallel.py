#!/usr/bin/env python3
"""Isolated serial/two-lane capacity test; never writes scored Table 1 trials."""
import concurrent.futures
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time

import run_low_clutter_table1 as official

REPO = official.REPO
WORK = official.WORK
ROOT = WORK / 'evaluations/parallel-capability-20261103'
SERVICE = 'krvla-low-clutter-full-table1-20261102.service'


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def analyze(folder, elapsed):
    client = (folder / 'client.log').read_text(errors='replace')
    server = (folder / 'server.log').read_text(errors='replace')
    inference = [float(x) for x in re.findall(r'Inference Time: ([\d.]+)s', client)]
    steps = [float(x) for x in re.findall(r'\[Step\d+\] Time: ([\d.]+)s', server)]
    action_counts = [(int(a), int(b)) for a, b in re.findall(r'Test done with \w+ in (\d+) steps and (\d+) actions', client)]
    records = official.runtime.read_records(folder)
    return {
        'wall_s_including_startup_and_ack': elapsed,
        'trials': len(records), 'successes': sum(r['success'] for r in records),
        'inference_chunks': len(inference), 'inference_p50_s': percentile(inference, .5),
        'inference_p95_s': percentile(inference, .95),
        'simulation_step_p50_s': percentile(steps, .5), 'simulation_step_p95_s': percentile(steps, .95),
        'simulation_step_wall_sum_s': sum(steps),
        'action_coverage': sum(b for a, b in action_counts) / sum(a for a, b in action_counts),
        'steps_and_actions_per_trial': action_counts,
        'output_queue_full_warnings': client.count('output queue is full'),
        'outcomes': [{'case': r['env_name'], 'success': r['success'], 'steps': r['n_steps']} for r in records],
        'excluded_from_scored_table': True,
    }


def lane(catalog, wave, index, sim_gpu, policy_gpu, listing):
    folder = ROOT / wave / f'lane{index}'
    folder.mkdir(parents=True, exist_ok=False)
    port = 27800 + index * 10
    env = dict(os.environ, DOM_OFFICIAL_SOURCE=catalog['official_source'],
        DOM_TRIAL_RESULTS=str(folder / 'trial_results.jsonl'), DOM_EXPECTED_SIM='4.5.0',
        DOM_EXPECTED_LAB='2.2.1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    client = official.runtime.client_command(catalog, 'dom100_best', folder)
    client[client.index('--img_port') + 1] = str(port)
    client[client.index('--act_port') + 1] = str(port + 1)
    server = [str(REPO / 'workstation/isaaclab-paper45.sh'), '-p',
        str(REPO / 'workstation/low_clutter_table1_server.py'), '--headless', '--enable_cameras',
        '--device', 'cuda:0', '--num_envs', '1', '--tolerance', '0.07',
        '--scene_dir', str(WORK / 'assets/scenes'), '--object_dir', str(WORK / 'assets/objects'),
        '--env_cfg', str(listing), '--output_dir', str(folder / 'videos'), '-n', '1', '--save',
        '--img_port', str(port), '--act_port', str(port + 1), '--kit_args',
        os.environ['KRVLA_KIT_ARGS'] + f' --/renderer/activeGpu={sim_gpu} --/renderer/multiGpu/enabled=false']
    official.write(folder / 'commands.json', {'client': client, 'server': server,
        'simulation_gpu': sim_gpu, 'policy_gpu': policy_gpu, 'scored': False})
    started = time.monotonic()
    processes = []
    try:
        with (folder / 'server.log').open('w') as slog, (folder / 'client.log').open('w') as clog:
            # Kit's renderer uses physical GPU indices; CUDA uses the masked local
            # index. Explicitly pin both and disable automatic multi-GPU rendering.
            processes.append(subprocess.Popen(server, env=dict(env, CUDA_VISIBLE_DEVICES=str(sim_gpu)),
                stdout=slog, stderr=subprocess.STDOUT, start_new_session=True))
            processes.append(subprocess.Popen(client, env=dict(env, CUDA_VISIBLE_DEVICES=str(policy_gpu)),
                stdout=clog, stderr=subprocess.STDOUT, start_new_session=True))
            while processes[1].poll() is None:
                if processes[0].poll() is not None:
                    raise RuntimeError(f'Simulator exited: {folder}')
                if time.monotonic() - started > 1200:
                    raise TimeoutError(str(folder))
                time.sleep(1)
            if processes[1].returncode:
                raise RuntimeError(f'Policy exited: {folder}')
        elapsed = time.monotonic() - started
        official.verify(folder, official.ROOT / 'results/dom100_best', listing, 1)
        result = analyze(folder, elapsed)
        official.write(folder / 'performance.json', result)
        return result
    finally:
        for process in reversed(processes):
            official.stop_group(process)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not any((ROOT / name).exists() for name in
                   ['serial_reference', 'parallel_shared', 'parallel_separate']), 'Do not overwrite benchmark trials'
    catalog = official.read(official.ROOT / 'catalog.json')
    official.validate(catalog)
    # Choose representative conditions before seeing the benchmark outcomes.
    configs = []
    for dimension in ['CR', 'LS', 'SR']:
        configs.append((official.ROOT / 'test_lists' / (dimension + '.txt')).read_text().splitlines()[0])
    listing = ROOT / 'cases.txt'
    listing.write_text('\n'.join(configs) + '\n')
    waves = [('serial_reference', [(0, 1)]),
             ('parallel_shared', [(0, 1), (0, 1)]),
             ('parallel_separate', [(0, 0), (1, 1)])]
    official.write(ROOT / 'plan.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': 'dom100_best', 'cases': configs, 'repeats': 1,
        'waves': waves, 'frozen_physics_prompts_and_policy': True,
        'all_results_excluded_from_table1': True,
        'limitation': 'A capacity smoke test cannot establish equivalent success rates or cover larger policy families.'})
    while not (official.ROOT / 'pause_after_sr_complete.json').exists():
        print('Waiting for verified SR completion and safe pause', flush=True)
        time.sleep(30)
    assert official.read(official.ROOT / 'pause_after_sr_complete.json')['service_inactive']
    assert subprocess.run(['systemctl', '--user', 'is-active', SERVICE], text=True,
                          stdout=subprocess.PIPE).stdout.strip() == 'inactive'
    results = {}
    for wave, placement in waves:
        official.write(ROOT / 'status.json', {'state': 'benchmarking', 'wave': wave})
        print('START', wave, placement, flush=True)
        with (ROOT / (wave + '-gpu.csv')).open('w') as gpu, (ROOT / (wave + '-process-gpu.txt')).open('w') as pgpu:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw',
                '--format=csv,noheader,nounits', '-l', '2'], stdout=gpu, stderr=subprocess.STDOUT)
            process_monitor = subprocess.Popen(['nvidia-smi', 'pmon', '-s', 'um', '-d', '2'],
                stdout=pgpu, stderr=subprocess.STDOUT)
            started = time.monotonic()
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(placement)) as pool:
                    futures = [pool.submit(lane, catalog, wave, i, sim, policy, listing)
                               for i, (sim, policy) in enumerate(placement)]
                    lanes = [f.result() for f in futures]
                results[wave] = {'wall_s_including_cleanup': time.monotonic() - started, 'lanes': lanes}
            finally:
                monitor.terminate()
                monitor.wait(timeout=10)
                process_monitor.terminate()
                process_monitor.wait(timeout=10)
        official.write(ROOT / 'performance.json', results)
        print('FINISHED', wave, json.dumps(results[wave]), flush=True)
    serial = results['serial_reference']['lanes'][0]
    for wave in ['parallel_shared', 'parallel_separate']:
        value = results[wave]
        value['aggregate_wall_speedup'] = 2 * serial['wall_s_including_startup_and_ack'] / max(
            r['wall_s_including_startup_and_ack'] for r in value['lanes'])
        value['worst_inference_p95_ratio'] = max(r['inference_p95_s'] for r in value['lanes']) / serial['inference_p95_s']
        value['worst_simulation_step_p50_ratio'] = max(r['simulation_step_p50_s'] for r in value['lanes']) / serial['simulation_step_p50_s']
        value['normalized_step_throughput_speedup'] = (
            sum(sum(n for n, _ in r['steps_and_actions_per_trial']) for r in value['lanes']) /
            max(r['wall_s_including_startup_and_ack'] for r in value['lanes']) /
            (sum(n for n, _ in serial['steps_and_actions_per_trial']) / serial['wall_s_including_startup_and_ack']))
    official.write(ROOT / 'performance.json', results)
    report = ('# Parallel evaluation capacity benchmark\n\n'
              'All trials are excluded from Table 1. Same DOM-best checkpoint and first CR, LS, SR cases, '
              'one repeat each. Physics, prompts, horizons and policy code are unchanged.\n\n'
              '| Layout | Lane | Wall s¹ | Chunk p50 / p95 s | Step p50 s | Actions / steps | Successes / trials |\n'
              '|---|---:|---:|---:|---:|---:|---:|\n')
    for wave, value in results.items():
        for i, row in enumerate(value['lanes']):
            report += (f"| {wave} | {i} | {row['wall_s_including_startup_and_ack']:.1f} | "
                       f"{row['inference_p50_s']:.3f} / {row['inference_p95_s']:.3f} | "
                       f"{row['simulation_step_p50_s']:.3f} | {row['action_coverage']:.3f} | "
                       f"{row['successes']} / {row['trials']} |\n")
    report += '\n¹ Includes process startup, environment resets, video writing and fixed episode acknowledgments.\n\n'
    for wave in ['parallel_shared', 'parallel_separate']:
        value = results[wave]
        report += (f"- {wave}: aggregate trials/wall throughput {value['aggregate_wall_speedup']:.2f}× serial; "
                   f"steps/wall throughput {value['normalized_step_throughput_speedup']:.2f}×; "
                   f"worst lane chunk p95 {value['worst_inference_p95_ratio']:.2f}× serial.\n")
    report += ('\nThis small capacity test cannot establish equal success rates. Success-dependent early termination '
               'also affects wall throughput; inspect step throughput and per-lane timing together. '
               'Larger policy families require their own timing validation. The official queue remains paused '
               'pending assessment; benchmark results never enter scored reports.\n')
    (ROOT / 'report.md').write_text(report)
    official.write(ROOT / 'status.json', {'state': 'complete', 'official_queue_still_paused': True,
        'scored_results_unchanged': True})


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        official.write(ROOT / 'failure.json', {'error': repr(error), 'scored': False})
        raise
