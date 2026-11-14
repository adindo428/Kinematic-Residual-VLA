#!/usr/bin/env python3
"""Preserve passive runtime evidence without instrumenting frozen evaluators."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'scratch/dom45/evaluations/low-clutter-full-table1-20261102'
OUT = ROOT / 'runtime_metrics'
LOCAL = ZoneInfo('America/New_York')
NATIVE = re.compile(r'\[Step(\d+)\] Inference Time: ([\d.]+)s; Sleep Time: ([\d.]+)s')
ADAPTER = re.compile(r'Chunk inference ([\d.]+)s; skipped (\d+) stale actions')
TASK = re.compile(r'\[Test(\d+)\] Received new task:')
DONE = re.compile(r'\[Test(\d+)\] Test done with (SUCCESS|FAILURE) in (\d+) steps and (\d+) actions')
CLOCK = re.compile(r'\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}')


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def stats(values):
    if not values:
        return {'samples': 0, 'mean_ms': None, 'p50_ms': None, 'p95_ms': None, 'std_ms': None}
    values = sorted(values)
    def quantile(p):
        position = (len(values) - 1) * p
        low = math.floor(position)
        high = math.ceil(position)
        return values[low] + (values[high] - values[low]) * (position - low)
    return {'samples': len(values), 'mean_ms': statistics.mean(values),
        'p50_ms': quantile(.5), 'p95_ms': quantile(.95),
        'std_ms': statistics.pstdev(values)}


def summarize():
    catalog = read(ROOT / 'catalog.json')
    start = datetime.fromisoformat(read(ROOT / 'parallel_execution.json')['started_utc'])
    groups = {}
    phases = []
    sources = []
    raw = OUT / 'chunk_latency_samples.csv'
    temporary = raw.with_suffix('.csv.tmp')
    fields = ['model', 'phase', 'dimension', 'execution_mode', 'logged_at_utc',
        'episode_sequence', 'worker_step', 'latency_ms', 'intentional_sleep_ms',
        'stale_actions_skipped', 'first_logged_chunk_in_episode', 'post_episode_completion',
        'measurement_boundary', 'source_log', 'line_number']
    with temporary.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in catalog['order']:
            for path in sorted((ROOT / 'results' / name).glob('**/client.log')):
                phase_runtime = read(path.parent / 'runtime.json') if (path.parent / 'runtime.json').exists() else {}
                relative = path.relative_to(ROOT)
                phase = 'pilot' if 'pilot' in relative.parts else 'full'
                dimension = path.parent.name.split('_', 1)[-1] if phase == 'full' else 'pilot'
                episode = None
                post = False
                first = True
                values = []
                steps = actions = completed = 0
                missing_clock = 0
                with path.open(errors='replace') as log:
                    for number, line in enumerate(log, 1):
                        if not line.endswith('\n'):
                            continue  # A concurrently appended line is read next snapshot.
                        task = TASK.search(line)
                        if task:
                            episode = int(task[1]) + 1
                            post = False
                            first = True
                        done = DONE.search(line)
                        if done:
                            completed += 1
                            steps += int(done[3])
                            actions += int(done[4])
                            post = True
                        native = NATIVE.search(line)
                        adapter = None if native else ADAPTER.search(line)
                        if not native and not adapter:
                            continue
                        clock = CLOCK.search(line)
                        if not clock:
                            missing_clock += 1
                            continue
                        timestamp = datetime.fromisoformat(clock[0].replace(',', '.')).replace(tzinfo=LOCAL).astimezone(timezone.utc)
                        mode = phase_runtime.get('execution_mode') or ('parallel_shared' if timestamp >= start else 'serial_split')
                        latency = 1000 * float(native[2] if native else adapter[1])
                        boundary = ('native_host_get_action_chunk_no_explicit_cuda_sync' if native else
                                    'adapter_infer_including_preprocessing_pose_addition_and_cpu_action_copy')
                        writer.writerow({'model': name, 'phase': phase, 'dimension': dimension,
                            'execution_mode': mode, 'logged_at_utc': timestamp.isoformat(),
                            'episode_sequence': episode, 'worker_step': native[1] if native else '',
                            'latency_ms': latency, 'intentional_sleep_ms': 1000 * float(native[3]) if native else '',
                            'stale_actions_skipped': int(adapter[2]) if adapter else '',
                            'first_logged_chunk_in_episode': first, 'post_episode_completion': post,
                            'measurement_boundary': boundary, 'source_log': str(relative), 'line_number': number})
                        if not post:
                            values.append(latency)
                            groups.setdefault((name, mode, phase, boundary), []).append(latency)
                        first = False
                records_path = path.parent / 'trial_results.jsonl'
                records = []
                if records_path.exists():
                    for line in records_path.read_text().splitlines():
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass  # Ignore a partial final write in this passive snapshot.
                timestamps = [datetime.fromisoformat(row['recorded_at_utc']) for row in records]
                span = (timestamps[-1] - timestamps[0]).total_seconds() if len(timestamps) > 1 else None
                phases.append({'model': name, 'phase': phase, 'dimension': dimension, **stats(values),
                    'completed_client_episodes': completed, 'completed_recorded_trials': len(records),
                    'action_coverage': actions / steps if steps else None,
                    'completion_interval_trial_rate_per_hour': (len(timestamps)-1)*3600/span if span and span > 0 else None,
                    'missing_timestamp_chunk_lines': missing_clock,
                    'fully_verified_phase': (path.parent / 'verification.json').exists()})
                stat = path.stat()
                sources.append({'path': str(relative), 'bytes_at_snapshot': stat.st_size,
                    'mtime_ns': stat.st_mtime_ns, 'device': stat.st_dev, 'inode': stat.st_ino})
    temporary.replace(raw)
    aggregated = [{'model': name, 'execution_mode': mode, 'phase': phase,
        'measurement_boundary': boundary, **stats(values)} for (name, mode, phase, boundary), values in groups.items()]
    for name in catalog['order']:
        if not any(row['model'] == name for row in aggregated):
            aggregated.append({'model': name, 'state': 'awaiting_logged_chunks', **stats([])})
    document = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'measurement_class': 'observational_evaluation_logs',
        'publish_as_controlled_KR_overhead': False, 'models': aggregated, 'phases': phases,
        'boundaries': {'native': 'Host timer inside _get_action_chunk; excludes explicit simulator-rate sleep and final CPU transfer; no explicit CUDA synchronization at timer boundaries.',
            'adapter': 'Adapter infer host timer including batch preparation, model prediction, reference-pose addition, validation and CPU action copy; excludes explicit simulator-rate sleep. Only consumed current-generation futures log timing.'},
        'limitations': ['Shared GPU contention affects latency; serial and parallel samples are separated.',
            'Phase-level distributions may mix hardware schedules; use per-sample execution_mode or aggregated model/mode rows.',
            'Post-completion in-flight native chunks are retained in CSV but excluded from summaries.',
            'Native chunks log only when dt_scale > 1. Adapter timings omit discarded futures.',
            'First real chunk of each episode is retained; these are not standardized controlled warmup measurements.',
            'Simulator supplies object velocity; no visual velocity-estimator latency is measured.',
            'Driver memory samples are not PyTorch allocator allocated/reserved peaks.',
            'DOM/KRV also differs in neural velocity conditioning. SmolVLA +/- KR is the cleaner KR ablation.',
            'Checkpoint/phase aggregate differences must not be interpreted as isolated KR overhead.'],
        'sources': sources}
    write(OUT / 'runtime_summary.json', document)
    resource_peaks = {}
    sample_count = 0
    resource_file = OUT / 'gpu_process_samples.jsonl'
    if resource_file.exists():
        for line in resource_file.read_text().splitlines():
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_count += 1
            totals = defaultdict(float)
            for process in sample['processes']:
                if process['model']:
                    role = 'policy' if process['role'].startswith('policy') else 'simulator'
                    totals[(process['model'], role, process['gpu_uuid'])] += process['driver_used_memory_mib']
            for (model, role, gpu), used in totals.items():
                key = model + '/' + role + '/' + gpu
                resource_peaks[key] = max(resource_peaks.get(key, 0), used)
    write(OUT / 'resource_summary.json', {'updated_utc': document['updated_utc'], 'samples': sample_count,
        'sampled_max_driver_memory_mib': resource_peaks,
        'not_allocator_or_true_peak_memory': True,
        'coverage_begins_utc': read(OUT / 'capture_manifest.json')['started_utc']})
    write(OUT / 'probe_snapshots.json', {name: read(path) for name in catalog['order']
        if (path := ROOT / 'results' / name / 'pilot/policy_probe.json').exists()})
    numeric_fields = ['model', 'execution_mode', 'phase', 'measurement_boundary', 'samples', 'mean_ms', 'p50_ms', 'p95_ms', 'std_ms', 'state']
    dest = OUT / 'runtime_summary.csv'
    with dest.with_suffix('.csv.tmp').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=numeric_fields)
        writer.writeheader()
        writer.writerows(aggregated)
    dest.with_suffix('.csv.tmp').replace(dest)
    report = ('# Runtime evidence for the paper\n\n'
        'Passive snapshots of the running Table 1 evaluations. Read-only collection preserves frozen inference '
        'and scoring. Raw samples: `chunk_latency_samples.csv`; definitions, phase coverage and caveats: '
        '`runtime_summary.json`. GPU/process samples begin when the collector starts; past GPU peaks cannot be recovered.\n\n'
        '| Model | Schedule | Phase | Logged chunks | Mean ms | p50 ms | p95 ms |\n'
        '|---|---|---|---:|---:|---:|---:|\n')
    for row in aggregated:
        if not row['samples']:
            report += f"| {row['model']} | pending | pending | 0 | pending | pending | pending |\n"
        else:
            report += (f"| {row['model']} | {row['execution_mode']} | {row['phase']} | {row['samples']} | "
                       f"{row['mean_ms']:.2f} | {row['p50_ms']:.2f} | {row['p95_ms']:.2f} |\n")
    report += ('\nNative and adapter timers have different boundaries. These are observational timings under '
        'the recorded workload, not an isolated KR cost or synchronized end-to-end latency. Do not copy them '
        'into the paper\'s matched runtime table as controlled measurements.\n\n'
        'For that table, measure baseline and KR sequentially on an otherwise idle identical GPU: same fixed '
        'observations, batch size 1, precision, observation window within each pair, 20-action horizon and solver. '
        'Use 20 warmup chunks, at least 200 timed chunks/model across multiple input examples and repeated runs; '
        'synchronize CUDA before/after complete timed regions. Record CPU wall E2E, CUDA policy time, transfers, '
        'velocity estimation (or explicitly oracle velocity), analytic correction, peak allocated/reserved memory '
        'after resetting allocator statistics, and parameter counts. Paired KR overhead is KR minus baseline '
        'in milliseconds and percent, with uncertainty. DOM/KRV requires stating the velocity-conditioning confound.\n\n'
        'The paper already reserves `paper/main.tex` Section Runtime and Failure Analysis / `tab:runtime`; '
        'its numeric entries remain pending until timing boundaries match that protocol.\n')
    (OUT / 'runtime_evidence.md.tmp').write_text(report)
    (OUT / 'runtime_evidence.md.tmp').replace(OUT / 'runtime_evidence.md')
    write(OUT / 'capture_status.json', {'state': 'capturing', 'updated_utc': document['updated_utc'],
        'source_logs': len(sources), 'summarized_logged_chunks': sum(row['samples'] for row in aggregated),
        'evaluation_instrumentation_changed': False})


def gpu_query(fields, kind='gpu'):
    result = subprocess.run(['nvidia-smi', '--query-' + kind + '=' + fields, '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=10, check=True)
    return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))


def process_identity(pid, known, seen=None):
    if seen is None:
        seen = set()
    if pid in seen or len(seen) > 8:
        return {'role': 'unattributed', 'model': None}
    seen.add(pid)
    try:
        argv = Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace').split('\0')
        for flag in ['-a', '--model']:
            if flag in argv and argv[argv.index(flag)+1] in known:
                return {'role': 'policy', 'model': argv[argv.index(flag)+1]}
        if any(arg.endswith('low_clutter_table1_server.py') for arg in argv):
            destination = argv[argv.index('--output_dir')+1]
            relative = Path(destination).relative_to(ROOT / 'results')
            return {'role': 'simulator', 'model': relative.parts[0]}
        parent = next(line.split()[1] for line in Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith('PPid:'))
        identity = process_identity(int(parent), known, seen)
        if identity['model']:
            return {**identity, 'role': identity['role'] + '_worker'}
    except (OSError, ValueError, IndexError, StopIteration):
        pass
    return {'role': 'unattributed', 'model': None}


def sample_resources():
    stamp = datetime.now(timezone.utc).isoformat()
    known = read(ROOT / 'catalog.json')['models']
    state = read(ROOT / 'status.json')
    gpus = gpu_query('index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw')
    processes = gpu_query('pid,gpu_uuid,used_memory', 'compute-apps')
    row = {'sampled_utc': stamp, 'evaluation_state': state['state'],
        'active_evaluations': state.get('active_evaluations', []),
        'gpu_fields': ['index', 'uuid', 'utilization_percent', 'memory_controller_utilization_percent', 'memory_used_mib', 'memory_total_mib', 'power_w'],
        'gpus': gpus, 'processes': [{'pid': int(p[0]), 'gpu_uuid': p[1], 'driver_used_memory_mib': float(p[2]),
            **process_identity(int(p[0]), known)} for p in processes if len(p) == 3 and p[0].isdigit()]}
    with (OUT / 'gpu_process_samples.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    write(OUT / 'latest_resources.json', row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    os.nice(10)
    if not (OUT / 'capture_manifest.json').exists():
        cpu = next((x.split(':', 1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')), 'unknown')
        write(OUT / 'capture_manifest.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
            'root': str(ROOT), 'cpu': cpu, 'logical_cpus_available': len(os.sched_getaffinity(0)),
            'gpu_fields': ['index', 'name', 'uuid', 'driver_version', 'memory_total_mib'],
            'gpus': gpu_query('index,name,uuid,driver_version,memory.total'),
            'gpu_sampling_seconds': 5, 'log_export_seconds': 300, 'log_timezone': str(LOCAL),
            'raw_gpu_memory_kind': 'sampled per-process driver usage; not allocator peak',
            'protocol_sha256': read(ROOT / 'catalog.json')['protocol_sha256']})
    next_export = 0
    while True:
        sample_resources()
        if time.monotonic() >= next_export:
            summarize()
            next_export = time.monotonic() + 300
        if args.once:
            return
        if read(ROOT / 'status.json')['state'] in ['complete', 'failed', 'interrupted']:
            summarize()
            write(OUT / 'capture_stopped.json', {'stopped_utc': datetime.now(timezone.utc).isoformat(),
                'evaluation_state': read(ROOT / 'status.json')['state']})
            return
        time.sleep(5)


if __name__ == '__main__':
    main()
