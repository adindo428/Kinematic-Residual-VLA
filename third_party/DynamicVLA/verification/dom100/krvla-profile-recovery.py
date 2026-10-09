"""Finish canonical profiling, retaining completed exact-source measurements."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--previous', required=True)
args = parser.parse_args()
root = Path(os.environ['DOM100_ROOT'])
project = Path(os.environ['DVLA_PROJECT_DIR'])
sys.path.insert(0, str(project))
from core.dom_training import load_family
from core import dom_profiling
from utils.dom_experiment import atomic_json, digest

manifest = load_family(root)
previous = root/'profiling'/args.previous
cached = json.loads((previous/'trials.json').read_text())
output = root/'profiling'/os.environ['SLURM_JOB_ID']
output.mkdir(parents=True, exist_ok=True)
canonical_run = subprocess.run
events = []


def bounded_trial(command, **kwargs):
    if 'profile-trial' not in command:
        return canonical_run(command, **kwargs)
    runtime = json.loads(command[command.index('--runtime')+1])
    mode = command[command.index('--mode')+1]
    variant = command[command.index('--variant')+1]
    destination = Path(command[command.index('--output')+1])
    for i, row in enumerate(cached):
        if row.get('runtime') == runtime and row.get('mode') == mode and row.get('variant') == variant and row.get('complete'):
            atomic_json(destination, row)
            events.append({'destination': str(destination), 'reused': str(previous/f'trial-{i:03d}.json')})
            atomic_json(output/'recovery-events.json', events)
            return subprocess.CompletedProcess(command, 0, stdout=(previous/f'trial-{i:03d}.log').read_text())
    if runtime['workers'] > 4:
        row = {'complete': False, 'runtime': runtime, 'oom': False,
               'error': 'Rejected: eight-worker candidates exhausted the 64 GB allocation in the preceding campaign.'}
        atomic_json(destination, row)
        return subprocess.CompletedProcess(command, 1, stdout=json.dumps(row))
    # Never wait indefinitely for a decoder, worker, or process pipe. A file
    # avoids descendants holding an inherited stdout pipe open after exit.
    with tempfile.TemporaryFile(mode='w+') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            status = process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            row = {'complete': False, 'runtime': runtime, 'oom': False, 'error': 'Trial exceeded its 600-second watchdog.'}
            atomic_json(destination, row)
            status = 124
        # Reap remaining descendants even when a failed parent exits first.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stream.seek(0)
        transcript = stream.read()
    events.append({'destination': str(destination), 'returncode': status, 'watchdog_seconds': 600})
    atomic_json(output/'recovery-events.json', events)
    return subprocess.CompletedProcess(command, status, stdout=transcript)


atomic_json(output/'recovery-provenance.json', {
    'manifest': manifest['fingerprint'], 'source_sha256': manifest['source_sha256'],
    'previous_trials': str(previous), 'canonical_model_and_trial_unchanged': True,
    'recovery_script_sha256': __import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),
    'maximum_workers': 4, 'trial_watchdog_seconds': 600,
})
dom_profiling.subprocess.run = bounded_trial
profile = dom_profiling.profile(root, output)
print(json.dumps(profile), flush=True)
