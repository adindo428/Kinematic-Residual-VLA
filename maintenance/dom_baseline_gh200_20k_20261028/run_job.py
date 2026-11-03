"""Check immutable inputs, prove readiness, then enter the native DOM trainer."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
FAMILY = ROOT / 'family'
sys.path.insert(0, str(ROOT / 'source'))
from utils.dom_experiment import atomic_json, digest, file_digest
from core.dom_training import load_family

def main():
    job = os.environ['SLURM_JOB_ID']
    assert int(os.environ['SLURM_JOB_NUM_NODES']) == 2
    lock = json.loads((ROOT / 'implementation-lock.json').read_text())
    for name, expected in lock['files'].items():
        assert file_digest(ROOT / name) == expected, name
    manifest = load_family(FAMILY)
    assert manifest['fingerprint'] == lock['manifest']
    assert file_digest(FAMILY / 'initialization/model.safetensors') == manifest['weights_sha256']
    for name in ('recipe', 'selected-runtime'):
        value = json.loads((FAMILY / (name + '.json')).read_text())
        assert digest({k: v for k, v in value.items() if k != 'fingerprint'}) == value['fingerprint']
        assert value['fingerprint'] == lock['recipe' if name == 'recipe' else 'runtime']
    state = {'job_id': job, 'account': os.environ['SLURM_JOB_ACCOUNT'], 'nodes': 2,
        'world_size': 8, 'variant': 'dom100', 'velocity_conditioning': False, 'kr': False,
        'training_episodes': 20000, 'epochs': 10, 'training_started': False,
        'manifest': manifest['fingerprint'], 'recipe': lock['recipe'],
        'stage': 'checking', 'wandb_mode': 'offline',
        'versions': {p: importlib.metadata.version(p) for p in
            ['torch', 'torchcodec', 'transformers', 'huggingface_hub', 'lerobot']}}
    atomic_json(ROOT / 'status.json', state)
    if not (FAMILY / 'runs/dom100/latest.json').exists():
        deadline = time.monotonic() + 30 * 60
        for phase in ('smoke', 'benchmark'):
            log = ROOT / 'logs' / f'{phase}-{job}.log'
            state.update(stage=phase, log=str(log))
            atomic_json(ROOT / 'status.json', state)
            with log.open('ab', buffering=0) as output:
                result = subprocess.run(['timeout', '--signal=TERM', '--kill-after=30s',
                    str(max(1, int(deadline-time.monotonic()))) + 's',
                    'bash', str(FAMILY / 'run_nodes.sh'), phase], stdout=output, stderr=subprocess.STDOUT)
            state[phase + '_exit'] = result.returncode
            atomic_json(ROOT / 'status.json', state)
            if result.returncode:
                state.update(stage='readiness_failed', failed_phase=phase)
                atomic_json(ROOT / 'status.json', state)
                return result.returncode
        profiles = [json.loads(p.read_text()) for p in (FAMILY / 'distributed-profiles').glob('*.json')]
        profile = next(p for p in profiles if p.get('benchmark_job_id') == job)
        assert profile['world_size'] == 8 and profile['variant'] == 'dom100'
        assert profile['manifest'] == manifest['fingerprint']
        assert profile['current_velocity_ignored'] and profile['sustained_reload_passed']
        assert profile['post_training_validation_passed'] and profile['safe']
        atomic_json(ROOT / 'readiness.json', profile)
    else:
        state['resume'] = True
    state.update(stage='training_starting', training_started=True)
    atomic_json(ROOT / 'status.json', state)
    os.execv('/bin/bash', ['bash', str(FAMILY / 'run_nodes.sh'), 'train'])

if __name__ == '__main__':
    sys.exit(main())
