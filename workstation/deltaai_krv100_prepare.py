#!/usr/bin/env python3
"""Verify transferred bytes and prepare an auditable DeltaAI continuation."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import netrc
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(os.environ['KRVLA_DELTA_ROOT'])
FAMILY = ROOT / 'family'
DATA = Path(os.environ['KRVLA_DELTA_DATA'])


def read(path):
    return json.loads(path.read_text())


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def checkpoint():
    import fcntl
    with (ROOT / 'migration.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _checkpoint()


def _checkpoint():
    import torch
    plan = read(ROOT / 'plan.json')
    receipt = ROOT / 'migration.json'
    if receipt.exists():
        assert read(receipt)['wandb_id'] == plan['wandb_id']
        return
    parent = ROOT / 'parent-checkpoint'
    original = FAMILY / 'runs/krv100/generations' / plan['source_generation']
    source = parent if parent.exists() else original
    for name, expected in plan['checkpoint_files'].items():
        assert sha(source / name) == expected, name
    state = torch.load(source / 'training_state.pt', map_location='cpu', weights_only=False)
    assert state['progress'] == plan['pointer']['progress']
    assert state['fingerprint'] == plan['experiment_fingerprint']
    assert state['wandb_id'] == '2qh0ijb1' and state['distributed']['world_size'] == 2
    assert state['scheduler']['last_epoch'] == 131138
    if not parent.exists():
        original.rename(parent)
    destination = FAMILY / 'runs/krv100'
    generation = 'u000131138-deltaai'
    output = destination / 'generations' / generation
    output.mkdir(parents=True, exist_ok=True)
    for name in ('model.safetensors',):
        if not (output / name).exists():
            os.link(parent / name, output / name)
    config = read(parent / 'config.json')
    config.update(wandb_id=plan['wandb_id'])
    atomic(output / 'config.json', config)
    provenance = read(parent / 'provenance.json')
    provenance.update(wandb_id=plan['wandb_id'], parent_wandb_id='2qh0ijb1',
                      parent_checkpoint_sha256=plan['checkpoint_files'],
                      storage_mapping={'dataset_root': str(DATA), 'initialization': str(FAMILY / 'initialization')},
                      maximum_total_epochs=10, hardware_transition='2x A6000 to 2x GH200; not bitwise equivalent')
    state.update(wandb_id=plan['wandb_id'], provenance=provenance)
    # Weights, moments, LR, scheduler, progress and both rank RNG records are
    # unchanged. The epoch-limit adapter clears the old two-epoch stop marker.
    torch.save(state, output / 'training_state.pt')
    atomic(output / 'provenance.json', provenance)
    atomic(destination / 'provenance.json', provenance)
    pointer = {**plan['pointer'], 'generation': generation}
    for name in ('latest.json', 'best.json', 'epoch-002.json'):
        atomic(destination / name, pointer)
    key = netrc.netrc(os.environ['NETRC']).authenticators('api.wandb.ai')[2]
    os.environ['WANDB_API_KEY'] = key
    import wandb
    api = wandb.Api(timeout=30)
    assert api.viewer.username == 'aparame'
    run = wandb.init(entity='aparame-clemson-university', project='Kinematic-Residual-VLA',
                     id=plan['wandb_id'], resume='allow', name='KRV100_Full_Dataset',
                     dir=str(ROOT / 'logs'), config={'parent_run_id': '2qh0ijb1',
                     'parent_epoch': 2, 'parent_updates': 131138, 'full_dataset': True,
                     'train_episodes': 203151, 'validation_episodes': 2076,
                     'maximum_total_epochs': 10, 'scheduler_horizon_epochs': 100,
                     'manifest_fingerprint': plan['manifest'], 'world_size': 2})
    run.summary.update({'Resume/parent_epoch': 2, 'Resume/parent_updates': 131138,
                        'Resume/verified_optimizer_and_rng': True})
    url = run.url
    run.finish()
    atomic(receipt, {'complete': True, 'wandb_id': plan['wandb_id'], 'wandb_url': url,
                    'parent': str(parent), 'parent_files_sha256': plan['checkpoint_files'],
                    'preserved': ['model', 'optimizer', 'scheduler', 'progress', 'rng_by_rank',
                                  'normalization', 'full_dataset_split', 'early_stopping_history'],
                    'changed': ['W&B branch identity', 'absolute storage path mapping',
                                'GPU architecture and software environment'],
                    'runtime': read(ROOT / 'runtime.json'),
                    'maximum_total_epochs': 10})
    print(json.dumps(read(receipt)), flush=True)


def verify_data():
    plan = read(ROOT / 'plan.json')
    manifest = read(FAMILY / 'manifest.json')
    split = read(FAMILY / 'split.json')['splits']
    episodes = set(split['train'] + split['validation'])
    parquets = read(FAMILY / 'parquet_hashes.json')
    phases = read(FAMILY / 'phase_hashes.json')
    videos = read(FAMILY / 'video_hashes.json')
    tasks = [(DATA / name, expected) for name, expected in plan['metadata_hashes'].items()]
    for ep in sorted(episodes):
        tasks.append((DATA / f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet', parquets[str(ep)]))
        tasks.append((FAMILY / f'phase_cache/{ep:06d}.npz', phases[str(ep)]))
    for name, item in videos.items():
        if int(Path(name).stem.split('_')[-1]) in episodes:
            tasks.append((DATA / name, item['sha256']))
    started = time.monotonic()
    failures = []

    def check(item):
        path, expected = item
        try:
            return None if sha(path) == expected else {'path': str(path), 'error': 'sha256 mismatch'}
        except OSError as error:
            return {'path': str(path), 'error': str(error)}

    with ThreadPoolExecutor(max_workers=8) as pool:
        for count, result in enumerate(pool.map(check, tasks), 1):
            if result:
                failures.append(result)
            if count % 10000 == 0:
                atomic(ROOT / 'verification-progress.json', {'checked': count, 'total': len(tasks),
                       'failures': len(failures), 'seconds': time.monotonic() - started})
    receipt = {'complete': not failures, 'manifest': manifest['fingerprint'],
               'dataset_revision': manifest['dataset_revision'], 'dataset_root': str(DATA),
               'training_episodes': len(split['train']), 'validation_episodes': len(split['validation']),
               'files_verified': len(tasks), 'errors': failures, 'seconds': time.monotonic() - started}
    atomic(ROOT / 'verified-data.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'errors'}), flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} transferred files failed verification; see verified-data.json')


if __name__ == '__main__':
    {'checkpoint': checkpoint, 'verify-data': verify_data}[sys.argv[1]]()
