#!/usr/bin/env python3
"""Explicitly migrate an idle KRV100 run from two to eight training ranks.

Keep model, optimizer, scheduler, sampler cursor and validation history intact.
Archive all original referenced checkpoints before publishing new pointers.
Ranks 0/1 retain their RNG records; ranks 2..7 get reproducible independent
streams derived from the source checkpoint hash and rank. This is a topology
transition, not a claim of numerical equivalence to the two-rank trajectory.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

ROOT = Path(os.environ['KRVLA_DELTA_ROOT'])
sys.path.insert(0, str(ROOT / 'source'))


def read(path):
    return json.loads(path.read_text())


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def new_rng(seed, device):
    import numpy as np
    import torch
    return {'python': random.Random(seed).getstate(),
            'numpy': np.random.RandomState(seed % 2**32).get_state(),
            'torch': torch.Generator(device='cpu').manual_seed(seed).get_state(),
            'cuda': [torch.Generator(device=device).manual_seed(seed).get_state()],
            'cuda_device_only': True}


def main():
    import torch
    from core.dom_checkpoint import run_lock
    from utils.dom_experiment import atomic_json, digest
    assert torch.cuda.is_available(), 'Generate CUDA RNG records on a compute GPU'
    destination = ROOT / 'family/runs/krv100'
    receipt = ROOT / 'topology-transition.json'
    with run_lock(destination):
        if receipt.exists():
            assert read(receipt)['to_world_size'] == 8
            return
        plan = read(ROOT / 'plan.json')
        assert plan['world_size'] == 8 and plan['topology_transition_authorized']
        old_job = str(plan['previous_production_job'])
        active = subprocess.check_output(['squeue', '-h', '-u', 'anayak2', '-o', '%i'], text=True)
        assert old_job not in active.split(), 'Previous training allocation is still active'
        stopped = read(ROOT / 'training-status.json')
        assert stopped['stage'] == 'checkpointed_for_requeue' and stopped['job_id'] == old_job
        archive = ROOT / 'topology-parent-world2'
        archive.mkdir(exist_ok=True)
        intent = archive / 'intent.json'
        if not intent.exists():
            pointers = {p.name: read(p) for p in destination.glob('*.json')
                        if p.name in ('latest.json', 'best.json', 'final.json')
                        or p.name.startswith('epoch-')}
            assert {'latest.json', 'best.json'} <= pointers.keys()
            assert all(p['fingerprint'] == plan['previous_experiment_fingerprint']
                       for p in pointers.values())
            latest = pointers['latest.json']['progress']
            assert all(latest[k] == stopped['progress'][k]
                       for k in ('epoch', 'cursor', 'updates', 'samples'))
            atomic_json(intent, {'pointers': pointers, 'provenance': read(destination / 'provenance.json')})
        original = read(intent)
        recipe = read(ROOT / 'family/recipe.json')
        fingerprint = digest({'manifest': plan['manifest'], 'recipe': recipe,
                              'variant': 'krv100', 'world_size': 8})
        assert fingerprint == plan['experiment_fingerprint']
        audit = {'complete': True, 'from_world_size': 2, 'to_world_size': 8,
                 'effective_batch': 352, 'previous_job_id': old_job,
                 'progress': original['pointers']['latest.json']['progress'],
                 'rng_policy': 'Retain ranks 0/1; ranks 2..7 seeded by SHA256(source-state-hash:rank)',
                 'bitwise_equivalent': False, 'generations': {}}
        provenance = {**original['provenance'], 'fingerprint': fingerprint, 'world_size': 8,
                      'hardware_transition': '2x GH200 to 2 nodes x 4 GH200; global batch 352 unchanged',
                      'previous_experiment_fingerprint': plan['previous_experiment_fingerprint'],
                      'topology_transition': {k: v for k, v in audit.items() if k != 'generations'}}
        replacements = {}
        for pointer in original['pointers'].values():
            name = pointer['generation']
            if name in replacements:
                continue
            backup = archive / name
            if not backup.exists():
                shutil.copytree(destination / 'generations' / name, backup, copy_function=os.link)
            state_hash = sha(backup / 'training_state.pt')
            saved = torch.load(backup / 'training_state.pt', map_location='cpu', weights_only=False)
            assert saved['fingerprint'] == plan['previous_experiment_fingerprint']
            assert saved['distributed']['world_size'] == 2
            assert saved['distributed']['effective_batch'] == 352
            assert saved['progress'] == pointer['progress']
            assert saved['scheduler']['last_epoch'] == saved['progress']['updates']
            assert saved['wandb_id'] == plan['wandb_id']
            records = list(saved['distributed']['rng_by_rank'])
            assert len(records) == 2
            seeds = {}
            for rank in range(2, 8):
                seed = int.from_bytes(hashlib.sha256(f'{state_hash}:{rank}'.encode()).digest()[:8], 'big') % (2**63 - 1)
                records.append(new_rng(seed, torch.device('cuda', 0)))
                seeds[str(rank)] = seed
            saved.update(fingerprint=fingerprint, provenance=provenance,
                         distributed={**saved['distributed'], 'world_size': 8, 'rng_by_rank': records})
            new_name = name + '-world8'
            output = destination / 'generations' / new_name
            if not output.exists():
                temporary = destination / 'generations' / ('.tmp-' + new_name)
                temporary.mkdir(exist_ok=True)
                if not (temporary / 'model.safetensors').exists():
                    os.link(backup / 'model.safetensors', temporary / 'model.safetensors')
                config = read(backup / 'config.json')
                config['experiment_fingerprint'] = fingerprint
                atomic_json(temporary / 'config.json', config)
                atomic_json(temporary / 'provenance.json', provenance)
                torch.save(saved, temporary / 'training_state.pt')
                for file in temporary.iterdir():
                    with file.open('rb') as stream:
                        os.fsync(stream.fileno())
                temporary.rename(output)
            replacements[name] = new_name
            audit['generations'][name] = {'generation': new_name,
                'source_state_sha256': state_hash, 'model_sha256': sha(backup / 'model.safetensors'),
                'new_rank_seeds': seeds}
        # All generations are complete before any pointer changes. Original
        # files stay archived outside the generation garbage collector.
        for name, pointer in original['pointers'].items():
            atomic_json(destination / name, {**pointer, 'fingerprint': fingerprint,
                        'generation': replacements[pointer['generation']]})
        atomic_json(destination / 'provenance.json', provenance)
        atomic_json(receipt, audit)
        print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
