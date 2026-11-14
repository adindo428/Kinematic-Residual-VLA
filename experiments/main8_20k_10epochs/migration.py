"""Restore trained weights, Adam moments, LR scheduler and selected rank RNG.

The subset is a new sampling stage. Historical full-data epochs do not count
toward its ten-epoch minimum. Parent normalization and drift centering remain
fixed to preserve the meaning of the restored weights and optimizer state.
"""
import gc
import json
from pathlib import Path
import torch
from common import sha, atomic
from subset import require
from budget import initial_progress


def restore_parent(root, recipe, policy, optimizer, scheduler, rank, restore_rng):
    root = Path(root)
    parent = root / 'parents' / recipe['name']
    migration = json.loads((parent / 'migration.json').read_text())
    path = parent / migration['checkpoint_file']
    require(sha(path) == migration['checkpoint_sha256'], 'Frozen parent checkpoint changed')
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if recipe['method'] == 'dom':
        from safetensors.torch import load_model
        weights = parent / 'model.safetensors'
        require(sha(weights) == migration['weights_sha256'], 'DOM trained weights changed')
        load_model(policy.policy, str(weights), strict=True)
        require(state['fingerprint'] == migration['identity'], 'DOM checkpoint fingerprint differs')
        saved_rng = state['distributed']['rng_by_rank']
    else:
        require(state['identity'] == migration['identity'], 'Parent checkpoint identity differs')
        policy.load_state_dict(state['model'], strict=True)
        saved_rng = state['rng']
    require(state['progress'] == migration['progress'], 'Parent progress differs from saved receipt')
    require(state['progress']['updates'] == recipe['parent_updates'] and state['progress']['samples'] == recipe['parent_samples'], 'Parent budget origin differs')
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    gc.collect()
    torch.cuda.empty_cache()
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    require(scheduler.last_epoch == recipe['parent_updates'], 'Parent scheduler counter differs')
    # Existing ranks retain their streams. New ranks on a larger allocation
    # get deterministic independent streams because no saved stream exists.
    rng = saved_rng[rank % len(saved_rng)]
    if isinstance(rng['cuda'], list):
        rng = {**rng, 'cuda': rng['cuda'][0]}
    restore_rng(rng)
    if rank >= len(saved_rng):
        import random
        import numpy as np
        seed=recipe['seed']+recipe['parent_updates']+rank
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if rank == 0:
        atomic(root / 'runs' / recipe['name'] / 'subset-migration.json', {
            'parent': migration, 'parent_optimizer_scheduler_loaded': True,
            'parent_normalization_preserved': True, 'existing_parent_rank_rng_restored': True,
            'additional_rank_streams_initialized': max(0,recipe['world_size']-len(saved_rng)),
            'subset_sampler_reset': True, 'subset_progress': initial_progress(recipe),
            'minimum_additional_subset_epochs': recipe['min_epochs'],
            'prior_full_data_exposure': True, 'world_size_changed': len(saved_rng) != recipe['world_size']})
    return initial_progress(recipe)
