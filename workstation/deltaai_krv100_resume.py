#!/usr/bin/env python3
"""DeltaAI adapter for the full-data KRV100 checkpoint.

The source, family identity, global sampler, optimizer and cosine horizon remain
those of the epoch-two run. Storage paths and W&B identity are adapted, and an
explicit checkpoint transition records the move from two to eight GH200 ranks.
"""
import argparse
from datetime import timedelta
import json
import logging
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(os.environ.get('KRVLA_DELTA_ROOT', Path(__file__).resolve().parent.parent))
FAMILY = ROOT / 'family'
SOURCE = ROOT / 'source'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(ROOT / 'controller'))


def install_paths(compact=True):
    import netrc
    os.environ['WANDB_API_KEY'] = netrc.netrc(os.environ['NETRC']).authenticators('api.wandb.ai')[2]
    import core.dom_training as base
    original = base.load_family

    def relocated(root, check_source=True):
        manifest = original(root, check_source=check_source)
        # Validate the saved bytes first. These two filesystem locations are
        # execution mappings, explicitly recorded outside the frozen manifest.
        manifest['dataset_root'] = os.environ['KRVLA_DELTA_DATA']
        manifest['initialization'] = str(FAMILY / 'initialization')
        return manifest

    base.load_family = relocated
    import core.dom_distributed_training as training
    training.load_family = relocated
    original_policy = base.build_policy

    def checked_policy(*args, **kwargs):
        from utils.dom_experiment import digest
        policy = original_policy(*args, **kwargs)
        reference = json.loads((ROOT / 'parameter-order.json').read_text())
        assert digest({k: v for k, v in reference.items() if k != 'fingerprint'}) == reference['fingerprint']
        actual = [{'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype)}
                  for name, value in policy.named_parameters() if value.requires_grad]
        assert actual == reference['parameters'], 'ARM parameter order or dtype differs from saved optimizer mapping'
        return policy

    base.build_policy = checked_policy
    training.build_policy = checked_policy
    if compact:
        from training_metadata import install_metadata_compaction
        install_metadata_compaction(training)
    return base, training


def check_readiness(runtime, manifest):
    import torch.distributed as dist
    from utils.dom_experiment import digest
    report = json.loads((ROOT / 'readiness.json').read_text())
    assert report['complete'] and report['safe']
    assert report['manifest'] == manifest['fingerprint']
    assert report['runtime'] == runtime and report['world_size'] == dist.get_world_size()
    assert report['strict_reload'] and report['rank_rng_reload']
    transition = json.loads((ROOT / 'topology-transition.json').read_text())
    assert report['resumed_from_update'] == transition['progress']['updates']
    assert report['full_dataset_frames'] == manifest['train_frames']
    assert digest({k: v for k, v in report.items() if k != 'fingerprint'}) == report['fingerprint']
    return report


def recover_completed(destination, control, control_sha, manifest, plan):
    """Never enter another epoch after a completed cap/early-stop checkpoint."""
    import torch
    from core.dom_checkpoint import resolve_generation
    pointer = json.loads((destination / 'latest.json').read_text())
    progress = pointer['progress']
    limit = control['stop_after_completed_epochs']
    assert progress['epoch'] <= limit, 'Checkpoint exceeds the authorized epoch cap'
    terminal = (progress['epoch'] == limit or progress.get('early_stopped'))
    if not terminal or progress.get('pending_validation'):
        return None
    assert progress['cursor'] == 0, 'Completed checkpoint must be at an epoch boundary'
    if progress.get('completion_reason') == 'user_epoch_limit':
        assert progress['epoch_limit_control_sha256'] == control_sha
    path = resolve_generation(destination)
    state = torch.load(path / 'training_state.pt', map_location='cpu', weights_only=False)
    assert state['progress'] == progress and state['fingerprint'] == pointer['fingerprint']
    assert state['wandb_id'] == plan['wandb_id']
    return {**progress, 'fingerprint': state['fingerprint'], 'best': state['best'],
            'checkpoint': str(path), 'manifest': manifest['fingerprint'],
            'original_scheduler_horizon_epochs': control['scheduler_horizon_epochs']}


def probe(base, training, runtime, device, manifest):
    """Exercise the actual epoch-two optimizer on the entire training sampler."""
    import gc
    import math
    from itertools import islice
    import torch
    import torch.distributed as dist
    from core.dom_checkpoint import resolve_generation
    from core.dom_distributed import leader_call, save_distributed
    from utils.dom_experiment import atomic_json, digest
    split = json.loads((FAMILY / 'split.json').read_text())['splits']
    recipe = json.loads((FAMILY / 'recipe.json').read_text())
    destination = FAMILY / 'runs/krv100'
    world = dist.get_world_size()
    start = json.loads((destination / 'latest.json').read_text())['progress']
    if world == 8:
        transition = json.loads((ROOT / 'topology-transition.json').read_text())
        assert start == transition['progress']
    else:
        assert world == 2, 'Interactive preflight uses two GPUs'
    fingerprint = digest({'manifest': manifest['fingerprint'], 'recipe': recipe,
                          'variant': 'krv100', 'world_size': world})
    base.seed_all()
    data = training.build_dataset(FAMILY, split['train'], training=True)
    assert len(data) == manifest['train_frames'] == 23080253
    policy = base.build_policy(FAMILY, 'krv100', device)
    assert policy.config.condition_on_object_velocity and policy.config.enable_residual_flow
    optimizer = base.make_optimizer(policy, recipe['lr'], runtime['fused'])
    scheduler = base.make_scheduler(optimizer, recipe['epochs'] * math.ceil(len(data) / 352))
    saved = training.load_generation(destination, policy, optimizer, scheduler,
                                     fingerprint, restore_random=False)
    assert saved['progress'] == start and saved['distributed']['world_size'] == world
    assert scheduler.last_epoch == saved['progress']['updates']
    initial_lr = optimizer.param_groups[0]['lr']
    model = training.wrap_policy(policy, device)
    training.restore_distributed(saved)
    progress = dict(saved['progress'])
    del saved
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    loader = training.distributed_loader(data, runtime, epoch=start['epoch'], cursor=start['cursor'],
                                          sampling=recipe['sampling'])
    records = []
    iterator = training.updates(model, optimizer, scheduler, loader, progress,
                                runtime['precision'], device)
    for record in islice(iterator, 8):
        records.append(record)
        logging.info('probe rank=%d update=%d loss=%g rate=%g', dist.get_rank(),
                     progress['updates'], record['loss'], record['samples_per_second'])
    iterator.close()
    del iterator, loader
    gc.collect()
    assert progress['updates'] == start['updates'] + 8
    assert progress['samples'] == start['samples'] + 8 * 352
    assert progress['cursor'] == start['cursor'] + 8 * 352
    # Canonical buffers are shared solely for the checkpoint-reload comparison.
    for buffer in policy.buffers():
        dist.broadcast(buffer, src=0)
    small = next(iter(training.distributed_loader(data, {**runtime, 'microbatch': 2, 'workers': 0},
                                                  epoch=start['epoch'], lengths=[world * 2])))
    small = base.to_device(small, device)
    noise = base.fixed_noise(small, policy, device)
    policy.eval()
    with torch.no_grad(), base.amp_context(runtime['precision'], device):
        policy.reset()
        before = policy.predict_action_chunk(small, noise=noise)
    probe_root = ROOT / f'probe-checkpoint-world{world}'
    save_distributed(probe_root, policy, optimizer, scheduler, progress, fingerprint,
                     None, best=None, device=device)
    expected_random = torch.rand(4, device=device)
    reloaded = training.load_generation(probe_root, policy, optimizer, scheduler,
                                       fingerprint, restore_random=False)
    training.restore_distributed(reloaded)
    torch.testing.assert_close(expected_random, torch.rand(4, device=device), rtol=0, atol=0)
    with torch.no_grad(), base.amp_context(runtime['precision'], device):
        policy.reset()
        torch.testing.assert_close(before, policy.predict_action_chunk(small, noise=noise))
    validation = None
    if dist.get_rank() == 0:
        val = training.build_dataset(FAMILY, split['validation'])
        anchors = json.loads((FAMILY / 'validation_anchors.json').read_text())[:8]
        validation = base.evaluate(policy, base.make_loader(val, {**runtime, 'workers': 0},
                                                           indices=anchors), runtime['precision'], device)
    validation = leader_call(lambda: validation)
    properties = torch.cuda.get_device_properties(device)
    peak = torch.cuda.max_memory_reserved(device)
    safe = torch.tensor(int(peak <= .85 * properties.total_memory), device=device)
    dist.all_reduce(safe, op=dist.ReduceOp.MIN)
    assert bool(safe), 'GH200 probe lacks 15% GPU memory headroom'
    hardware = [None] * world
    dist.all_gather_object(hardware, {'name': properties.name, 'total_memory': properties.total_memory,
                                   'torch': torch.__version__, 'cuda': torch.version.cuda,
                                   'peak_gpu_bytes': peak})
    smoke_reports = [json.loads(p.read_text()) for p in (FAMILY / 'distributed-profiles').glob('*.json')]
    assert any(p.get('safe') and p.get('strict_reload') and p.get('rank_rng_reload')
               and p.get('future_inputs_ignored') and p.get('runtime') == runtime
               and p.get('manifest') == manifest['fingerprint']
               and p.get('world_size') == world for p in smoke_reports)
    report = {'complete': True, 'safe': True, 'manifest': manifest['fingerprint'],
              'runtime': runtime, 'world_size': world, 'hardware_by_rank': hardware,
              'strict_reload': True, 'rank_rng_reload': True,
              'optimizer_parameter_order_verified': True,
              'resumed_from_update': start['updates'], 'resumed_lr': initial_lr,
              'full_dataset_frames': len(data), 'measured_updates': 8,
              'records': records, 'validation_after_updates': validation,
              'production_checkpoint_untouched': str(resolve_generation(destination)),
              'job_id': os.environ['SLURM_JOB_ID']}
    report['fingerprint'] = digest(report)
    report_path = ROOT / ('readiness.json' if world == 8 else 'interactive-readiness.json')
    leader_call(lambda: atomic_json(report_path, report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('smoke', 'probe', 'train'))
    args = parser.parse_args()
    # The smoke fixture performs its own per-episode statistics compaction.
    # Production compaction removes those unused records entirely.
    base, training = install_paths(compact=args.phase != 'smoke')
    runtime = json.loads((ROOT / 'runtime.json').read_text())
    if args.phase == 'smoke':
        # Use the existing real-update, uneven-tail, future-input and RNG test.
        sys.argv = ['readiness_krv100.py', '--source', str(SOURCE), '--family-root', str(FAMILY),
                    '--runtime', json.dumps(runtime)]
        runpy.run_path(str(ROOT / 'controller/readiness_krv100.py'), run_name='__main__')
        return 0
    import torch
    import torch.distributed as dist
    from utils.dom_experiment import atomic_json, file_digest
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    world = int(os.environ['WORLD_SIZE'])
    assert world in (2, 8)
    assert int(os.environ['LOCAL_WORLD_SIZE']) == (4 if world == 8 else 1)
    assert args.phase != 'train' or world == 8, 'Production requires eight GPUs'
    local = int(os.environ['LOCAL_RANK'])
    torch.set_num_threads(1)
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    assert 'GH200' in torch.cuda.get_device_properties(device).name
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    manifest = base.load_family(FAMILY)
    if args.phase == 'probe':
        probe(base, training, runtime, device, manifest)
        dist.destroy_process_group()
        return 0
    report = check_readiness(runtime, manifest)
    actual_hardware = [None] * dist.get_world_size()
    properties = torch.cuda.get_device_properties(device)
    dist.all_gather_object(actual_hardware, {'name': properties.name, 'total_memory': properties.total_memory,
                                           'torch': torch.__version__, 'cuda': torch.version.cuda})
    for measured, actual in zip(report['hardware_by_rank'], actual_hardware):
        assert all(measured[k] == v for k, v in actual.items())
    verified = json.loads((ROOT / 'verified-data.json').read_text())
    assert verified['complete'] and verified['manifest'] == manifest['fingerprint']
    recipe = json.loads((FAMILY / 'recipe.json').read_text())
    from epoch_training_control import EpochLimitReached, install_epoch_limit, load_control
    control, control_sha = load_control(FAMILY, manifest, recipe, file_digest)
    assert control['stop_after_completed_epochs'] == 10
    destination = FAMILY / 'runs/krv100'
    plan = json.loads((ROOT / 'plan.json').read_text())
    from core.dom_distributed import leader_call
    completed = leader_call(lambda: recover_completed(destination, control, control_sha, manifest, plan))
    if completed is not None:
        leader_call(lambda: atomic_json(destination / 'training_complete.json', completed))
        leader_call(lambda: atomic_json(ROOT / 'training-status.json',
                    {'stage': 'complete', 'wandb_id': plan['wandb_id'], 'progress': completed}))
        dist.destroy_process_group()
        return 0
    from train_krv100 import install_checkpoint_cadence
    install_checkpoint_cadence(training, destination)
    install_epoch_limit(training, destination, control, control_sha)
    import wandb
    if not callable(getattr(wandb.util, 'generate_id', None)):
        from wandb.sdk.lib.runid import generate_id
        wandb.util.generate_id = generate_id
    init = wandb.init
    inherited_provenance = json.loads((destination / 'provenance.json').read_text())
    original_publish = training.atomic_json

    def publish(path, value):
        if Path(path) == destination / 'provenance.json':
            value = {**inherited_provenance, **value}
        return original_publish(path, value)

    training.atomic_json = publish

    def init_writer(**kwargs):
        assert kwargs['id'] == plan['wandb_id']
        kwargs.update(name='KRV100_Full_Dataset', resume='must', allow_val_change=True,
                      tags=['full-dataset', 'epoch002-resume', 'deltaai', 'krv100'])
        kwargs['config'].update(parent_run_id='2qh0ijb1', parent_epoch=2,
                                parent_updates=131138, maximum_total_epochs=10,
                                original_scheduler_horizon_epochs=100,
                                topology_transition=json.loads((ROOT / 'topology-transition.json').read_text()),
                                storage_mapping='validated frozen manifest with relocated absolute paths')
        writer = init(**kwargs)
        writer.summary.update({'Resume/parent_epoch': 2, 'Resume/parent_updates': 131138,
                               'Readiness/full_data_files_verified': verified['files_verified'],
                               'Readiness/resumed_optimizer_probe_updates': report['measured_updates'],
                               'Readiness/strict_reload': True, 'Readiness/rank_rng_reload': True,
                               'Resume/optimizer_parameter_order_verified': True,
                               'Training/maximum_total_epochs': 10})
        return writer

    wandb.init = init_writer
    leader_call(lambda: atomic_json(ROOT / 'training-status.json',
                {'stage': 'training', 'job_id': os.environ['SLURM_JOB_ID'], 'wandb_id': plan['wandb_id']}))
    try:
        result = training.train_distributed(FAMILY, 'krv100', runtime, recipe, device)
    except EpochLimitReached as stop:
        result = stop.result
    if result.get('interrupted'):
        leader_call(lambda: atomic_json(ROOT / 'training-status.json',
                    {'stage': 'checkpointed_for_requeue', 'job_id': os.environ['SLURM_JOB_ID'],
                     'wandb_id': plan['wandb_id'], 'progress': result}))
    else:
        leader_call(lambda: atomic_json(destination / 'training_complete.json', result))
        leader_call(lambda: atomic_json(ROOT / 'training-status.json',
                    {'stage': 'complete', 'wandb_id': plan['wandb_id'], 'progress': result}))
    dist.destroy_process_group()
    return 75 if result.get('interrupted') else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
