"""Two-node DOM baseline entry, with a strict same-attempt readiness gate."""
import argparse
from datetime import timedelta
import json
import logging
import os
from pathlib import Path
import sys

sys.path.insert(0, os.environ['DVLA_PROJECT_DIR'])
import torch
import torch.distributed as dist
from core.dom_distributed import initialize_control_group, leader_call
from core.dom_distributed_training import distributed_smoke, train_distributed
from core.dom_training import load_family
from utils.dom_experiment import atomic_json, digest, file_digest
from cache_admission import install

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--family-root', type=Path, required=True)
    p.add_argument('--runtime-file', type=Path, required=True)
    p.add_argument('--command', choices=['smoke', 'train'], required=True)
    args = p.parse_args()
    root = args.family_root
    world, local = int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
    assert world == 8 and int(os.environ['LOCAL_WORLD_SIZE']) == 4 and torch.cuda.device_count() == 4
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    torch.cuda.set_device(local)
    torch.set_num_threads(1)
    device = torch.device('cuda', local)
    props = torch.cuda.get_device_properties(device)
    assert 'GH200' in props.name
    manifest = load_family(root)
    assert manifest['variant'] == 'dom100'
    assert not manifest['velocity_conditioning'] and not manifest['kr']
    selected = json.loads(args.runtime_file.read_text())
    assert digest({k: v for k, v in selected.items() if k != 'fingerprint'}) == selected['fingerprint']
    assert selected['manifest'] == manifest['fingerprint']
    runtime = selected['runtime']
    assert runtime['microbatch'] == 44 and runtime['precision'] == 'bf16'
    recipe = json.loads((root / 'recipe.json').read_text())
    assert digest({k: v for k, v in recipe.items() if k != 'fingerprint'}) == recipe['fingerprint']
    assert recipe['manifest'] == manifest['fingerprint'] and recipe['epochs'] == 10
    assert recipe['effective_batch'] == 352 and recipe['lr'] == 3e-5
    assert recipe['runtime_fingerprint'] == digest(runtime)
    install(root)
    hardware = {'name': props.name, 'total_memory': props.total_memory, 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'capability': [props.major, props.minor]}
    key = digest({'hardware': hardware, 'world_size': world, 'runtime': runtime, 'variant': 'dom100'})[:16]
    dist.init_process_group('nccl', timeout=timedelta(minutes=10))
    try:
        initialize_control_group()
        if args.command == 'smoke':
            distributed_smoke(root, 'dom100', runtime, device)
            return 0
        profile = json.loads((root / 'distributed-profiles' / (key + '.json')).read_text())
        assert digest({k: v for k, v in profile.items() if k != 'fingerprint'}) == profile['fingerprint']
        assert profile['manifest'] == manifest['fingerprint'] and profile['world_size'] == world
        assert profile['benchmark_job_id'] == os.environ['SLURM_JOB_ID'] or (root / 'runs/dom100/latest.json').exists()
        assert profile['benchmark_driver_sha256'] == file_digest(root / 'native_benchmark.py')
        assert all(profile.get(k) for k in ['complete', 'safe', 'full_data_benchmark', 'strict_reload',
            'rank_rng_reload', 'post_training_validation_passed', 'sustained_reload_passed', 'current_velocity_ignored'])
        assert profile['benchmark_samples'] == 70400 and profile['measured_updates'] == 180
        continuation = root / 'runs/dom100' / ('continuation-' + os.environ['SLURM_JOB_ID'] + '.json')
        leader_call(lambda: continuation.unlink(missing_ok=True))
        result = train_distributed(root, 'dom100', runtime, recipe, device)
        if result.get('interrupted'):
            leader_call(lambda: atomic_json(continuation, {'manifest': manifest['fingerprint'], 'world_size': world,
                'updates': result['updates'], 'samples': result['samples'], 'checkpoint_saved': True}))
        return 75 if result.get('interrupted') else 0
    finally:
        dist.destroy_process_group()

if __name__ == '__main__':
    sys.exit(main())
