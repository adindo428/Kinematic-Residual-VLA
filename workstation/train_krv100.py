#!/usr/bin/env python3
"""Launch frozen numerical training after shared two-GPU readiness approval."""
import argparse
from datetime import timedelta
import json
import logging
import os
from pathlib import Path
import sys
import time


def install_checkpoint_cadence(training, destination, interval_seconds=300):
    """Save through the existing atomic writer only at optimizer boundaries."""
    original_save, original_updates = training.save_distributed, training.updates
    original_load = training.load_generation
    checkpoint = {'arguments': None, 'saved_at': time.monotonic(), 'best': None}

    def load(*args, **kwargs):
        result = original_load(*args, **kwargs)
        checkpoint['best'] = result['best']
        return result

    def save(root, policy, optimizer, scheduler, progress, fingerprint, wandb_id,
             best=None, milestone=False, device='cuda', final_checkpoint=False):
        result = original_save(root, policy, optimizer, scheduler, progress, fingerprint,
                               wandb_id, best, milestone, device, final_checkpoint)
        checkpoint['arguments'] = (root, policy, optimizer, scheduler, progress,
                                   fingerprint, wandb_id, best, device)
        checkpoint['saved_at'] = time.monotonic()
        return result

    def updates(*args, **kwargs):
        import torch
        import torch.distributed as dist
        if checkpoint['arguments'] is None:
            # A resumed run already has its initial checkpoint and skips save().
            model, optimizer, scheduler, _, progress, _, device = args
            provenance = json.loads((destination/'provenance.json').read_text())
            checkpoint['arguments'] = (destination, model.module, optimizer, scheduler,
                                       progress, provenance['fingerprint'], provenance['wandb_id'],
                                       checkpoint['best'], device)
        for record in original_updates(*args, **kwargs):
            # The frozen loop has completed optimizer.step and cleared grads.
            # All ranks agree on the save; reuse its progress and RNG writer.
            due = torch.tensor(int(time.monotonic()-checkpoint['saved_at'] >= interval_seconds),
                               device=args[-1])
            dist.all_reduce(due, op=dist.ReduceOp.MAX)
            if bool(due):
                values = checkpoint['arguments']
                if values is None:
                    raise RuntimeError('Missing canonical checkpoint arguments')
                root, policy, optimizer, scheduler, progress, fingerprint, run_id, best, device = values
                save(root, policy, optimizer, scheduler, progress, fingerprint,
                     run_id, best=best, device=device)
                if dist.get_rank() == 0:
                    logging.info('Saved periodic checkpoint at optimizer update %d', progress['updates'])
            yield record

    training.save_distributed, training.updates, training.load_generation = save, updates, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--family-root', type=Path, required=True)
    parser.add_argument('--variant', choices=('krv100',), default='krv100')
    parser.add_argument('--command', choices=('train',), default='train')
    parser.add_argument('--microbatch', type=int, required=True)
    parser.add_argument('--workers', type=int, choices=(0,1,2,4,8), required=True)
    parser.add_argument('--prefetch', type=int, choices=(2,4), required=True)
    parser.add_argument('--pin-memory', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fused', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--performance-profile', type=Path, required=True)
    parser.add_argument('--resume-stopped', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    import torch.distributed as dist
    from torch.distributed.elastic.multiprocessing.errors import record
    from core.dom_training import load_family
    import core.dom_distributed_training as training
    from core.dom_distributed import leader_call
    from utils.dom_experiment import atomic_json, digest, file_digest

    @record
    def run():
        logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
        world, local = int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
        if world!=2 or int(os.environ['LOCAL_WORLD_SIZE'])!=2 or not 1<=args.microbatch<=176:
            raise ValueError('This measured workstation launch requires exactly two GPUs and batch 352')
        torch.set_num_threads(1)
        torch.cuda.set_device(local)
        device = torch.device('cuda',local)
        properties = torch.cuda.get_device_properties(device)
        dist.init_process_group('nccl',timeout=timedelta(minutes=30))
        hardware = {'name':properties.name,'total_memory':properties.total_memory,
                    'torch':torch.__version__,'cuda':torch.version.cuda,
                    'capability':[properties.major,properties.minor]}
        actual_hardware = [None]*world
        dist.all_gather_object(actual_hardware,hardware)
        root = args.family_root.resolve()
        manifest = load_family(root)
        recipe = json.loads((root/'recipe.json').read_text())
        checked = dict(recipe);expected = checked.pop('fingerprint')
        if digest(checked)!=expected or recipe['manifest']!=manifest['fingerprint']:
            raise ValueError('Training recipe differs from frozen provenance')
        from epoch_training_control import EpochLimitReached, install_epoch_limit, load_control
        control, control_sha = load_control(root,manifest,recipe,file_digest)
        runtime = {'microbatch':args.microbatch,'workers':args.workers,'prefetch':args.prefetch,
                   'pin_memory':args.pin_memory,'fused':args.fused,'precision':recipe['precision']}
        performance = json.loads(args.performance_profile.read_text())
        checked = dict(performance);expected = checked.pop('fingerprint')
        adapter_sha = file_digest(Path(__file__).with_name('readiness_krv100.py'))
        metadata_sha = file_digest(Path(__file__).with_name('training_metadata.py'))
        if (digest(checked)!=expected or not performance.get('complete') or not performance.get('safe')
                or not performance.get('correctness_checks_passed') or len(performance.get('confirmation',[]))!=2
                or performance['manifest']!=manifest['fingerprint'] or performance['variant']!=args.variant
                or performance['world_size']!=world or performance['runtime']!=runtime
                or performance['hardware_by_rank']!=actual_hardware
                or performance['execution_controller'].get('metadata_adapter_sha256')!=metadata_sha
                or performance['execution_controller']['readiness_adapter_sha256']!=adapter_sha):
            raise ValueError('Measured throughput approval differs from this data, runtime or both GPUs')
        # Both ranks use the leader's report key; their exact hardware identities
        # remain independently checked above and in the shared readiness record.
        key = digest({'hardware':actual_hardware[0],'world_size':world,'runtime':runtime,'variant':args.variant})[:16]
        readiness = json.loads((root/'distributed-profiles'/f'{key}.json').read_text())
        checked = dict(readiness);expected = checked.pop('fingerprint')
        if (digest(checked)!=expected or not readiness.get('complete') or not readiness.get('safe')
                or readiness['manifest']!=manifest['fingerprint'] or readiness['runtime']!=runtime
                or readiness['world_size']!=world or readiness['variant']!=args.variant
                or readiness.get('hardware_by_rank')!=actual_hardware
                or readiness.get('readiness_adapter_sha256')!=adapter_sha
                or readiness.get('samples')!=353 or readiness.get('optimizer_updates')!=2
                or not all(readiness.get(k) for k in ('strict_reload','rank_rng_reload','future_inputs_ignored'))):
            raise ValueError('Complete shared readiness approval is required before training')
        continuation = root/'runs'/args.variant/f"continuation-{os.environ.get('SLURM_JOB_ID','local')}.json"
        leader_call(lambda:continuation.unlink(missing_ok=True))
        print(json.dumps({'phase':'launching_frozen_training','rank':dist.get_rank(),
                          'shared_readiness_key':key,'runtime':runtime}),flush=True)
        # W&B 0.30 moved its run-ID helper. Preserve the frozen training source
        # and supply the installed SDK's same helper at its historical location.
        import wandb
        if not callable(getattr(wandb.util,'generate_id',None)):
            from wandb.sdk.lib.runid import generate_id
            wandb.util.generate_id = generate_id
            logging.info('Installed W&B run-ID compatibility adapter for SDK %s',wandb.__version__)
        install_checkpoint_cadence(training, root/'runs'/args.variant)
        from training_metadata import install_metadata_compaction
        install_metadata_compaction(training)
        logging.info('Atomic checkpoint interval is 300 seconds at completed optimizer updates')
        if control:
            install_epoch_limit(training,root/'runs'/args.variant,control,control_sha)
            logging.info('User requested stop after %d completed epoch(s); original LR schedule retained',
                         control['stop_after_completed_epochs'])
        completion_path = root/'runs'/args.variant/'training_complete.json'
        existing = json.loads(completion_path.read_text()) if completion_path.exists() else None
        latest_path=root/'runs'/args.variant/'latest.json'
        latest=json.loads(latest_path.read_text()) if latest_path.exists() else None
        if control and latest and not existing:
            saved_progress=latest['progress']
            if (saved_progress.get('completion_reason')=='user_epoch_limit'
                    and saved_progress.get('epoch_limit_control_sha256')==control_sha):
                # Recover if a process died after the terminal atomic save but
                # before publishing completion; never start another epoch.
                final=json.loads((root/'runs'/args.variant/'final.json').read_text())
                if (final!=latest or saved_progress['epoch']!=control['stop_after_completed_epochs']
                        or saved_progress['cursor']!=0 or saved_progress.get('pending_validation')):
                    raise ValueError('Invalid saved epoch-limit boundary')
                checkpoint=root/'runs'/args.variant/'generations'/latest['generation']
                def recover():
                    state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
                    if state['progress']!=saved_progress or state['fingerprint']!=latest['fingerprint']:
                        raise ValueError('Saved epoch-limit checkpoint differs from its pointer')
                    return {**saved_progress,'fingerprint':state['fingerprint'],'best':state['best'],
                            'checkpoint':str(checkpoint),'manifest':manifest['fingerprint'],
                            'original_scheduler_horizon_epochs':recipe['epochs']}
                existing=leader_call(recover)
                leader_call(lambda:atomic_json(completion_path,existing))
        if (control and existing and existing.get('completion_reason')=='user_epoch_limit'
                and existing.get('epoch_limit_control_sha256')==control_sha):
            result = existing
        else:
            try:
                result = training.train_distributed(root,args.variant,runtime,recipe,device,resume_stopped=args.resume_stopped)
            except EpochLimitReached as stop:
                result = stop.result
                # The frozen trainer's finally block has closed W&B successfully.
                # All ranks reached the same completed, validated atomic save.
                leader_call(lambda:atomic_json(completion_path,result))
        if result.get('interrupted'):
            leader_call(lambda:atomic_json(continuation,{'manifest':manifest['fingerprint'],
                        'world_size':world,'updates':result['updates'],'samples':result['samples'],
                        'checkpoint_saved':True}))
        if dist.get_rank()==0:
            print(json.dumps(result),flush=True)
        dist.destroy_process_group()
        return 75 if result.get('interrupted') else 0
    return run()


if __name__=='__main__':
    try:
        code = main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    sys.exit(code)
