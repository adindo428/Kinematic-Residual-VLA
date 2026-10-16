#!/usr/bin/env python3
"""Run saved-source readiness with only its selected episode metadata in workers.

This adapter leaves the frozen production/measurement source and dataset intact.
It removes unused records from the eight-episode readiness fixture, while using
the frozen optimizer, sampler, tail, inference and checkpoint/RNG checks.
"""
import argparse
from datetime import timedelta
import faulthandler
import json
import os
from pathlib import Path
import sys


def compact_fixture(data):
    episodes = {int(ep) for ep in data.episodes}
    meta = data.base.meta
    meta.episodes = {ep: meta.episodes[ep] for ep in episodes}
    meta.episodes_stats = {ep: meta.episodes_stats[ep] for ep in episodes}
    for attribute in ('parquet_hashes', 'phase_hashes'):
        values = getattr(data, attribute)
        if values is not None:
            setattr(data, attribute, {str(ep): values[str(ep)] for ep in episodes})
    if data.video_hashes is not None:
        names = {str(meta.get_video_file_path(ep, camera))
                 for ep in episodes for camera in data.base.video_keys}
        data.video_hashes = {name: data.video_hashes[name] for name in names}
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--family-root', type=Path, required=True)
    parser.add_argument('--runtime', type=json.loads, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    import torch.distributed as dist
    from torch.distributed.elastic.multiprocessing.errors import record
    import core.dom_distributed_training as worker
    from utils.dom_experiment import atomic_json, digest, file_digest

    @record
    def run():
        faulthandler.enable()
        faulthandler.dump_traceback_later(300, repeat=True)
        local = int(os.environ['LOCAL_RANK'])
        torch.set_num_threads(1)
        torch.cuda.set_device(local)
        dist.init_process_group('nccl', timeout=timedelta(minutes=10))
        original = worker.build_dataset
        def build_fixture(*positional, **keywords):
            data = compact_fixture(original(*positional, **keywords))
            print(json.dumps({'phase':'readiness_fixture_ready', 'rank':dist.get_rank(),
                              'episodes':len(data.episodes), 'metadata_episodes':len(data.base.meta.episodes),
                              'metadata_stats':len(data.base.meta.episodes_stats)}), flush=True)
            return data
        worker.build_dataset = build_fixture
        # Log optimizer boundaries without altering the measured update code.
        original_updates = worker.updates
        def updates(*positional, **keywords):
            for result in original_updates(*positional, **keywords):
                print(json.dumps({'phase':'readiness_update_complete', 'rank':dist.get_rank(),
                                  'samples':result['samples']}), flush=True)
                yield result
            # DDP intentionally retains local BatchNorm running statistics in
            # production. Readiness compares the shared leader checkpoint, so
            # use its canonical buffers before the before/after inference check.
            model = positional[0]
            for buffer in model.module.buffers():
                dist.broadcast(buffer, src=0)
            print(json.dumps({'phase':'readiness_checkpoint_buffers_aligned',
                              'rank':dist.get_rank()}),flush=True)
        worker.updates = updates
        original_save = worker.save_distributed
        original_load = worker.load_generation
        shared_checkpoint = {}
        def save_shared(destination, policy, optimizer, scheduler, progress, fingerprint, wandb_id, **keywords):
            common = [(str(destination), fingerprint) if dist.get_rank()==0 else None]
            dist.broadcast_object_list(common, src=0)
            shared_checkpoint['root'], shared_checkpoint['fingerprint'] = common[0]
            return original_save(Path(common[0][0]), policy, optimizer, scheduler, progress,
                                 common[0][1], wandb_id, **keywords)
        def load_shared(destination, policy, optimizer, scheduler, fingerprint, **keywords):
            return original_load(Path(shared_checkpoint['root']), policy, optimizer, scheduler,
                                 shared_checkpoint['fingerprint'], **keywords)
        worker.save_distributed = save_shared
        worker.load_generation = load_shared
        result = worker.distributed_smoke(args.family_root, 'krv100', args.runtime, torch.device('cuda',local))
        hardware = [None]*dist.get_world_size()
        dist.all_gather_object(hardware, result['hardware'])
        common_key = digest({'hardware':hardware[0], 'world_size':2,
                             'runtime':args.runtime, 'variant':'krv100'})[:16]
        result.update(hardware_by_rank=hardware, shared_checkpoint_root=shared_checkpoint['root'],
                      readiness_adapter_sha256=file_digest(Path(__file__)))
        result.pop('fingerprint')
        result['fingerprint'] = digest(result)
        worker.leader_call(lambda: atomic_json(args.family_root/'distributed-profiles'/f'{common_key}.json',result))
        print(json.dumps({'phase':'readiness_complete', 'rank':dist.get_rank()}), flush=True)
        faulthandler.cancel_dump_traceback_later()
        # On errors let torchrun terminate peers; collective shutdown can stall.
        dist.destroy_process_group()
    run()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        # Do not wait in CUDA/NCCL object destructors after a fatal rank error.
        # @record has already saved its traceback for torchrun's supervisor.
        os._exit(1)
