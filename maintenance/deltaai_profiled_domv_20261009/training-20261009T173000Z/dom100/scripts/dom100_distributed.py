#!/usr/bin/env python3
"""Run an approved DOM variant on four/eight GPUs via torchrun."""
import argparse
from datetime import timedelta
import json
import logging
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from core.dom_distributed_training import distributed_smoke, train_distributed
from core.dom_distributed import leader_call
from core.dom_training import load_family
from utils.dom_experiment import VARIANTS, digest, atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family-root',type=Path,required=True)
    parser.add_argument('--variant',choices=VARIANTS,default='krv100')
    parser.add_argument('--command',choices=('smoke','profile','train'),default='train')
    parser.add_argument('--microbatch',type=int,default=44)
    parser.add_argument('--workers',type=int,choices=(0,1,2,4,8),default=1)
    parser.add_argument('--runtime-file',type=Path)
    parser.add_argument('--resume-stopped',action='store_true',help='Explicitly continue an early-stopped run with its optimizer and RNG state')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world not in (4,8) or int(os.environ.get('LOCAL_WORLD_SIZE',world))!=world:
        raise ValueError('This allocation must contain four or eight GPUs on one node')
    if not 1<=args.microbatch<=352//world:
        raise ValueError('Microbatch must not exceed the local portion of batch 352')
    torch.cuda.set_device(local)
    torch.set_num_threads(1)
    device=torch.device('cuda',local)
    properties=torch.cuda.get_device_properties(device)
    if not any(name in properties.name.upper() for name in ('A100','H100','H200')):
        raise ValueError('Distributed training requires a tested A100/H100/H200 class')
    dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    try:
        root=args.family_root.resolve();manifest=load_family(root)
        continuation=root/'runs'/args.variant/f"continuation-{os.environ.get('SLURM_JOB_ID','local')}.json"
        if args.command != 'smoke':
            leader_call(lambda:continuation.unlink(missing_ok=True))
        recipe=json.loads((root/'recipe.json').read_text())
        check=dict(recipe);fingerprint=check.pop('fingerprint')
        if digest(check)!=fingerprint or recipe['manifest']!=manifest['fingerprint']:
            raise ValueError('Distributed training recipe differs from frozen provenance')
        runtime={'microbatch':args.microbatch,'workers':args.workers,'prefetch':2,
                 'pin_memory':True,'fused':True,'precision':recipe['precision']}
        if args.runtime_file:
            selected=json.loads(args.runtime_file.read_text())
            selected_check=dict(selected); selected_fingerprint=selected_check.pop('fingerprint')
            if digest(selected_check)!=selected_fingerprint:
                raise ValueError('Selected runtime provenance was modified')
            runtime=selected['runtime']
            if (selected['manifest']!=manifest['fingerprint'] or runtime['precision']!=recipe['precision']
                    or runtime.get('sampling','frames')!=recipe['sampling']
                    or not 1<=runtime['microbatch']<=352//world
                    or runtime['workers'] not in (0,1,2,4,8) or runtime['prefetch'] not in (2,4)
                    or type(runtime['pin_memory']) is not bool or type(runtime['fused']) is not bool):
                raise ValueError('Selected runtime differs from the frozen study')
            if args.command == 'train' and recipe.get('runtime_fingerprint')!=digest(runtime):
                raise ValueError('Production runtime differs from the approved recipe')
        elif args.command == 'train':
            raise ValueError('Production requires the selected profiling runtime file')
        hardware={'name':properties.name,'total_memory':properties.total_memory,
                  'torch':torch.__version__,'cuda':torch.version.cuda,'capability':[properties.major,properties.minor]}
        key=digest({'hardware':hardware,'world_size':world,'runtime':runtime,'variant':args.variant})[:16]
        profile_path=root/'distributed-profiles'/f'{key}.json'
        verified=False
        if profile_path.exists():
            profile=json.loads(profile_path.read_text());copy=dict(profile);expected=copy.pop('fingerprint')
            verified=(digest(copy)==expected and profile['manifest']==manifest['fingerprint'] and profile['complete'] and profile['safe']
                      and (args.command != 'train' or (profile.get('full_data_benchmark',False)
                           and selected.get('distributed_profile_fingerprint')==expected)))
        if args.command in ('smoke','profile') and not verified:
            distributed_smoke(root,args.variant,runtime,device)
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            verified=True
        if args.command == 'profile':
            from core.dom_runtime_benchmark import benchmark
            benchmark(root,args.variant,runtime,device)
        if args.command == 'train':
            if not verified:raise ValueError('Actual distributed backward/checkpoint/RNG checks must pass first')
            result=train_distributed(root,args.variant,runtime,recipe,device,resume_stopped=args.resume_stopped)
            if result.get('interrupted'):
                leader_call(lambda:atomic_json(continuation,{'manifest':manifest['fingerprint'],
                    'world_size':world,'updates':result['updates'],'samples':result['samples'],'checkpoint_saved':True}))
            if dist.get_rank()==0:print(json.dumps(result),flush=True)
            return 75 if result.get('interrupted') else 0
        return 0
    finally:
        dist.destroy_process_group()


if __name__=='__main__':sys.exit(main())
