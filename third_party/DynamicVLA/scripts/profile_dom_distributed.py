#!/usr/bin/env python3
"""Isolated torchrun worker for the SSD KRVLA+Velocity throughput search."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from core.dom_distributed_profiling import profile_trial


@record
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family-root', type=Path, required=True)
    parser.add_argument('--runtime', type=json.loads, required=True)
    parser.add_argument('--mode', choices=('gpu_only','end_to_end'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--updates', type=int, default=5)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--epoch', type=int, default=0,
                        help='Deterministic profiling sample order; trial updates are discarded')
    args = parser.parse_args()
    if int(os.environ['WORLD_SIZE']) != 2:
        raise ValueError('The workstation profiler requires exactly two GPUs')
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    torch.set_num_threads(1)
    dist.init_process_group('nccl',timeout=timedelta(minutes=10))
    # A failed rank must exit promptly so torchrun can stop its peers. Explicit
    # collective cleanup on an error can wait for peers still doing backward.
    profile_trial(args.family_root,args.runtime,args.mode,torch.device('cuda',local),
                  args.output,args.warmup,args.updates,args.trace,args.epoch)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
