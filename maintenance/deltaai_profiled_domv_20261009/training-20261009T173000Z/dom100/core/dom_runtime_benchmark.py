"""Measure a selected runtime on the full frozen dataset and four GPUs."""
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from core.dom_distributed_training import distributed_loader, wrap_policy, updates
from core.dom_distributed import leader_call
from core.dom_training import build_dataset, build_policy, load_family, make_optimizer, make_scheduler, make_loader, seed_all
from core.dom_profiling import Telemetry
from utils.dom_experiment import digest, atomic_json


def benchmark(root, variant, runtime, device):
    root=Path(root);manifest=load_family(root);seed_all()
    world=dist.get_world_size();properties=torch.cuda.get_device_properties(device)
    hardware={'name':properties.name,'total_memory':properties.total_memory,
              'torch':torch.__version__,'cuda':torch.version.cuda,'capability':[properties.major,properties.minor]}
    key=digest({'hardware':hardware,'world_size':world,'runtime':runtime,'variant':variant})[:16]
    path=root/'distributed-profiles'/f'{key}.json'
    approved=json.loads(path.read_text())
    assert approved['safe'] and approved['strict_reload'] and approved['rank_rng_reload']
    with Telemetry() as telemetry:
        split=json.loads((root/'split.json').read_text())['splits']
        data=build_dataset(root,split['train'],training=True)
        val_loader=None
        if dist.get_rank()==0:
            val=build_dataset(root,split['validation'])
            val_loader=make_loader(val,runtime,indices=json.loads((root/'validation_anchors.json').read_text()))
            next(iter(val_loader))
        dist.barrier()
        assert len(data)==manifest['train_frames']
        policy=build_policy(root,variant,device);model=wrap_policy(policy,device)
        torch.manual_seed(42+dist.get_rank());torch.cuda.manual_seed(42+dist.get_rank())
        optimizer=make_optimizer(policy,3e-5,runtime['fused'])
        scheduler=make_scheduler(optimizer,10000,warmup=0)
        progress={'epoch':0,'cursor':0,'updates':0,'samples':0}
        loader=distributed_loader(data,runtime,sampling=runtime['sampling'])
        torch.cuda.reset_peak_memory_stats(device)
        started=None
        iterator=updates(model,optimizer,scheduler,loader,progress,runtime['precision'],device)
        for _ in iterator:
            if progress['updates']==2:
                torch.cuda.synchronize(device);started=time.perf_counter()
            if progress['updates']==18:
                torch.cuda.synchronize(device);elapsed=time.perf_counter()-started
                break
        del iterator
        peak=torch.cuda.max_memory_reserved(device)
        ram=max(telemetry.ram,default=0)
        if not ram:raise ValueError('Host memory telemetry unavailable')
        measured=torch.tensor([elapsed,peak,ram],device=device,dtype=torch.float64)
        dist.all_reduce(measured[:2],op=dist.ReduceOp.MAX)
        dist.all_reduce(measured[2:],op=dist.ReduceOp.SUM)
        limit=int(__import__('os').environ['SLURM_MEM_PER_NODE'])*1024**2
        if float(measured[1])>.85*properties.total_memory or float(measured[2])>.85*limit:
            raise ValueError('Four-GPU full-data runtime lacks 15% memory headroom')
        approved.pop('fingerprint')
        approved.update(full_data_benchmark=True,measured_updates=16,warmup_updates=2,
                        samples_per_second=5632/float(measured[0]),
                        max_gpu_reserved_bytes=float(measured[1]),sum_peak_ram_bytes=float(measured[2]),
                        benchmark_samples=progress['samples'],safe=True)
        approved['fingerprint']=digest(approved)
        leader_call(lambda:atomic_json(path,approved))
    del loader,val_loader
    return approved
