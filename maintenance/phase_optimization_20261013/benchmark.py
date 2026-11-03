"""Isolated runtime benchmarks and paired sampling pilots; never writes production."""
import argparse, gc, json, math, os, sys, time
from pathlib import Path
from datetime import timedelta
sys.path.insert(0, os.environ['DVLA_PROJECT_DIR'])
import torch
import psutil
import torch.distributed as dist
from core.dom_distributed import init_control_group, leader_call, synchronize_checkpoint_buffers
from core.dom_distributed_training import distributed_loader, updates, wrap_policy
from core.dom_training import build_dataset, build_policy, make_optimizer, make_scheduler, seed_all, make_loader, evaluate
from utils.dom_experiment import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mb',type=int,required=True);p.add_argument('--workers',type=int,required=True);p.add_argument('--sampling',choices=['frames','blocks'],default='frames');p.add_argument('--steps',type=int,default=30);p.add_argument('--validate',action='store_true');a=p.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.set_num_threads(1)
    dist.init_process_group('nccl',timeout=timedelta(minutes=60));group=init_control_group();seed_all()
    runtime=dict(microbatch=a.mb,workers=a.workers,prefetch=2,pin_memory=True,fused=True,precision='bf16')
    manifest=json.loads((a.root/'manifest.json').read_text());episodes=json.loads((a.root/'pilot_episodes.json').read_text())[:4096]
    data=build_dataset(a.root,episodes,training=True);policy=build_policy(a.root,'krvp100',device);model=wrap_policy(policy,device)
    optimizer=make_optimizer(policy,3e-5,True);scheduler=make_scheduler(optimizer,100*math.ceil(manifest['train_frames']/352),warmup=1000)
    progress=dict(epoch=0,cursor=0,samples=0,updates=0);records=[];torch.cuda.reset_peak_memory_stats(device)
    while progress['updates']<a.steps:
        loader=distributed_loader(data,runtime,progress['epoch'],progress['cursor'],a.sampling)
        iterator=updates(model,optimizer,scheduler,loader,progress,'bf16',device)
        for record in iterator:
            records.append(record)
            proc=psutil.Process(); record['host_rss']=proc.memory_info().rss+sum(c.memory_info().rss for c in proc.children(recursive=True) if c.is_running())
            if progress['updates']>=a.steps:break
        iterator.close();del iterator,loader;gc.collect()
        if progress['updates']<a.steps:progress['epoch']+=1;progress['cursor']=0
    peak=torch.tensor(torch.cuda.max_memory_reserved(device),device=device,dtype=torch.int64);dist.all_reduce(peak,op=dist.ReduceOp.MAX)
    ram=torch.tensor(max(r['host_rss'] for r in records),dtype=torch.float64);dist.all_reduce(ram,op=dist.ReduceOp.SUM,group=group)
    safe=peak.item()<=.85*torch.cuda.get_device_properties(device).total_memory and ram.item()<=.85*int(os.environ['SLURM_MEM_PER_NODE'])*1024**2
    measured=records[5:];result=dict(runtime=runtime,sampling=a.sampling,progress=progress,safe=safe,peak_gpu_bytes=peak.item(),samples_per_second=sum(r['samples'] for r in measured)/sum(r['update_latency_s'] for r in measured),mean_data_wait_s=sum(r['data_wait_s'] for r in measured)/len(measured),manifest=manifest['fingerprint'],job_id=os.environ.get('SLURM_JOB_ID'),pilot_episode_count=len(episodes))
    if a.validate:
        synchronize_checkpoint_buffers(policy)
        def validation():
            val=build_dataset(a.root,json.loads((a.root/'split.json').read_text())['splits']['validation'])
            anchors=json.loads((a.root/'validation_anchors.json').read_text())
            return evaluate(policy,make_loader(val,runtime,indices=anchors),'bf16',device)
        result['validation']=leader_call(validation)
    leader_call(lambda:atomic_json(a.out,result));dist.destroy_process_group()
if __name__=='__main__':main()
