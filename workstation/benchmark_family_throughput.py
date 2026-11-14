"""Disposable throughput measurements at unchanged effective batch/objective."""
import argparse
import gc
import json
from pathlib import Path
import time
import statistics

import numpy as np
import torch
from torch.utils.data import default_collate
from family_data import RealFamilyDataset
from family_policy import load_released_policy
from train_family_finetune import cuda,slice_batch
from real_robot.action_objective import physical_loss,horizon_weights
from real_robot.training_augmentation import augment_training_batch
from real_robot.fast_training_augmentation import augment_family_batch
from real_robot.training_ema import TrainingEMA
from utils.real_robot_contract import absolute_pose


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.campaign.parent;c=json.loads(a.campaign.read_text());recipe=c['training_recipe'];objective=recipe['action_objective']
    stats=json.loads((root/'real_statistics.json').read_text());entry=c['entries'][0]
    data=RealFamilyDataset(c,entry['family']);indices=np.random.default_rng(42).choice(len(data),16,replace=False)
    cpu=default_collate([data[int(i)] for i in indices]);batch=cuda(cpu)
    model=load_released_policy(root,entry,stats);state=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    model.native.load_state_dict(state['model'],strict=True);model.train();parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=3e-6,betas=(.9,.95),weight_decay=.01);optimizer.load_state_dict(state['optimizer'])
    del state;ema=TrainingEMA(model);report={'compute':[],'loader':[]}
    def save():a.output.write_text(json.dumps(report,indent=2)+'\n')
    def update(micro,aux_micro,fast,auxiliary):
        optimizer.zero_grad(set_to_none=True);pool=[]
        for start in range(0,16,micro):
            b,_=(augment_family_batch if fast else augment_training_batch)(slice_batch(batch,start,start+micro),recipe['augmentation'])
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(b,horizon_weights(objective,20,'cuda'))
            (loss*micro/16).backward()
            if start==0:pool=[b]
        if auxiliary:
            b,_=(augment_family_batch if fast else augment_training_batch)(slice_batch(batch,0,4),recipe['augmentation'])
            for start in range(0,4,aux_micro):
                small=slice_batch(b,start,start+aux_micro)
                target=absolute_pose(small['action'],small['reference_state'][:,None,:],'wrapped_xyz_v1')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    prediction=model.sample_absolute(small);aux,_=physical_loss(prediction,target,~small['action_is_pad'],objective)
                (aux*objective['auxiliary_weight']*len(small['action'])/4).backward()
        torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step();ema.update(model)
    for micro,aux_micro,fast in [(2,1,False),(2,1,True),(4,4,True),(8,4,True),(16,4,True)]:
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        try:
            update(micro,aux_micro,fast,True);torch.cuda.synchronize();durations=[]
            for step in range(4):
                started=time.perf_counter();update(micro,aux_micro,fast,step==0);torch.cuda.synchronize();durations.append(time.perf_counter()-started)
            row=dict(flow_microbatch=micro,auxiliary_microbatch=aux_micro,vectorized_photometric=fast,
                effective_batch=16,mean_update_s=statistics.mean(durations),samples_per_second=16/statistics.mean(durations),
                update_seconds=durations,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();row=dict(flow_microbatch=micro,failed='CUDA OOM')
        report['compute'].append(row);save();print(json.dumps(row),flush=True)
    # Parent/model never enters workers. Fresh datasets avoid live decoder handles.
    for workers in [0,2,4]:
        dataset=RealFamilyDataset(c,entry['family']);order=np.random.default_rng(42).choice(len(dataset),256,replace=False).tolist()
        loader=torch.utils.data.DataLoader(dataset,batch_size=16,sampler=order,num_workers=workers,pin_memory=True,
            **(dict(prefetch_factor=2,multiprocessing_context='spawn',persistent_workers=True) if workers else {}))
        times=[];started=time.perf_counter();it=iter(loader)
        for i in range(16):
            b=next(it);now=time.perf_counter();times.append(now-started);started=now
        row=dict(workers=workers,microbatch=16,first_batch_s=times[0],steady_batch_s=statistics.mean(times[4:]),
            steady_samples_per_second=16/statistics.mean(times[4:]))
        report['loader'].append(row);save();print(json.dumps(row),flush=True)
        del it,loader,dataset;gc.collect()


if __name__=='__main__':main()
