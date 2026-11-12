"""Benchmark one released checkpoint before training, without saving changed weights."""
import argparse
import copy
import gc
import json
import math
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch
from torch.utils.data import default_collate

from family_data import RealFamilyDataset
from family_policy import load_released_policy
from train_family_finetune import cuda,slice_batch,write
from real_robot.action_objective import horizon_weights,physical_loss
from real_robot.fast_training_augmentation import augment_family_batch
from real_robot.training_augmentation import augment_training_batch
from real_robot.training_ema import TrainingEMA
from utils.real_robot_contract import absolute_pose


def select_configuration(rows,budget_bytes):
    eligible=[r for r in rows if r.get('passed') and r['peak_reserved_bytes']<=budget_bytes]
    if not eligible:raise ValueError('No finite configuration fits the reserved-memory budget')
    return min(eligible,key=lambda r:r['mean_update_s'])


def clear_caches(model):
    for module in model.modules():
        reset=getattr(module,'reset_cache',None)
        if callable(reset):reset()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--family',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.campaign.parent;campaign=json.loads(args.campaign.read_text());entry=next(x for x in campaign['entries'] if x['family']==args.family)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(family=args.family,checkpoint_id=entry['checkpoint']['id'],checkpoint_sha256=entry['sha256'],repo_revision=campaign['revision'],
        state='loading',passed=False,effective_batch=16,auxiliary_examples=4,auxiliary_every_updates=4,
        weights_saved=False,architecture_changes=False,candidates=[],loader=[],velocity_conditioning=entry.get('velocity_conditioning'))
    def save():write(args.output,report)
    save()
    try:
        torch.manual_seed(42);random.seed(42);np.random.seed(42)
        local=copy.deepcopy(campaign);local.setdefault('family_performance',{})[args.family]={'decoder_cache_episodes':8}
        recipe=local['training_recipe'];objective=recipe['action_objective'];stats=json.loads((root/'real_statistics.json').read_text())
        if objective['auxiliary_batch_size']!=4 or objective['auxiliary_every_steps']!=4:
            raise ValueError('This benchmark requires the frozen four-example/every-four-update recipe')
        report['timing_protocol']='One untimed warmup, then four timed optimizer/EMA updates; reconstruction on the first timed update. Data-loader startup excluded from steady-state rate.'
        verified=json.loads((root/'downloads.json').read_text())['models'][args.family]
        if verified.get('state')!='verified' or verified['sha256']!=entry['sha256']:raise RuntimeError('Unverified checkpoint')
        model=load_released_policy(root,entry,stats);model.train()
        parameters=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=recipe['learning_rate'],betas=(.9,.95),weight_decay=recipe['weight_decay'])
        ema=TrainingEMA(model,campaign['ema']['decay'])
        data=RealFamilyDataset(local,args.family)
        indices=np.random.default_rng(42).choice(len(data),16,replace=False)
        batch=cuda(default_collate([data[int(i)] for i in indices]))
        report['velocity_input_check']=model.check_velocity_tokens(batch)
        if model.method=='pi05':
            small=slice_batch(batch,0,1);noise=torch.randn(1,20,32,device='cuda')
            model.eval()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):reference=model.sample_absolute(small,noise,reference=True)
            model.train()
            with torch.autocast('cuda',dtype=torch.bfloat16):training_sample=model.sample_absolute(small,noise)
            torch.testing.assert_close(training_sample.detach(),reference,atol=2e-5,rtol=2e-5)
            training_sample.sum().backward();optimizer.zero_grad(set_to_none=True)
            report['training_eval_sampler_max_error']=float((training_sample.detach()-reference).abs().max())
            del reference,training_sample
        free,total=torch.cuda.mem_get_info()
        external=max(0,total-free-torch.cuda.memory_reserved())
        margin=max(1536*2**20,int(.06*total));budget=total-external-margin
        report.update(state='benchmarking_compute',architecture_signature=model.architecture_signature,
            parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in parameters),
            total_device_bytes=total,external_device_bytes=external,safety_margin_bytes=margin,reserved_memory_budget_bytes=budget)
        save()

        def update(micro,aux_micro,fast,with_auxiliary):
            optimizer.zero_grad(set_to_none=True);examples=[]
            for start in range(0,16,micro):
                small,_=(augment_family_batch if fast else augment_training_batch)(slice_batch(batch,start,start+micro),recipe['augmentation'])
                with torch.autocast('cuda',dtype=torch.bfloat16):flow=model(small,horizon_weights(objective,20,'cuda'))
                if not torch.isfinite(flow):raise RuntimeError('Nonfinite benchmark flow loss')
                (flow*micro/16).backward()
                if sum(len(b['action']) for b in examples)<4:examples.append(small)
            if with_auxiliary:
                from train_family_finetune import merge_batches
                pool=slice_batch(merge_batches(examples),0,4)
                for start in range(0,4,aux_micro):
                    small=slice_batch(pool,start,start+aux_micro)
                    target=absolute_pose(small['action'],small['reference_state'][:,None,:],'wrapped_xyz_v1')
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        prediction=model.sample_absolute(small);loss,_=physical_loss(prediction,target,~small['action_is_pad'],objective)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite benchmark reconstruction loss')
                    (loss*objective['auxiliary_weight']*len(small['action'])/4).backward()
            torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            optimizer.step();ema.update(model)

        def measure(micro,aux_micro,fast):
            optimizer.zero_grad(set_to_none=True);clear_caches(model);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            row=dict(flow_microbatch=micro,auxiliary_microbatch=aux_micro,vectorized_photometric=fast,passed=False)
            try:
                update(micro,aux_micro,fast,True);torch.cuda.synchronize()
                times=[]
                for i in range(4):
                    start=time.perf_counter();update(micro,aux_micro,fast,i==0);torch.cuda.synchronize();times.append(time.perf_counter()-start)
                model.assert_architecture()
                row.update(passed=True,mean_update_s=statistics.mean(times),samples_per_second=16/statistics.mean(times),
                    update_seconds=times,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            except torch.cuda.OutOfMemoryError:
                row.update(error='CUDA out of memory',oom=True)
                optimizer.zero_grad(set_to_none=True);clear_caches(model);gc.collect();torch.cuda.empty_cache()
            report['candidates'].append(row);save();print(json.dumps(row),flush=True)
            return row

        # Establish a real optimizer/EMA baseline before testing larger activations.
        baseline=measure(1,1,False)
        if not baseline['passed']:raise RuntimeError('Baseline optimizer update cannot fit on this GPU')
        chosen=baseline
        for micro in [2,4,8,16]:
            row=measure(micro,1,True)
            if not row['passed']:break
            if row['peak_reserved_bytes']>budget:break
            chosen=select_configuration(report['candidates'],budget)
        for auxiliary in [2,4]:
            row=measure(chosen['flow_microbatch'],auxiliary,True)
            if not row['passed']:break
            if row['peak_reserved_bytes']>budget:break
        chosen=select_configuration(report['candidates'],budget)
        report.update(state='benchmarking_loader');save()
        micro=chosen['flow_microbatch']
        for workers in [0,2,4]:
            dataset=RealFamilyDataset(local,args.family)
            order=np.random.default_rng(42).choice(len(dataset),256,replace=False).tolist()
            prefetch=max(2,math.ceil(16/(workers*micro))) if workers else 2
            loader=torch.utils.data.DataLoader(dataset,batch_size=micro,sampler=order,num_workers=workers,pin_memory=True,
                **(dict(prefetch_factor=prefetch,persistent_workers=True,multiprocessing_context='spawn') if workers else {}))
            iterator=iter(loader);times=[];start=time.perf_counter()
            for _ in range(math.ceil(256/micro)):
                sample=next(iterator);now=time.perf_counter();times.append(now-start);start=now
            warm=max(2,math.ceil(64/micro));steady=statistics.mean(times[warm:])
            row=dict(workers=workers,prefetch_factor=prefetch,flow_microbatch=micro,first_batch_s=times[0],steady_batch_s=steady,
                samples_per_second=micro/steady,decoder_cache_episodes=8)
            report['loader'].append(row);save();print(json.dumps(row),flush=True)
            del iterator,loader,dataset;gc.collect()
        loader_choice=max(report['loader'],key=lambda r:r['samples_per_second'])
        report.update(state='completed',passed=True,selected=dict(flow_microbatch=chosen['flow_microbatch'],
            auxiliary_microbatch=chosen['auxiliary_microbatch'],vectorized_photometric=chosen['vectorized_photometric'],
            loader_workers=loader_choice['workers'],prefetch_factor=loader_choice['prefetch_factor'],decoder_cache_episodes=8),
            selected_compute=chosen,selected_loader=loader_choice)
        save()
    except BaseException as exc:
        report.update(state='failed',error=f'{type(exc).__name__}: {exc}');save();raise


if __name__=='__main__':main()
