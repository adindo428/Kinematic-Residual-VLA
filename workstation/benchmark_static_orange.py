"""Benchmark disposable native KRV updates on the cached orange subset."""
import argparse,gc,hashlib,json,statistics,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from smoke_real_finetune import build_training_components,PROJECT
from cached_dom_data import CachedDOMData
from train_real_finetune import worker_init
from real_robot.action_objective import sampled_absolute_actions,physical_loss,horizon_weights
from real_robot.training_augmentation import model_inputs
from utils.dom_data import DOMData
from utils.real_robot_contract import absolute_pose


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();c=json.loads((root/'launch_contract.json').read_text());pref=c['checkpoint_preferences'][0]
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=True
    model,cfg,base,_=build_training_components(SimpleNamespace(bundle=PROJECT/'scratch/real_robot/finetuning-checkpoints',dataset=Path(c['local_verified_dataset']),revision=c['revision'],family='krv100',method_overrides=c['methods']['krv100'],checkpoint=Path(pref['path']),checkpoint_id=pref['checkpoint_id'],expected_sha256=pref['sha256']))
    # Compare independently decoded video and cached adapter samples before timing.
    standard=DOMData(base,horizon=20,cache_bytes=64*1024**2);item=standard[0]
    from smoke_real_finetune import LeRobotDataset
    second=LeRobotDataset(c['repo_id'],root=Path(c['local_verified_dataset']),split='train',revision=c['revision'],euler_delta_convention=c['euler_delta_convention'],required_features=['observation.images.opst_cam','observation.images.wrist_cam','observation.state','observation.environment_state','action'],delta_timestamps={'observation':[-2,0],'action':list(range(20))})
    cached=CachedDOMData(second,c['decoded_video_cache'],horizon=20,cache_bytes=64*1024**2);other=cached[0]
    for index in [0,len(cached)//2,len(cached)-1]:
        item=standard[index];other=cached[index]
        for key in item:
            if torch.is_tensor(item[key]):torch.testing.assert_close(item[key],other[key],atol=0,rtol=0)
            else:assert item[key]==other[key],key
    del standard,item,other;model.cuda().train();objective=c['action_objective'];results=[]
    # Train only disposable model copies. Timing includes physical auxiliary updates
    # at exactly the same frequency as the actual run.
    for size in [16,20,24]:
        loader=None;iterator=None;optimizer=None;batch=None
        try:
            order=torch.randperm(len(cached),generator=torch.Generator().manual_seed(42))[:size*6].tolist()
            loader=torch.utils.data.DataLoader(cached,batch_size=size,sampler=order,num_workers=c['num_workers'],pin_memory=True,prefetch_factor=c['prefetch_factor'],multiprocessing_context='spawn',persistent_workers=True,worker_init_fn=worker_init)
            started=time.perf_counter();iterator=iter(loader);cpu=next(iterator);startup=time.perf_counter()-started
            batch={k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in cpu.items()};batch=model_inputs(batch,cfg)
            optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],betas=(.9,.95),weight_decay=c['weight_decay'],fused=True)
            def update(auxiliary):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):flow,_=model(batch,horizon_weights=horizon_weights(objective,20,'cuda'))
                if not torch.isfinite(flow):raise RuntimeError('Nonfinite benchmark flow loss')
                flow.backward()
                if auxiliary:
                    count=objective['auxiliary_batch_size'];small={k:v[:count] if torch.is_tensor(v) or isinstance(v,list) else v for k,v in batch.items()}
                    target=absolute_pose(small['action'],small['observation.state'][:,-1:,:],cfg.euler_delta_convention)
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=physical_loss(sampled_absolute_actions(model,small),target,~small['action_is_pad'],objective)
                    (loss*objective['auxiliary_weight']).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            torch.cuda.reset_peak_memory_stats();update(True);torch.cuda.synchronize();times=[]
            for step in range(8):
                started=time.perf_counter();update(step%objective['auxiliary_every_steps']==0);torch.cuda.synchronize();times.append(time.perf_counter()-started)
            waits=[]
            for _ in range(4):
                started=time.perf_counter();next(iterator);waits.append(time.perf_counter()-started)
            row=dict(batch_size=size,passed=True,samples_per_second=size/statistics.mean(times),update_seconds=times,loader_startup_s=startup,loader_wait_seconds=waits,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            results.append(row);print(json.dumps(row),flush=True)
        except torch.OutOfMemoryError:
            results.append(dict(batch_size=size,passed=False,reason='cuda_out_of_memory'));print('OOM at batch',size,flush=True)
        finally:
            if iterator is not None:
                for _ in iterator:pass
                iterator._shutdown_workers()
            model.zero_grad(set_to_none=True);del optimizer,iterator,loader,batch;gc.collect();torch.cuda.empty_cache()
        if not results[-1]['passed']:break
    valid=[r for r in results if r['passed']]
    if not valid:raise RuntimeError('No working benchmark configuration')
    best=max(valid,key=lambda r:r['samples_per_second']);c['microbatch']=best['batch_size'];c['effective_batch_size']=best['batch_size']
    (root/'launch_contract.json').write_text(json.dumps(c,indent=2)+'\n')
    report=dict(passed=True,cache_adapter_exact_parity=True,weights_saved=False,selected=best,benchmarks=results,precision='bf16',optimizer='fused AdamW',architecture_changes=False)
    (root/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n');print('Selected batch',best['batch_size'],flush=True)


if __name__=='__main__':main()
