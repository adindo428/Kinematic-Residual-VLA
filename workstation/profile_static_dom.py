"""Confirm optimized DOM-family throughput with disposable optimizer updates."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import torch
from smoke_real_finetune import build_training_components,PROJECT
sys.path.insert(0,str(PROJECT))
from utils.dom_data import DOMData
from utils.real_robot_contract import absolute_pose
from real_robot.action_objective import sampled_absolute_actions,physical_loss,horizon_weights
from real_robot.fast_training_augmentation import augment_family_batch
from real_robot.training_augmentation import model_inputs
from train_real_finetune import worker_init


def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--family',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    c=json.loads(a.contract.read_text());pref=next(x for x in c['checkpoint_preferences'] if x['checkpoint_id']==c['checkpoint'][a.family])
    model,cfg,data,_=build_training_components(SimpleNamespace(bundle=PROJECT/'scratch/real_robot/finetuning-checkpoints',
        dataset=Path(c['local_verified_dataset']),revision=c['revision'],family=a.family,
        method_overrides=c['methods'][a.family],checkpoint=Path(pref['path']),checkpoint_id=pref['checkpoint_id'],expected_sha256=pref['sha256']))
    signature=hashlib.sha256(json.dumps([(n,list(p.shape)) for n,p in model.named_parameters()]).encode()).hexdigest()
    model.cuda();data=DOMData(data,horizon=20,cache_bytes=c['decoder_cache_bytes'])
    order=torch.randperm(len(data),generator=torch.Generator().manual_seed(42))[:64].tolist()
    loader=torch.utils.data.DataLoader(data,batch_size=16,sampler=order,num_workers=c['num_workers'],pin_memory=True,
        prefetch_factor=c['prefetch_factor'],multiprocessing_context='spawn',persistent_workers=True,worker_init_fn=worker_init)
    started=time.perf_counter();iterator=iter(loader);cpu=next(iterator);startup=time.perf_counter()-started
    batch={k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in cpu.items()}
    probe={k:v[:2] if torch.is_tensor(v) or isinstance(v,list) else v for k,v in batch.items()}
    probe=model_inputs(probe,cfg);noise=torch.randn(2,20,cfg.max_action_dim,device='cuda');model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        model.reset();regular=model.predict_action_chunk(dict(probe),noise=noise)
        regular=absolute_pose(regular,probe['observation.state'][:,-1:,:],cfg.euler_delta_convention)
        explicit=sampled_absolute_actions(model,dict(probe),noise=noise)
    torch.testing.assert_close(regular.float(),explicit.float(),atol=2e-6,rtol=2e-6)
    model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],betas=(.9,.95),weight_decay=c['weight_decay'])
    objective=c['action_objective']
    def update(auxiliary):
        b,_=augment_family_batch(dict(batch),c['augmentation']);b=model_inputs(b,cfg)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):flow,_=model(b,horizon_weights=horizon_weights(objective,20,'cuda'))
        if not torch.isfinite(flow):raise RuntimeError('Nonfinite flow loss')
        flow.backward()
        if auxiliary:
            small={k:v[:objective['auxiliary_batch_size']] if torch.is_tensor(v) or isinstance(v,list) else v for k,v in b.items()}
            target=absolute_pose(small['action'],small['observation.state'][:,-1:,:],cfg.euler_delta_convention)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=sampled_absolute_actions(model,small)
                loss,_=physical_loss(prediction,target,~small['action_is_pad'],objective)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite physical loss')
            (loss*objective['auxiliary_weight']).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    torch.cuda.reset_peak_memory_stats();update(True);torch.cuda.synchronize();times=[]
    for i in range(4):
        started=time.perf_counter();update(i==0);torch.cuda.synchronize();times.append(time.perf_counter()-started)
    loader_times=[]
    for _ in range(3):
        started=time.perf_counter();next(iterator);loader_times.append(time.perf_counter()-started)
    assert signature==hashlib.sha256(json.dumps([(n,list(p.shape)) for n,p in model.named_parameters()]).encode()).hexdigest()
    report=dict(family=a.family,passed=True,weights_saved=False,architecture_changes=False,architecture_signature=signature,
        checkpoint_sha256=pref['sha256']['model.safetensors'],gate=cfg.residual_phase_gate_mode,
        effective_batch=16,microbatch=16,workers=c['num_workers'],prefetch=c['prefetch_factor'],
        batched_decode=True,vectorized_photometric=True,finite_gradients=True,optimizer_step=True,
        sampler_parity_max_error=float((regular-explicit).abs().max()),
        samples_per_second=16/statistics.mean(times),update_seconds=times,
        loader_startup_s=startup,loader_prefetched_batch_seconds=loader_times,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
