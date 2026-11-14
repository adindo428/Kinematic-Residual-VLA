"""Two-V100 DOMv post-training. Launched by torchrun inside a Slurm GPU job."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from train_dom_posttraining import components,loader,gpu,write
from dom_posttraining import optimizer_groups,lr_factor,accumulation_size
from real_robot.action_objective import sampled_absolute_actions,physical_loss,physical_errors,fixed_rng
from real_robot.training_augmentation import model_inputs
from real_robot.fast_training_augmentation import augment_family_batch
from real_robot.training_ema import TrainingEMA
from utils.real_robot_contract import absolute_pose


class DistributedEpochSampler(torch.utils.data.Sampler):
    """One seeded global weighted draw, sharded by rank, equal full microbatches."""
    def __init__(self,weights,rank,world,microbatch,seed=42):
        if not 0<=rank<world or microbatch<1:raise ValueError('Invalid rank/microbatch')
        self.weights=torch.as_tensor(weights,dtype=torch.double);self.rank=rank;self.world=world;self.seed=seed;self.epoch=0
        self.total=math.ceil(len(weights)/(world*microbatch))*world*microbatch
    def __len__(self):return self.total//self.world
    def __iter__(self):
        indices=torch.multinomial(self.weights,self.total,True,generator=torch.Generator().manual_seed(self.seed+self.epoch))
        return iter((self.epoch,int(i)) for i in indices[self.rank::self.world])


def physical_validation(model,data,indices,recipe,batch_size):
    """Same physical metrics and seeded noise as local training, CUDA FP16."""
    model.eval();device=next(model.parameters()).device;sums={};counts={}
    subset=torch.utils.data.Subset(data,indices)
    with fixed_rng(recipe.get('validation_seed',12345),device),torch.no_grad():
        for batch in torch.utils.data.DataLoader(subset,batch_size=batch_size,shuffle=False,num_workers=0):
            ids=batch['episode_index'].tolist();frames=batch['frame_index'].tolist();batch=gpu(batch)
            valid=~batch['action_is_pad'];target=absolute_pose(batch['action'],batch['observation.state'][:,-1:,:],model.config.euler_delta_convention)
            for seed in recipe['validation_noise_seeds']:
                noise=torch.stack([torch.randn((model.config.chunk_size,model.config.max_action_dim),device=device,
                    generator=torch.Generator(device=device).manual_seed(seed+ep*100000+fr)) for ep,fr in zip(ids,frames)])
                with torch.autocast('cuda',dtype=torch.float16):prediction=sampled_absolute_actions(model,model_inputs(batch,model.config),noise)
                if not torch.isfinite(prediction).all():raise RuntimeError('Nonfinite FP16 validation actions')
                loss,_=physical_loss(prediction,target,valid,recipe)
                sums['selection_score']=sums.get('selection_score',0.)+float(loss)*len(ids);counts['selection_score']=counts.get('selection_score',0)+len(ids)
                errors=physical_errors(prediction.float(),target.float())
                for scope,mask in [('first',valid & (torch.arange(valid.shape[1],device=device)[None,:]==0)),('full',valid)]:
                    for name,value in errors.items():
                        key=scope+'/'+name;sums[key]=sums.get(key,0.)+float(value[mask].sum());counts[key]=counts.get(key,0)+int(mask.sum())
    return {key:sums[key]/counts[key] for key in sums}


def run(c,root,preflight=False,resume=False):
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=2:raise ValueError('This recipe requires exactly two GPUs')
    torch.cuda.set_device(local);dist.init_process_group('nccl',timeout=timedelta(hours=2),device_id=torch.device('cuda',local))
    if 'V100' not in torch.cuda.get_device_name(local):raise ValueError('Requested V100 allocation was not supplied')
    torch.manual_seed(42+rank);random.seed(42+rank);np.random.seed(42+rank)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=True
    root=Path(root).resolve();output=root/'domv'
    if rank==0:output.mkdir(parents=True,exist_ok=preflight or resume)
    dist.barrier()
    affinity=len(os.sched_getaffinity(0))
    if affinity<c['num_workers']*world+2:raise RuntimeError(f'CPU affinity exposes only {affinity} cores; fix Slurm binding before training')
    if rank==0:
        startup=dict(world_size=world,gpu=torch.cuda.get_device_name(local),gpu_memory_gib=torch.cuda.get_device_properties(local).total_memory/2**30,
            cpu_affinity_cores=affinity,precision='fp16_grad_scaler',slurm_job_id=os.environ.get('SLURM_JOB_ID'))
        write(root/'startup.json',startup);print('GPU startup '+json.dumps(startup),flush=True)
    model,cfg,data,validation,weights=components(c,'domv')
    # Released backbone tensors include BF16. V100 AMP needs FP32 master
    # parameters/gradients for GradScaler; widening preserves initial values.
    model.float()
    if any(p.dtype!=torch.float32 for p in model.parameters() if p.requires_grad):raise RuntimeError('FP32 master parameters required')
    if cfg.enable_residual_flow or not cfg.condition_on_object_velocity:raise ValueError('DOMv method mismatch')
    groups=optimizer_groups(model,c['learning_rate'],c['backbone_learning_rate'],c['weight_decay'])
    optimizer=torch.optim.AdamW(groups,betas=(.9,.95),eps=c['optimizer_eps'],fused=True)
    base_lrs=[group['lr'] for group in optimizer.param_groups]
    scaler=torch.amp.GradScaler('cuda',init_scale=128.)
    ema=TrainingEMA(model,c['ema_decay']) if rank==0 else None
    ddp=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True,gradient_as_bucket_view=True)
    sampler=DistributedEpochSampler(weights,rank,world,c['microbatch']);train_loader=loader(data,c,sampler=sampler)
    if c['microbatch']*c['gradient_accumulation']*world!=c['effective_batch_size']:raise ValueError('Global effective batch mismatch')
    updates_per_epoch=math.ceil(len(train_loader)/c['gradient_accumulation']);total_updates=updates_per_epoch*c['max_epochs']
    step=0;start_epoch=1;best=float('inf');stale=0;skipped=0;runlog=None
    if resume:
        state=torch.load(output/'training_state.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['raw_model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        if rank==0:ema.load_state_dict(state['ema'])
        step=state['step'];start_epoch=state['epoch']+1;best=state['best'];stale=state['stale']
        rng=state['rank_rng'][rank];torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'],local);np.random.set_state(rng['numpy']);random.setstate(rng['python']);del state
    def status(state,**extra):
        if rank==0:write(output/'status.json',dict(dict(state=state,step=step,family='domv',world_size=world,precision='fp16_grad_scaler'),**extra))
    try:
        if preflight:
            sample=model_inputs(gpu(next(iter(train_loader))),cfg);model.eval()
            noise=torch.randn(len(sample['action']),20,32,device='cuda')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                explicit=sampled_absolute_actions(model,sample,noise.clone())
                reference=model._get_action_chunk(model._prepare_batch(sample),noise=noise.clone())
                reference=absolute_pose(reference,sample['observation.state'][:,-1:,:],cfg.euler_delta_convention)
            torch.testing.assert_close(explicit,reference,atol=2e-5,rtol=2e-5)
            model.train();optimizer.zero_grad(set_to_none=True)
            for _ in range(3):
                with torch.autocast('cuda',dtype=torch.float16):loss,_=ddp(sample)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite preflight loss')
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale()>=old_scale:
                    step+=1
                    if rank==0:ema.update(model)
            checksum=torch.stack([p.detach().float().sum() for p in model.parameters()]);checks=[torch.zeros_like(checksum) for _ in range(world)];dist.all_gather(checks,checksum)
            if not torch.equal(checks[0],checks[1]) or step==0 or not torch.isfinite(norm):raise RuntimeError('DDP synchronization/optimizer preflight failed')
            if rank==0:write(root/'preflight.json',dict(passed=True,world_size=world,gpu=torch.cuda.get_device_name(local),precision='fp16_grad_scaler',
                cpu_affinity_cores=affinity,parameter_dtype='float32',original_base=c['checkpoint_preferences'][0],native_sampler_max_error=float((explicit-reference).abs().max()),optimizer_updates=step,
                gradient_norm=float(norm),parameters_synchronized=True,weights_saved=False,train_anchors=len(data),global_epoch_draws=sampler.total,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
            dist.barrier();return
        if rank==0:
            import wandb
            runlog=wandb.init(project=c['wandb']['project'],entity=c['wandb']['entity'],name='palmetto-v100x2-fresh-domv',config=dict(c,world_size=world),mode=c['wandb'].get('mode','offline'))
            write(output/'run_config.json',dict(c,world_size=world,train_anchors=len(data),validation_anchors=len(validation),global_epoch_draws=sampler.total,optimizer_updates_per_epoch=updates_per_epoch))
            if not resume:
                status('validating_initial_base',epoch=0)
                initial=physical_validation(model,validation,c['action_validation_indices'],c['action_objective'],c['microbatch'])
                write(output/'baseline_selection.json',initial);runlog.log({'initial/'+k:v for k,v in initial.items()},step=0)
        dist.barrier()
        for epoch in range(start_epoch,c['max_epochs']+1):
            sampler.epoch=epoch;model.train();optimizer.zero_grad(set_to_none=True);started=time.monotonic();loss_sum=0.;samples=0;windows=0
            for index,batch in enumerate(train_loader):
                batch,_=augment_family_batch(gpu(batch),c['augmentation']);batch=model_inputs(batch,cfg)
                boundary=(index+1)%c['gradient_accumulation']==0 or index+1==len(train_loader)
                denominator=accumulation_size(index,c['microbatch'],c['gradient_accumulation'],len(sampler))
                factor=lr_factor(step,total_updates,c['scheduler']['warmup_steps'],c['scheduler']['min_lr_ratio'])
                for group,base_lr in zip(optimizer.param_groups,base_lrs):group['lr']=base_lr*factor
                with nullcontext() if boundary else ddp.no_sync():
                    with torch.autocast('cuda',dtype=torch.float16):loss,_=ddp(batch)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training flow loss')
                    scaler.scale(loss*len(batch['action'])/denominator).backward()
                loss_sum+=float(loss.detach())*len(batch['action']);samples+=len(batch['action'])
                if not boundary:continue
                scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);windows+=1
                if scaler.get_scale()>=old_scale:
                    step+=1
                    if rank==0:ema.update(model)
                else:skipped+=1
                if windows==1 or windows%20==0:
                    metrics=dict(epoch=epoch-1+samples/len(sampler),flow_loss=float(loss.detach()),epoch_flow_loss=loss_sum/samples,
                        gradient_norm=float(norm) if torch.isfinite(norm) else None,global_samples_per_second=world*samples/(time.monotonic()-started),
                        skipped_overflow_updates=skipped,loss_scale=scaler.get_scale(),action_lr=max(g['lr'] for g in optimizer.param_groups))
                    status('training',**metrics)
                    if rank==0:runlog.log({'train/'+k:v for k,v in metrics.items()},step=step);print('domv '+json.dumps(dict(step=step,**metrics)),flush=True)
                stop=torch.tensor(int((root/'STOP').exists()),device='cuda');dist.all_reduce(stop,op=dist.ReduceOp.MAX)
                if stop.item():status('stopped',epoch=epoch,completed_epoch=False,last_completed_epoch=epoch-1);return
            dist.barrier()
            if rank==0:
                status('validating',epoch=epoch);model.eval();flow_sum=0.;flow_count=0
                with torch.random.fork_rng(devices=[local]),torch.no_grad():
                    torch.manual_seed(12345)
                    for batch in loader(validation,c):
                        batch=model_inputs(gpu(batch),cfg)
                        with torch.autocast('cuda',dtype=torch.float16):loss,_=model(batch)
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite validation flow loss')
                        flow_sum+=float(loss)*len(batch['action']);flow_count+=len(batch['action'])
                scores={}
                for kind,context in [('raw',nullcontext()),('ema',ema.apply(model))]:
                    with context:scores[kind]=physical_validation(model,validation,c['action_validation_indices'],c['action_objective'],c['microbatch'])
                kind=min(scores,key=lambda k:scores[k]['selection_score']);score=scores[kind]['selection_score'];flow=flow_sum/flow_count
                if not math.isfinite(score):raise RuntimeError('Nonfinite selected physical score')
                improved=score<best
                if improved:best=score;stale=0
                else:stale+=1
                temporary=output/f'epoch-{epoch:03d}.writing';temporary.mkdir()
                with ema.apply(model) if kind=='ema' else nullcontext():model.save_pretrained(temporary)
                provenance=dict(epoch=epoch,completed_epoch=True,step=step,validation_loss=flow,selection_score=score,selected_weights=kind,
                    checkpoint=c['checkpoint']['domv'],dataset=c['local_verified_dataset'],quality_policy_sha256=c['dataset_quality_policy_sha256'],
                    instruction_mode='recorded',phase_gate_mode=cfg.residual_phase_gate_mode,world_size=world,precision='fp16_grad_scaler',parameter_dtype='float32',autocast_dtype='float16',validation_metrics=scores)
                write(temporary/'provenance.json',provenance);destination=output/f'epoch-{epoch:03d}';temporary.rename(destination)
                pointer=dict(path=str(destination),epoch=epoch,validation_loss=flow,selection_score=score,weights=kind,completed_epoch=True)
                write(output/'latest_checkpoint.json',pointer);write(output/f'epoch_history/{epoch:03d}.json',provenance)
                if improved:write(output/'best_checkpoint.json',pointer)
                runlog.log(dict(completed_epoch=epoch,validation_flow=flow,selection_score=score,best_action_score=best,
                    **{kind+'/'+k:v for kind,m in scores.items() for k,v in m.items()}),step=step)
                print(f'domv completed epoch={epoch} score={score:.6f} best={best:.6f} weights={kind}',flush=True)
            dist.barrier()
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(local),numpy=np.random.get_state(),python=random.getstate())
            rank_rng=[None]*world if rank==0 else None;dist.gather_object(rng,rank_rng,dst=0)
            if rank==0:
                state=dict(epoch=epoch,step=step,best=best,stale=stale,raw_model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),ema=ema.state_dict(),rank_rng=rank_rng)
                torch.save(state,output/'training_state.tmp');(output/'training_state.tmp').replace(output/'training_state.pt');del state
                best_path=Path(json.loads((output/'best_checkpoint.json').read_text())['path'])
                for old in sorted(output.glob('epoch-*/provenance.json'))[:-c['checkpoint_keep_last']]:
                    if old.parent!=best_path:shutil.rmtree(old.parent)
                status('epoch_complete',epoch=epoch,selection_score=score,best_action_score=best,selected_weights=kind)
            stop=torch.tensor(int(rank==0 and epoch>=c['min_epochs'] and stale>=c['early_stopping_patience']),device='cuda');dist.broadcast(stop,src=0)
            if stop.item():break
        status('completed',epoch=epoch,best_action_score=best)
    except BaseException as error:
        status('failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        if runlog is not None:runlog.finish()
        dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--contract',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    run(json.loads(args.contract.read_text()),args.output,args.preflight,args.resume)
