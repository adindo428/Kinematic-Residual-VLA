"""Fresh original-DOM real post-training; no robot/device-control imports."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from smoke_real_finetune import build_training_components,LeRobotDataset,PROJECT
sys.path.insert(0,str(PROJECT))
from dom_posttraining import PostTrainingData,EpochWeightedSampler,optimizer_groups,lr_factor,accumulation_size
from real_robot.action_objective import action_validation,sampled_absolute_actions
from real_robot.fast_training_augmentation import augment_family_batch
from real_robot.training_augmentation import model_inputs
from real_robot.training_ema import TrainingEMA
from utils.real_robot_contract import absolute_pose


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def seed_worker(_):torch.set_num_threads(1)


def components(contract,family):
    from prepare_dom_posttraining import sha
    if sha(Path(contract['local_verified_dataset'])/'meta/quality_policy.json')!=contract['dataset_quality_policy_sha256']:raise ValueError('Dataset quality policy changed')
    if sha(contract['sampling_weights'])!=contract['sampling_weights_sha256']:raise ValueError('Sampler weights changed')
    preference=contract['checkpoint_preferences'][0]
    if contract['checkpoint'][family]!=preference['checkpoint_id'] or preference['source']['repo']!='hzxie/dynamic-vla-DOM':raise ValueError('Original DOM initialization required')
    model,cfg,base,item=build_training_components(SimpleNamespace(bundle=PROJECT/'scratch/real_robot/finetuning-checkpoints',
        dataset=Path(contract['local_verified_dataset']),revision=contract['revision'],family=family,checkpoint=preference['path'],
        checkpoint_id=preference['checkpoint_id'],expected_sha256=preference['sha256'],method_overrides=contract['methods'][family]))
    mode=cfg.residual_phase_gate_mode if cfg.enable_residual_flow else 'constant'
    phase_cache=Path(contract['phase_cache']) if contract.get('phase_cache') else None
    train=PostTrainingData(base,horizon=cfg.chunk_size,phase_gate_mode=mode,phase_config=cfg.residual_phase_gate_config,
        cache_bytes=contract['decoder_cache_bytes'],phase_cache=phase_cache)
    validation=LeRobotDataset(contract['repo_id'],root=Path(contract['local_verified_dataset']),split='validation',revision=contract['revision'],
        euler_delta_convention=contract['euler_delta_convention'],required_features=['observation.images.opst_cam','observation.images.wrist_cam',
        'observation.state','observation.environment_state','action'],delta_timestamps={'observation':[-2,0],'action':list(range(cfg.chunk_size))})
    validation=PostTrainingData(validation,horizon=cfg.chunk_size,phase_gate_mode=mode,phase_config=cfg.residual_phase_gate_config,
        cache_bytes=contract['decoder_cache_bytes'],phase_cache=phase_cache)
    if len(contract['action_validation_indices'])!=contract['action_objective']['validation_anchors']:raise ValueError('Incomplete action-validation probe')
    weights=np.load(contract['sampling_weights'])
    if weights.shape!=(len(train),):raise ValueError('Sampler weights/data anchors mismatch')
    model=model.cuda();return model,cfg,train,validation,weights


def loader(data,c,*,sampler=None,workers=None):
    count=c['num_workers'] if workers is None else workers
    options=dict(batch_size=c['microbatch'],num_workers=count,pin_memory=True,sampler=sampler,shuffle=False)
    if count:options.update(prefetch_factor=c['prefetch_factor'],persistent_workers=True,multiprocessing_context='spawn',worker_init_fn=seed_worker)
    return torch.utils.data.DataLoader(data,**options)


def gpu(batch):return {k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}


def preflight(c,family,output):
    torch.manual_seed(42);random.seed(42);np.random.seed(42)
    model,cfg,data,validation,weights=components(c,family)
    optimizer=torch.optim.AdamW(optimizer_groups(model,c['learning_rate'],c['backbone_learning_rate'],c['weight_decay']),betas=(.9,.95),eps=c['optimizer_eps'])
    ema=TrainingEMA(model,c['ema_decay']);sample=next(iter(loader(torch.utils.data.Subset(data,list(range(c['microbatch']))),c,workers=0)))
    batch=model_inputs(gpu(sample),cfg);model.eval()
    noise=torch.randn(len(batch['action']),cfg.chunk_size,cfg.max_action_dim,device='cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        explicit=sampled_absolute_actions(model,batch,noise.clone())
        reference=model._get_action_chunk(model._prepare_batch(batch),noise=noise.clone())
        reference=absolute_pose(reference,batch['observation.state'][:,-1:,:],cfg.euler_delta_convention)
    torch.testing.assert_close(explicit,reference,rtol=2e-5,atol=2e-5)
    model.train();optimizer.zero_grad(set_to_none=True);times=[]
    for i in range(3):
        torch.cuda.synchronize();started=time.monotonic()
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=model(batch)
        if not torch.isfinite(loss):raise ValueError('Nonfinite preflight flow loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();ema.update(model);optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize();times.append(time.monotonic()-started)
    report=dict(passed=True,family=family,original_base=c['checkpoint_preferences'][0],weights_saved=False,hardware_accessed=False,
        native_sampler_max_error=float((explicit-reference).abs().max()),finite_gradients=True,optimizer_and_ema_steps=True,
        phase_gate_mode=cfg.residual_phase_gate_mode,recorded_prompt=sample['task'],
        train_anchors=len(data),validation_anchors=len(validation),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        microbatch=c['microbatch'],gradient_accumulation=c['gradient_accumulation'],effective_batch=c['effective_batch_size'],
        update_seconds=times,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        parameter_groups=[{k:v for k,v in group.items() if k!='params'}|dict(parameters=sum(p.numel() for p in group['params'])) for group in optimizer.param_groups])
    write(output,report);print(json.dumps(report,indent=2),flush=True)


def train(c,family,root,resume=False):
    root=Path(root).resolve();output=root/family;output.mkdir(parents=True,exist_ok=resume)
    torch.manual_seed(42);random.seed(42);np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    model,cfg,data,validation,weights=components(c,family)
    optimizer=torch.optim.AdamW(optimizer_groups(model,c['learning_rate'],c['backbone_learning_rate'],c['weight_decay']),betas=(.9,.95),eps=c['optimizer_eps'],fused=True)
    base_lrs=[group['lr'] for group in optimizer.param_groups];ema=TrainingEMA(model,c['ema_decay'])
    sampler=EpochWeightedSampler(weights);train_loader=loader(data,c,sampler=sampler);val_loader=loader(validation,c)
    updates_per_epoch=math.ceil(len(train_loader)/c['gradient_accumulation']);total_updates=updates_per_epoch*c['max_epochs']
    step=0;start_epoch=1;best=float('inf');stale=0;run=None
    if resume:
        state=torch.load(output/'training_state.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['raw_model']);optimizer.load_state_dict(state['optimizer']);ema.load_state_dict(state['ema'])
        step=state['step'];start_epoch=state['epoch']+1;best=state['best'];stale=state['stale']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        del state
    try:
        import wandb
        run=wandb.init(project=c['wandb']['project'],entity=c['wandb']['entity'],name=c['wandb']['run_names'][family],config=dict(c,family=family),mode=c['wandb'].get('mode','online'))
        write(output/'run_config.json',dict(c,family=family,train_anchors=len(data),validation_anchors=len(validation),optimizer_updates_per_epoch=updates_per_epoch))
        def status(state,**extra):write(output/'status.json',dict(dict(state=state,family=family,step=step),**extra))
        if not resume:
            status('validating_initial_base',epoch=0)
            initial=action_validation(model,validation,c['action_validation_indices'],c['action_objective'],batch_size=c['microbatch'])
            write(output/'baseline_selection.json',initial);run.log({'initial/'+key:value for key,value in initial.items()},step=0)
            print(f'{family} original-base physical score={initial["selection_score"]:.6f}',flush=True)
        for epoch in range(start_epoch,c['max_epochs']+1):
            sampler.epoch=epoch;model.train();optimizer.zero_grad(set_to_none=True);started=time.monotonic();loss_sum=0.;samples=0
            for index,batch in enumerate(train_loader):
                batch=gpu(batch);batch,_=augment_family_batch(batch,c['augmentation']);batch=model_inputs(batch,cfg)
                count=len(batch['action']);denominator=accumulation_size(index,c['microbatch'],c['gradient_accumulation'],len(data))
                factor=lr_factor(step,total_updates,c['scheduler']['warmup_steps'],c['scheduler']['min_lr_ratio'])
                for group,base_lr in zip(optimizer.param_groups,base_lrs):group['lr']=base_lr*factor
                with torch.autocast('cuda',dtype=torch.bfloat16):loss,details=model(batch)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite flow loss')
                (loss*(count/denominator)).backward();loss_sum+=float(loss.detach())*count;samples+=count
                if (index+1)%c['gradient_accumulation'] and index+1<len(train_loader):continue
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True);ema.update(model);step+=1
                if step==1 or step%20==0:
                    metrics=dict(epoch=epoch-1+samples/len(data),step=step,flow_loss=float(loss.detach()),epoch_flow_loss=loss_sum/samples,
                        gradient_norm=float(norm),samples_per_second=samples/(time.monotonic()-started),action_lr=max(g['lr'] for g in optimizer.param_groups))
                    run.log({'train/'+k:v for k,v in metrics.items()},step=step);status('training',**metrics)
                    print(f'{family} '+json.dumps(metrics),flush=True)
                if (root/'STOP').exists():
                    status('stopped',epoch=epoch,completed_epoch=False,last_completed_epoch=epoch-1,
                        reason='Requested stop; resume from last completed epoch. Partial weights excluded from evaluation.')
                    return
            model.eval();flow_sum=0.;flow_count=0
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
                torch.manual_seed(12345)
                for batch in val_loader:
                    batch=model_inputs(gpu(batch),cfg)
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=model(batch)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite validation flow loss')
                    flow_sum+=float(loss)*len(batch['action']);flow_count+=len(batch['action'])
            scores={}
            for kind,context in [('raw',nullcontext()),('ema',ema.apply(model))]:
                with context:scores[kind]=action_validation(model,validation,c['action_validation_indices'],c['action_objective'],batch_size=c['microbatch'])
            kind=min(scores,key=lambda k:scores[k]['selection_score']);score=scores[kind]['selection_score'];flow=flow_sum/flow_count
            if not math.isfinite(score):raise RuntimeError('Nonfinite physical validation score')
            improved=score<best
            if improved:best=score;stale=0
            else:stale+=1
            temporary=output/f'epoch-{epoch:03d}.writing';temporary.mkdir()
            with ema.apply(model) if kind=='ema' else nullcontext():model.save_pretrained(temporary)
            provenance=dict(epoch=epoch,completed_epoch=True,step=step,validation_loss=flow,selection_score=score,selected_weights=kind,
                checkpoint=c['checkpoint'][family],initialization=c['initialization'],dataset=c['local_verified_dataset'],
                quality_policy_sha256=c['dataset_quality_policy_sha256'],instruction_mode='recorded',phase_gate_mode=cfg.residual_phase_gate_mode,
                wandb_run_id=run.id,validation_metrics=scores)
            write(temporary/'provenance.json',provenance);destination=output/f'epoch-{epoch:03d}';temporary.rename(destination)
            pointer=dict(path=str(destination.resolve()),epoch=epoch,validation_loss=flow,selection_score=score,weights=kind,completed_epoch=True)
            write(output/'latest_checkpoint.json',pointer)
            if improved:write(output/'best_checkpoint.json',pointer)
            write(output/f'epoch_history/{epoch:03d}.json',provenance)
            recovery=dict(epoch=epoch,step=step,best=best,stale=stale,raw_model=model.state_dict(),optimizer=optimizer.state_dict(),ema=ema.state_dict(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
            torch.save(recovery,output/'training_state.tmp');(output/'training_state.tmp').replace(output/'training_state.pt');del recovery
            best_path=Path(json.loads((output/'best_checkpoint.json').read_text())['path'])
            for old in sorted(output.glob('epoch-*/provenance.json'))[:-c['checkpoint_keep_last']]:
                if old.parent!=best_path:shutil.rmtree(old.parent)
            status('epoch_complete',epoch=epoch,best_action_score=best,selection_score=score,validation_flow_loss=flow,selected_weights=kind,stale_epochs=stale)
            run.log(dict(validation_flow=flow,selection_score=score,best_action_score=best,completed_epoch=epoch,
                **{kind+'/'+key:value for kind,metrics in scores.items() for key,value in metrics.items()}),step=step)
            print(f'{family} completed epoch={epoch} physical={score:.6f} best={best:.6f} selected={kind} flow={flow:.6f}',flush=True)
            if epoch>=c['min_epochs'] and stale>=c['early_stopping_patience']:break
        status('completed',epoch=epoch,best_action_score=best,reason='validation_patience' if epoch<c['max_epochs'] else 'max_epochs')
    except BaseException as error:
        write(output/'status.json',dict(state='failed',family=family,step=step,error=f'{type(error).__name__}: {error}'));raise
    finally:
        if run is not None:run.finish()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True);p.add_argument('--family',choices=['krv100','domv'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight',action='store_true');p.add_argument('--resume',action='store_true')
    args=p.parse_args();contract=json.loads(args.contract.read_text())
    if args.preflight:preflight(contract,args.family,args.output)
    else:train(contract,args.family,args.output,args.resume)
