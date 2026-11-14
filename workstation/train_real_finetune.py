"""Sequential workstation fine-tuning; local checkpoints and online W&B metrics."""
import argparse
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
import wandb
from safetensors.torch import load_model
from smoke_real_finetune import build_training_components, LeRobotDataset,PROJECT
sys.path.insert(0,str(PROJECT))
from real_robot.training_augmentation import augment_training_batch,model_inputs
from real_robot.fast_training_augmentation import augment_family_batch
from real_robot.action_objective import (horizon_weights, physical_loss, sampled_absolute_actions,
    action_validation)


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def completed_progress(output):
    """Reconstruct patience from completed epochs, including the last epoch."""
    best = float('inf'); stale = 0; latest = None
    baseline=output/'baseline_selection.json'
    if baseline.exists():best=json.loads(baseline.read_text())['selection_score']
    records={json.loads(path.read_text())['epoch']:(path.parent,json.loads(path.read_text()))
        for path in output.glob('epoch-*/provenance.json')}
    for path in (output/'epoch_history').glob('*.json'):
        record=json.loads(path.read_text());records.setdefault(record['epoch'],(None,record))
    for epoch in sorted(records):
        directory,record=records[epoch]
        if not record['completed_epoch']:
            continue
        value = record.get('selection_score', record['validation_loss'])
        if not math.isfinite(value):
            raise ValueError('Non-finite completed validation loss')
        if value < best:
            best = value; stale = 0
        else:
            stale += 1
        if directory is not None:latest = (directory, record)
    return latest, best, stale


def worker_init(_):torch.set_num_threads(1)


def train(contract, family, output, resume=False):
    output.mkdir(parents=True, exist_ok=resume)
    latest, best, stale = completed_progress(output) if resume else (None, float('inf'), 0)
    if resume and latest is None:
        raise RuntimeError('Continuation requires a completed-epoch checkpoint')
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    dataset = Path(contract['local_verified_dataset'])
    assert hashlib.sha256((dataset/'meta/quality_policy.json').read_bytes()).hexdigest() == contract['dataset_quality_policy_sha256']
    preference = next(p for p in contract['checkpoint_preferences'] if p['checkpoint_id'] == contract['checkpoint'][family])
    args = SimpleNamespace(bundle=Path('scratch/real_robot/finetuning-checkpoints'), dataset=dataset,
                           revision=contract['revision'], alias=preference.get('alias'),family=family,
                           method_overrides=contract.get('methods',{}).get(family),
                           checkpoint=preference.get('path'),checkpoint_id=preference['checkpoint_id'],expected_sha256=preference.get('sha256'))
    model, cfg, data, item = build_training_components(args)
    validation = LeRobotDataset(contract['repo_id'], root=dataset, split=contract.get('evaluation_split','validation'), revision=contract['revision'],
        euler_delta_convention=contract['euler_delta_convention'], required_features=[
            'observation.images.opst_cam','observation.images.wrist_cam','observation.state',
            'observation.environment_state','action'],
        delta_timestamps={'observation':contract['observation_history'],'action':list(range(cfg.chunk_size))})
    if contract.get('decoded_video_cache'):
        from cached_dom_data import CachedDOMData
        data=CachedDOMData(data,contract['decoded_video_cache'],instruction_mode=contract.get('instruction_mode','generated'),horizon=cfg.chunk_size,cache_bytes=contract.get('decoder_cache_bytes',64*1024**2))
        validation=CachedDOMData(validation,contract['decoded_video_cache'],instruction_mode=contract.get('instruction_mode','generated'),horizon=cfg.chunk_size,cache_bytes=contract.get('decoder_cache_bytes',64*1024**2))
    elif contract.get('batched_decode'):
        from utils.dom_data import DOMData
        data=DOMData(data,horizon=cfg.chunk_size,cache_bytes=contract.get('decoder_cache_bytes',512*1024**2))
        validation=DOMData(validation,horizon=cfg.chunk_size,cache_bytes=contract.get('decoder_cache_bytes',512*1024**2))
    loader_args = dict(batch_size=contract['microbatch'], num_workers=contract['num_workers'],pin_memory=True)
    if contract['num_workers']:
        loader_args.update(prefetch_factor=contract['prefetch_factor'],multiprocessing_context='spawn',
                           persistent_workers=contract.get('persistent_workers',False),worker_init_fn=worker_init)
    sampler=None
    if contract.get('sampling_weights'):
        weight_path=Path(contract['sampling_weights'])
        assert hashlib.sha256(weight_path.read_bytes()).hexdigest()==contract['sampling_weights_sha256']
        weights=np.load(weight_path)
        if weights.shape!=(len(data),) or not np.isfinite(weights).all() or (weights<=0).any():raise ValueError('Invalid phase sampling weights')
        sampler=torch.utils.data.WeightedRandomSampler(weights,len(data),replacement=True,
            generator=torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(data, shuffle=sampler is None,sampler=sampler,
        generator=torch.Generator().manual_seed(42), **loader_args)
    val_loader = torch.utils.data.DataLoader(validation, shuffle=False, **loader_args)
    if contract.get('fused_adamw'):
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.benchmark=True
    model.cuda().train()
    objective=contract.get('action_objective')
    probe_indices=None
    if objective:
        probe_indices=contract.get('action_validation_indices')
        if probe_indices is None:
            probe_indices=sorted(np.random.default_rng(objective['validation_seed']).choice(len(validation),
                min(objective['validation_anchors'],len(validation)),replace=False).tolist())
        if len(set(probe_indices))!=len(probe_indices) or min(probe_indices)<0 or max(probe_indices)>=len(validation):
            raise ValueError('Invalid action validation indices')
        write_json(output/'action_validation_indices.json',dict(indices=probe_indices,
            dataset_quality_policy_sha256=contract['dataset_quality_policy_sha256'],noise_seeds=objective['validation_noise_seeds']))
    weight_decay=contract.get('weight_decay',.01);augmentation=contract.get('augmentation',{})
    schedule=contract.get('scheduler',{'type':'constant'});warmup=int(schedule.get('warmup_steps',0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=contract['learning_rate'], betas=(.9,.95), weight_decay=weight_decay,**({'fused':True} if contract.get('fused_adamw') else {}))
    scheduler=None
    if schedule['type']=='warmup_plateau':
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=schedule.get('factor',.5),
            patience=schedule.get('patience',2),min_lr=schedule.get('min_lr',3e-7))
    elif schedule['type']!='constant':raise ValueError('Unsupported fine-tuning scheduler')
    step = 0; start_epoch = 1; resume_id = None
    if resume:
        directory, record = latest
        load_model(model, str(directory/'model.safetensors'), strict=True)
        state = torch.load(directory/'training_state.pt', map_location='cpu', weights_only=False)
        assert state['completed_epoch'] and state['epoch'] == record['epoch']
        optimizer.load_state_dict(state['optimizer'])
        if scheduler is not None:scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
        train_loader.generator.set_state(state['loader_rng'])
        if sampler is not None:sampler.generator.set_state(state['sampler_rng'])
        step = state['step']; start_epoch = state['epoch']+1
        resume_id = record['wandb_run_id']
    run_config = dict(contract, family=family, initialization_checkpoint=item['id'],
                      train_anchors=len(data), validation_anchors=len(validation), seed=42,
                      optimizer='AdamW', weight_decay=weight_decay, betas=[.9,.95], clip_grad_norm=1.,
                      scheduler=schedule, validation_noise_seed=12345,
                      condition_on_object_velocity=cfg.condition_on_object_velocity,enable_residual_flow=cfg.enable_residual_flow)
    write_json(output/'run_config.json', run_config)
    run = wandb.init(entity=contract['wandb']['entity'], project=contract['wandb']['project'],
        name=contract['wandb']['run_names'][family], mode='online', dir=str(output), config=run_config,
        id=resume_id, resume='must' if resume else None, allow_val_change=True,
        tags=['real-robot','finetune',family,'FastViT',
            'released-pretrained-initialization' if preference.get('source') else 'completed-epoch-initialization'])
    def status(**fields):
        write_json(output/'status.json', dict(family=family, step=step, wandb_url=run.url, **fields))
    def checkpoint(name, epoch, complete, val=None, selection_score=None):
        temporary = output/(name+'.writing')
        temporary.mkdir()
        model.save_pretrained(temporary)
        torch.save(dict(optimizer=optimizer.state_dict(), epoch=epoch, completed_epoch=complete, step=step,
            scheduler=None if scheduler is None else scheduler.state_dict(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            loader_rng=train_loader.generator.get_state(),sampler_rng=None if sampler is None else sampler.generator.get_state()), temporary/'training_state.pt')
        write_json(temporary/'provenance.json', dict(checkpoint=item['id'], epoch=epoch, completed_epoch=complete,
            step=step, validation_loss=val, dataset=str(dataset), quality_policy_sha256=contract['dataset_quality_policy_sha256'],
            instruction_mode=contract.get('instruction_mode','generated'),
            **({'selection_score':selection_score} if selection_score is not None else {}),
            wandb_run_id=run.id, resume_note='Partial checkpoints contain recovery weights; exact mid-epoch data replay is not implemented.'))
        destination = output/name
        temporary.rename(destination)
        write_json(output/'latest_checkpoint.json', {'path':str(destination.resolve()),'completed_epoch':complete,'step':step})
        return destination
    try:
        if objective and not resume:
            initial=action_validation(model,validation,probe_indices,objective)
            best=initial['selection_score']
            write_json(output/'baseline_selection.json',initial)
            write_json(output/'best_checkpoint.json',dict(path=str(Path(preference['path']).resolve()),epoch=0,
                selection_score=best,selected_by='sampled_action_score',initialization=True))
            run.log({'validation/action_'+k:v for k,v in initial.items()},step=0)
            print(f'{family} initial action score={best:.6f}',flush=True)
        status(state='training', epoch=start_epoch-1, train_anchors=len(data), validation_anchors=len(validation))
        previous_recovery = None
        for epoch in range(start_epoch, contract['max_epochs']+1):
            model.train(); epoch_sum=0.; epoch_count=0; started=time.monotonic()
            for batch in train_loader:
                batch={k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
                batch,aug_metrics=(augment_family_batch if contract.get('vectorized_photometric') else augment_training_batch)(batch,augmentation,training=True)
                batch=model_inputs(batch,cfg)
                if step<warmup:
                    for group in optimizer.param_groups:group['lr']=contract['learning_rate']*(step+1)/warmup
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss, details=model(batch,**({'horizon_weights':horizon_weights(objective,cfg.chunk_size,'cuda')} if objective else {}))
                if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
                loss.backward()
                aux_metrics={}
                if objective and step%objective['auxiliary_every_steps']==0:
                    count=min(objective['auxiliary_batch_size'],len(batch['observation.state']))
                    # Rotate the auxiliary slice; deterministic first-two-only sampling would bias it.
                    offset=(step//objective['auxiliary_every_steps']*count)%len(batch['observation.state'])
                    indices=(torch.arange(count,device='cuda')+offset)%len(batch['observation.state'])
                    small={k:v[indices] if torch.is_tensor(v) and v.ndim and len(v)==len(batch['observation.state']) else
                        [v[int(i)] for i in indices] if isinstance(v,list) and len(v)==len(batch['observation.state']) else v
                        for k,v in batch.items()}
                    from utils.real_robot_contract import absolute_pose
                    target=absolute_pose(small['action'],small['observation.state'][:,-1:,:],cfg.euler_delta_convention)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        pred=sampled_absolute_actions(model,small)
                        aux,aux_metrics=physical_loss(pred,target,~small['action_is_pad'],objective)
                        auxiliary_loss=objective['auxiliary_weight']*aux
                    if not torch.isfinite(auxiliary_loss):raise RuntimeError('Nonfinite reconstruction loss')
                    auxiliary_loss.backward()
                    aux_metrics['weighted_loss']=float(auxiliary_loss.detach())
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step(); step+=1
                size=batch['observation.state'].shape[0]
                value=float(loss.detach()); epoch_sum+=value*size; epoch_count+=size
                if step==1 or step%20==0:
                    metrics={'train/loss':value,'train/grad_norm':float(norm),'train/lr':optimizer.param_groups[0]['lr'],
                        'epoch':epoch-1+epoch_count/len(data),'gpu/allocated_gib':torch.cuda.memory_allocated()/2**30,
                        'train/samples_per_second':epoch_count/(time.monotonic()-started)}
                    metrics.update({'augmentation/'+k:v for k,v in aug_metrics.items()})
                    metrics.update({'reconstruction/'+k:v for k,v in aux_metrics.items()})
                    run.log(metrics,step=step)
                    status(state='training',epoch=epoch,loss=value)
                    print(f'{family} epoch={epoch} step={step} loss={value:.6f}',flush=True)
                if step==contract.get('first_recovery_checkpoint_step',100) or step%contract.get('checkpoint_every_steps',500)==0:
                    saved=checkpoint(f'recovery-step-{step:07d}',epoch,False)
                    if previous_recovery is not None: shutil.rmtree(previous_recovery)
                    previous_recovery=saved
            model.eval(); val_sum=0.; val_count=0
            # Fixed noise makes successive validation losses comparable without altering training RNG.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(12345)
                with torch.no_grad():
                    for validation_batch_index,batch in enumerate(val_loader):
                        if validation_batch_index>=contract.get('validation_flow_max_batches',float('inf')):break
                        batch={k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
                        batch=model_inputs(batch,cfg)
                        with torch.autocast('cuda',dtype=torch.bfloat16): loss,_=model(batch)
                        if not torch.isfinite(loss): raise RuntimeError('Non-finite validation loss')
                        size=batch['observation.state'].shape[0]
                        val_sum+=float(loss)*size; val_count+=size
            val=val_sum/val_count
            action_metrics=action_validation(model,validation,probe_indices,objective) if objective else {}
            selection_score=action_metrics.get('selection_score',val)
            if action_metrics:
                write_json(output/f'action-validation-epoch-{epoch:03d}.json',action_metrics)
                run.log({'validation/action_'+k:v for k,v in action_metrics.items()},step=step)
            if scheduler is not None and step>=warmup:scheduler.step(selection_score)
            saved=checkpoint(f'epoch-{epoch:03d}',epoch,True,val,selection_score if objective else None)
            improved=selection_score < best
            if improved:
                best=selection_score; stale=0
                write_json(output/'best_checkpoint.json',dict(path=str(saved.resolve()),epoch=epoch,validation_loss=val,
                    **(dict(selection_score=best,selected_by='sampled_action_score') if objective else {})))
                run.summary['best_action_score' if objective else 'best_validation_loss']=best; run.summary['best_epoch']=epoch
            else: stale+=1
            run.log({'validation/loss':val,'train/epoch_loss':epoch_sum/epoch_count,'completed_epoch':epoch},step=step)
            status(state='epoch_complete',epoch=epoch,validation_loss=val,
                **(dict(action_selection_score=selection_score,best_action_score=best) if objective else dict(best_validation_loss=best)))
            print(f'{family} completed epoch={epoch} flow={val:.6f} selection={selection_score:.6f} best={best:.6f}',flush=True)
            keep=contract.get('checkpoint_keep_last')
            if keep is not None:
                best_path=Path(json.loads((output/'best_checkpoint.json').read_text())['path']).resolve()
                completed=sorted(output.glob('epoch-*/provenance.json'))
                for old in completed[:-keep]:
                    if old.parent.resolve()!=best_path:
                        (output/'epoch_history').mkdir(exist_ok=True)
                        write_json(output/'epoch_history'/(old.parent.name+'.json'),json.loads(old.read_text()))
                        shutil.rmtree(old.parent)
            if stale>=contract['early_stopping_patience']: break
        status(state='completed',epoch=epoch,**(dict(best_action_score=best) if objective else dict(best_validation_loss=best)))
        run.finish()
    except BaseException as exc:
        status(state='failed',error=f'{type(exc).__name__}: {exc}')
        run.finish(exit_code=1)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    contract=json.loads(args.contract.read_text())
    # Separate child processes release all GPU/decoder resources between models.
    import subprocess
    args.output.mkdir(parents=True,exist_ok=True)
    families=contract.get('families',['krv100','domv'])
    for family in families:
        write_json(args.output/'queue.json',dict(active=family,state='running',order=families))
        command=[str(Path('workstation/finetune-python.sh').resolve()),str(Path(__file__).resolve()),
                 '--contract',str(args.contract.resolve()),'--output',str(args.output.resolve())]
        env=dict(os.environ, REAL_FINETUNE_FAMILY=family)
        result=subprocess.run(command,env=env)
        if result.returncode:
            write_json(args.output/'queue.json',dict(active=family,state='failed',returncode=result.returncode))
            raise SystemExit(result.returncode)
    write_json(args.output/'queue.json',dict(state='completed',order=families))


if __name__=='__main__':
    if os.environ.get('REAL_FINETUNE_FAMILY'):
        parser=argparse.ArgumentParser(); parser.add_argument('--contract',type=Path); parser.add_argument('--output',type=Path)
        args=parser.parse_args(); family=os.environ['REAL_FINETUNE_FAMILY']
        contract = json.loads(args.contract.read_text())
        if family == 'domv' and 'families' not in contract:
            # An already-running KRV100 process keeps its original in-memory limits.
            # Extend it at its clean completion boundary before starting DOMV.
            latest, _, stale = completed_progress(args.output/'krv100')
            if latest is None:
                raise RuntimeError('DOMV requires completed KRV100 epochs')
            if latest[1]['epoch'] < contract['max_epochs'] and stale < contract['early_stopping_patience']:
                import subprocess
                write_json(args.output/'queue.json',dict(active='krv100',state='continuing',order=['krv100','domv']))
                subprocess.run([str(Path('workstation/finetune-python.sh').resolve()),str(Path(__file__).resolve()),
                    '--contract',str(args.contract.resolve()),'--output',str(args.output.resolve())],
                    env=dict(os.environ,REAL_FINETUNE_FAMILY='krv100',REAL_FINETUNE_RESUME='1'),check=True)
                write_json(args.output/'queue.json',dict(active='domv',state='running',order=['krv100','domv']))
        train(contract,family,args.output/family,resume=os.environ.get('REAL_FINETUNE_RESUME')=='1')
    else: main()
