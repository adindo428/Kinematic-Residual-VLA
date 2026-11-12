"""Architecture-preserving real-data fine-tuning for released baseline/KR pairs.

One family per process; must pass the disposable GPU check before training.
"""
import argparse
from contextlib import nullcontext
import itertools
import hashlib
import json
import math
from pathlib import Path
import random
import time
from datetime import datetime,timezone
import shutil

import numpy as np
import torch
import wandb

from family_data import RealFamilyDataset
from family_policy import load_released_policy,signature
from real_robot.action_objective import horizon_weights,physical_loss,physical_errors,fixed_rng
from real_robot.training_augmentation import augment_training_batch
from real_robot.fast_training_augmentation import augment_family_batch
from real_robot.training_ema import TrainingEMA
from utils.real_robot_contract import absolute_pose
from family_resume import resume_order,select_checkpoint


def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def cuda(batch):return {k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}


def slice_batch(batch,start,end):return {k:v[start:end] if torch.is_tensor(v) or isinstance(v,list) else v for k,v in batch.items()}


def merge_batches(batches):
    return {k:torch.cat([b[k] for b in batches]) if torch.is_tensor(batches[0][k]) else
            sum([b[k] for b in batches],[]) if isinstance(batches[0][k],list) else batches[0][k] for k in batches[0]}


def performance_settings(campaign,method,family=None):
    settings=dict(flow_microbatch=2 if method=='smolvla' else 1,auxiliary_microbatch=1,
        loader_workers=0,prefetch_factor=2,vectorized_photometric=False)
    settings.update(campaign.get('performance',{}).get(method,{}))
    if family is not None:settings.update(campaign.get('family_performance',{}).get(family,{}))
    if settings['flow_microbatch']<=0 or 16%settings['flow_microbatch'] or settings['auxiliary_microbatch']<=0:
        raise ValueError('Microbatches must preserve the effective flow batch of 16')
    return settings


def worker_init(_):torch.set_num_threads(1)


def training_loader(data,sampler,micro,settings,generator):
    workers=settings['loader_workers']
    return torch.utils.data.DataLoader(data,batch_size=micro,sampler=sampler,num_workers=workers,pin_memory=True,generator=generator,
        **(dict(prefetch_factor=settings['prefetch_factor'],multiprocessing_context='spawn',persistent_workers=True,worker_init_fn=worker_init) if workers else {}))


@torch.no_grad()
def validate(model,data,indices,objective):
    model.eval();sums={};counts={}
    with fixed_rng(objective['validation_seed'],torch.device('cuda')):
        loader=torch.utils.data.DataLoader(torch.utils.data.Subset(data,indices),batch_size=1,num_workers=0)
        for batch in loader:
            batch=cuda(batch);ep=int(batch['episode_index'][0]);frame=int(batch['frame_index'][0])
            target=absolute_pose(batch['action'],batch['reference_state'][:,None,:],'wrapped_xyz_v1');valid=~batch['action_is_pad']
            for seed in objective['validation_noise_seeds']:
                noise=torch.randn((1,20,32),device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed+ep*100000+frame))
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model.sample_absolute(batch,noise)
                if not torch.isfinite(pred).all():raise RuntimeError('Nonfinite sampled validation prediction')
                loss,_=physical_loss(pred,target,valid,objective)
                sums['selection_score']=sums.get('selection_score',0)+float(loss);counts['selection_score']=counts.get('selection_score',0)+1
                for scope,mask in [('first',valid & (torch.arange(20,device='cuda')[None,:]==0)),('full',valid)]:
                    for key,value in physical_errors(pred.float(),target.float()).items():
                        key=scope+'/'+key;sums[key]=sums.get(key,0)+float(value[mask].sum());counts[key]=counts.get(key,0)+int(mask.sum())
    return {k:sums[k]/counts[k] for k in sums}


def disposable_check(model,data,objective,output):
    """Verify native inference, gradient sampler, optimizer and frozen architecture."""
    check_batch=2 if model.method=='smolvla' else 1
    batch=cuda(next(iter(torch.utils.data.DataLoader(data,batch_size=check_batch,num_workers=0))))
    velocity_check=model.check_velocity_tokens(batch)
    model.eval();noise=torch.randn(check_batch,20,32,device='cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        # Compare the original released sampler under the checkpoint's training
        # prompt/affine contract. VLASH's high-level API re-tokenizes that
        # prepared prompt differently; PI05's API does not accept fixed noise.
        native=model.sample_absolute(batch,noise,reference=True)
        explicit=model.sample_absolute(batch,noise)
    torch.testing.assert_close(native,explicit,atol=2e-5,rtol=2e-5)
    model.train();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-6,betas=(.9,.95))
    torch.cuda.reset_peak_memory_stats();optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        flow=model(batch,horizon_weights(objective,20,'cuda'))
    flow.backward();optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        pred=model.sample_absolute(batch,noise)
        if model.method=='pi05':torch.testing.assert_close(pred.detach(),native,atol=2e-5,rtol=2e-5)
        target=absolute_pose(batch['action'],batch['reference_state'][:,None,:],'wrapped_xyz_v1')
        reconstruction,_=physical_loss(pred,target,~batch['action_is_pad'],objective)
    reconstruction.backward()
    grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().sum()>0 for g in grads):
        raise RuntimeError('Sampled action reconstruction failed to backpropagate')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();model.assert_architecture()
    ema=TrainingEMA(model);ema.update(model)
    with ema.apply(model):model.assert_architecture()
    result=dict(passed=True,parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),architecture_signature=model.architecture_signature,
        matched_native_decode_max_error=float((native-explicit).abs().max()),flow_loss=float(flow.detach()),
        reconstruction_loss=float(reconstruction.detach()),gradient_norm=float(norm),gradient_tensors=len(grads),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,weights_saved=False,velocity_conditioning=model.velocity_conditioning,velocity_input_check=velocity_check)
    write(output,result);print(json.dumps(result),flush=True)


def train(args,campaign,entry,model,data,stats):
    output=args.output;output.mkdir(parents=True,exist_ok=args.resume)
    c=campaign['training_recipe'];objective=c['action_objective'];family=entry['family']
    val=RealFamilyDataset(campaign,family,'validation');indices=c['action_validation_indices']
    weights=np.load(c['sampling_weights'])
    if hashlib.sha256(Path(c['sampling_weights']).read_bytes()).hexdigest()!=c['sampling_weights_sha256']:
        raise RuntimeError('Balanced sampler weights changed')
    if len(weights)!=len(data):raise RuntimeError('Balanced sampler anchor layout differs')
    sampler=torch.utils.data.WeightedRandomSampler(weights,len(data),replacement=True,generator=torch.Generator().manual_seed(42))
    settings=performance_settings(campaign,model.method,family)
    micro=settings['flow_microbatch'];accumulation=16//micro
    loader_generator=torch.Generator().manual_seed(42)
    loader=training_loader(data,sampler,micro,settings,loader_generator)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=c['learning_rate'],betas=(.9,.95),weight_decay=c['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=2,min_lr=3e-7)
    ema=TrainingEMA(model,campaign['ema']['decay']);step=0;best=float('inf');stale=0
    start_epoch=1;partial_order=None;resume_path=None;resume_id=None
    if args.resume:
        resume_path,snapshot,completed=select_checkpoint(output,args.resume_checkpoint)
        if snapshot.get('velocity_conditioning')!=model.velocity_conditioning:
            raise RuntimeError('Resume conditioning differs; start a new run when adding velocity prompts')
        if snapshot['architecture_signature']!=model.architecture_signature or snapshot['statistics']!=stats:
            raise RuntimeError('Resume architecture or normalization differs')
        start_epoch,partial_order,consumed=resume_order(snapshot,weights,len(data),completed)
        model.native.load_state_dict(snapshot['model'],strict=True)
        optimizer.load_state_dict(snapshot['optimizer']);scheduler.load_state_dict(snapshot['scheduler']);ema.load_state_dict(snapshot['ema'])
        step=snapshot['step'];rng=snapshot['rng']
        torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state_all(rng['cuda']);np.random.set_state(rng['numpy']);random.setstate(rng['python'])
        sampler.generator.set_state(rng['sampler'])
        if 'loader' in rng:loader_generator.set_state(rng['loader'])
        baseline=json.loads((output/'baseline_selection.json').read_text());best=baseline['selection_score']
        last_completed=start_epoch-1
        for path in sorted(output.glob('validation-epoch-*.json')):
            if int(path.stem.rsplit('-',1)[1])>last_completed:continue
            record=json.loads(path.read_text());score=record[record['chosen']]['selection_score']
            if score<best:best=score;stale=0
            else:stale+=1
        pointer=json.loads((output/'best_checkpoint.json').read_text())
        pointer['path']=str((output/Path(pointer['path']).name).resolve());write(output/'best_checkpoint.json',pointer)
        if not math.isclose(best,pointer['selection_score'],rel_tol=1e-10):raise RuntimeError('Best checkpoint and epoch metrics disagree')
        previous_status=json.loads((output/'status.json').read_text());resume_id=previous_status['wandb_url'].rstrip('/').rsplit('/',1)[-1]
        events_path=output/'resume_events.json';events=json.loads(events_path.read_text()) if events_path.exists() else []
        events.append(dict(time_utc=datetime.now(timezone.utc).isoformat(),checkpoint=str(resume_path.resolve()),step=step,
            epoch=start_epoch,consumed_anchors=consumed,remaining_anchors=None if partial_order is None else len(partial_order),
            performance=settings,optimizer_restored=True,ema_updates=ema.updates,sampler_replay_verified=True,
            note='Sample order/cursor, model, optimizer, scheduler, EMA and RNG restored. Microbatch grouping, worker RNG and floating-point reduction order may differ.'))
        write(events_path,events);del snapshot
        print(f'{family} resumed step={step} epoch={start_epoch} remaining={None if partial_order is None else len(partial_order)} performance={settings}',flush=True)
    run=wandb.init(entity=c['wandb']['entity'],project=c['wandb']['project'],name=campaign.get('run_name_prefix','real-v6-')+family,
        id=resume_id,resume='must' if args.resume else None,allow_val_change=True,
        dir=str(output),config=dict(campaign=campaign,family=family,microbatch=micro,effective_batch=16,
        architecture_signature=model.architecture_signature,trainable_parameters=sum(p.numel() for p in parameters)))
    if args.resume and not (output/'run_config.pre-throughput.json').exists():shutil.copyfile(output/'run_config.json',output/'run_config.pre-throughput.json')
    write(output/'run_config.json',dict(campaign=campaign,entry=entry,statistics=stats,microbatch=micro,gradient_accumulation=accumulation,
        performance=settings,resumed_from=None if resume_path is None else str(resume_path.resolve()),
        architecture_signature=model.architecture_signature,trainable_names=[n for n,p in model.native.named_parameters() if p.requires_grad]))
    # Keep a small trainable-only copy for the paired final comparison. Frozen
    # backbone tensors are shared and are never rewritten during training.
    if not args.resume:torch.save({n:p.detach().cpu() for n,p in model.native.named_parameters() if p.requires_grad},output/'initial_trainable.pt')
    elif not (output/'initial_trainable.pt').exists():raise RuntimeError('Original trainable weights for final comparison are missing')
    def status(**values):write(output/'status.json',dict(family=family,step=step,wandb_url=run.url,**values))
    def save(name,epoch,kind,score=None,include_optimizer=True,completed_epoch=False):
        model.assert_architecture();path=output/name;temporary=path.with_suffix('.tmp')
        with ema.apply(model) if kind=='ema' else nullcontext():
            torch.save(dict(model=model.native.state_dict(),statistics=stats,entry=entry,epoch=epoch,step=step,
                selected_weights=kind,selection_score=score,architecture_signature=model.architecture_signature,
                velocity_conditioning=model.velocity_conditioning,
                completed_epoch=completed_epoch,effective_batch=16,performance=settings,
                optimizer=optimizer.state_dict() if include_optimizer else None,
                scheduler=scheduler.state_dict(),ema=ema.state_dict(),
                rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate(),sampler=sampler.generator.get_state(),loader=loader_generator.get_state())),temporary)
        temporary.replace(path);return path
    try:
        initial_path=output/'epoch-000-initial.pt'
        if not args.resume:
            status(state='baseline_validation',epoch=0)
            initial=validate(model,val,indices,objective);best=initial['selection_score'];write(output/'baseline_selection.json',initial)
            initial_path=save(initial_path.name,0,'raw',best,False)
            write(output/'best_checkpoint.json',dict(path=str(initial_path.resolve()),epoch=0,selection_score=best,weights='raw'))
        epoch=start_epoch-1
        for epoch in range(start_epoch,c['max_epochs']+1):
            model.train();status(state='training',epoch=epoch,performance=settings)
            partial_loader=training_loader(data,partial_order,micro,settings,loader_generator) if epoch==start_epoch and partial_order is not None else None
            iterator=iter(partial_loader if partial_loader is not None else loader)
            window_started=time.perf_counter();window_samples=0
            while True:
                group=list(itertools.islice(iterator,accumulation))
                if not group:break
                total=sum(len(b['action']) for b in group);optimizer.zero_grad(set_to_none=True)
                if step<200:
                    for g in optimizer.param_groups:g['lr']=c['learning_rate']*(step+1)/200
                example_pool=[];loss_sum=0.
                for batch in group:
                    batch=cuda(batch);batch,_=(augment_family_batch if settings['vectorized_photometric'] else augment_training_batch)(batch,c['augmentation'])
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(batch,horizon_weights(objective,20,'cuda'))
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training flow loss')
                    (loss*len(batch['action'])/total).backward();loss_sum+=float(loss.detach())*len(batch['action'])/total
                    if sum(len(b['action']) for b in example_pool)<objective['auxiliary_batch_size']:example_pool.append(batch)
                aux_metrics={}
                if step%objective['auxiliary_every_steps']==0:
                    pool=slice_batch(merge_batches(example_pool),0,objective['auxiliary_batch_size']);count=len(pool['action'])
                    # Four examples with sequential backward passes: same mean objective,
                    # bounded peak memory for all six native variants.
                    for i in range(0,count,settings['auxiliary_microbatch']):
                        small=slice_batch(pool,i,i+settings['auxiliary_microbatch']);target=absolute_pose(small['action'],small['reference_state'][:,None,:],'wrapped_xyz_v1')
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            prediction=model.sample_absolute(small);aux,terms=physical_loss(prediction,target,~small['action_is_pad'],objective)
                        if not torch.isfinite(aux):raise RuntimeError('Nonfinite reconstruction loss')
                        (objective['auxiliary_weight']*aux*len(small['action'])/count).backward()
                        for key,value in terms.items():aux_metrics[key]=aux_metrics.get(key,0.)+value*len(small['action'])/count
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                optimizer.step();ema.update(model);step+=1;window_samples+=total
                if step==1 or step%20==0 or aux_metrics:
                    run.log(dict({'train/flow_loss':loss_sum,'train/grad_norm':float(norm),'train/lr':optimizer.param_groups[0]['lr'],'epoch':epoch},
                        **{'reconstruction/'+k:v for k,v in aux_metrics.items()}),step=step)
                if step==1 or step%20==0:
                    rate=window_samples/(time.perf_counter()-window_started)
                    run.log({'train/samples_per_second':rate,'gpu/peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30},step=step)
                    status(state='training',epoch=epoch,flow_loss=loss_sum,samples_per_second=rate,performance=settings)
                    print(f'{family} epoch={epoch} step={step} flow={loss_sum:.6f} samples_per_second={rate:.2f}',flush=True)
                    window_started=time.perf_counter();window_samples=0
                if step==100 or step%500==0:
                    path=save('recovery.pt',epoch,'raw');write(output/'latest_checkpoint.json',dict(path=str(path.resolve()),completed_epoch=False,step=step))
            del iterator
            if partial_loader is not None:del partial_loader
            raw=validate(model,val,indices,objective)
            with ema.apply(model):averaged=validate(model,val,indices,objective)
            kind,chosen=('ema',averaged) if averaged['selection_score']<raw['selection_score'] else ('raw',raw)
            score=chosen['selection_score'];scheduler.step(score)
            write(output/f'validation-epoch-{epoch:03d}.json',dict(raw=raw,ema=averaged,chosen=kind))
            run.log({kind+'/'+k:v for kind,metrics in [('validation_raw',raw),('validation_ema',averaged)] for k,v in metrics.items()},step=step)
            # Always keep raw optimizer/EMA training state separate from EMA-selected inference weights.
            latest=save('latest.pt',epoch,'raw',raw['selection_score'],completed_epoch=True)
            write(output/'latest_checkpoint.json',dict(path=str(latest.resolve()),completed_epoch=True,epoch=epoch,step=step))
            if score<best:
                best=score;stale=0;path=save('best.pt',epoch,kind,score,False)
                write(output/'best_checkpoint.json',dict(path=str(path.resolve()),epoch=epoch,weights=kind,selection_score=score))
                run.summary['best_action_score']=best;run.summary['best_epoch']=epoch
                if initial_path.exists():initial_path.unlink()
            else:stale+=1
            status(state='epoch_complete',epoch=epoch,selection_score=score,best_action_score=best,selected_weights=kind)
            print(f'{family} epoch={epoch} action_score={score:.6f} best={best:.6f} chosen={kind}',flush=True)
            if stale>=c['early_stopping_patience']:break
        status(state='final_evaluation',epoch=epoch,best_action_score=best)
        if 'final_evaluation_episodes' in c:selected_episodes=c['final_evaluation_episodes']
        else:
            old_report=Path(__file__).resolve().parents[1]/'scratch/real_robot/action-evaluation/validation-10-seed12345/summary.json'
            selected_episodes=json.loads(old_report.read_text())['episodes']
        ten_indices=[i for i,(ep,_) in enumerate(val.anchors) if ep in selected_episodes]
        paired_objective=dict(objective,validation_noise_seeds=[12345])
        initial=torch.load(output/'initial_trainable.pt',map_location='cpu',weights_only=True)
        with torch.no_grad():
            for n,p in model.native.named_parameters():
                if n in initial:p.copy_(initial[n].to(p.device))
        before=validate(model,val,ten_indices,paired_objective);del initial
        selected=json.loads((output/'best_checkpoint.json').read_text())
        checkpoint=torch.load(selected['path'],map_location='cpu',weights_only=False,mmap=True)
        with torch.no_grad():
            for n,p in model.native.named_parameters():
                if p.requires_grad:p.copy_(checkpoint['model'][n].to(p.device))
        del checkpoint
        after=validate(model,val,ten_indices,paired_objective)
        write(output/'before_after.json',dict(episodes=selected_episodes,anchors=len(ten_indices),noise_seeds=[12345],before=before,after=after,
            interpretation='Same real normalization, native architecture, instruction seed, observations, targets and noise before/after; selected checkpoint, not last epoch. Expert-target agreement, not robot task success.'))
        status(state='completed',epoch=epoch,best_action_score=best);run.finish()
    except BaseException as exc:
        status(state='failed',error=f'{type(exc).__name__}: {exc}');run.finish(exit_code=1);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--family',required=True)
    p.add_argument('--mode',choices=['inspect','check','train'],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--resume-checkpoint',type=Path)
    args=p.parse_args();root=args.campaign.parent;campaign=json.loads(args.campaign.read_text())
    entry=next(x for x in campaign['entries'] if x['family']==args.family)
    if args.mode=='train' and campaign.get('benchmark_before_training'):
        benchmark_path=root/'benchmarks'/f'{args.family}.json'
        benchmark=json.loads(benchmark_path.read_text())
        recorded=campaign.get('benchmark_manifest',{}).get(args.family)
        if recorded and hashlib.sha256(benchmark_path.read_bytes()).hexdigest()!=recorded['sha256']:
            raise RuntimeError('Saved benchmark changed after settings selection')
        if benchmark.get('velocity_conditioning')!=entry.get('velocity_conditioning'):raise RuntimeError('Benchmark conditioning differs')
        if not benchmark.get('passed') or benchmark.get('state')!='completed' or benchmark['checkpoint_sha256']!=entry['sha256']:
            raise RuntimeError('A completed matching benchmark is required before training')
        expected=performance_settings(campaign,args.family.split('_')[0],args.family)
        if any(expected.get(k)!=v for k,v in benchmark['selected'].items()):raise RuntimeError('Training settings differ from saved benchmark')
    verified=json.loads((root/'downloads.json').read_text())['models'].get(args.family,{})
    if verified.get('state')!='verified' or verified.get('sha256')!=entry['sha256']:raise RuntimeError('Checkpoint not verified')
    stats=json.loads((root/'real_statistics.json').read_text());torch.manual_seed(42);np.random.seed(42);random.seed(42)
    model=load_released_policy(root,entry,stats,device='cpu' if args.mode=='inspect' else 'cuda')
    if args.mode=='inspect':
        result=dict(strict_load=True,architecture_signature=model.architecture_signature,parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        write(args.output,result);print(json.dumps(result));return
    data=RealFamilyDataset(campaign,args.family)
    if len(data)!=stats['training_anchors']:raise RuntimeError('Training anchor population differs from statistics')
    if args.mode=='check':disposable_check(model,data,campaign['training_recipe']['action_objective'],args.output)
    else:
        readiness=json.loads((root/f'{args.family}-gpu-readiness.json').read_text())
        if readiness.get('velocity_conditioning')!=model.velocity_conditioning:raise RuntimeError('Readiness conditioning differs')
        if not readiness['passed'] or readiness['architecture_signature']!=model.architecture_signature:raise RuntimeError('Missing matching GPU readiness check')
        train(args,campaign,entry,model,data,stats)


if __name__=='__main__':main()
