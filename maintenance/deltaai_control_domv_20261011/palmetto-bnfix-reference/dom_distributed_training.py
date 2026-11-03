"""Four/eight-GPU DOM training with a single global sample cursor."""
import contextlib
import json
import logging
import math
import os
from pathlib import Path
import signal
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from core.dom_stopping import record_validation, validate_rule
from core.dom_checkpoint import load_generation, restore_rng, resolve_generation
from core.dom_distributed import (RankBatchSampler, RankedData, distributed_lock, leader_call,
    rank_check, save_distributed, synchronize_checkpoint_buffers)
from core.dom_training import (amp_context, build_dataset, build_policy, enqueue_evaluations,
    evaluate, fixed_noise, load_family, make_loader, make_optimizer, make_scheduler, seed_all,
    to_device, worker_init)
from utils.dom_experiment import atomic_json, digest, simulator_evaluation_mode


def distributed_loader(data, runtime, epoch=0, cursor=0, sampling='frames', lengths=None):
    sampler = RankBatchSampler(data.lengths if lengths is None else lengths, runtime['microbatch'],
        dist.get_rank(),dist.get_world_size(),epoch,cursor,sampling)
    options = dict(num_workers=runtime['workers'], pin_memory=runtime['pin_memory'],worker_init_fn=worker_init,
                   generator=torch.Generator().manual_seed(42+epoch))
    if runtime['workers']:
        options.update(prefetch_factor=runtime['prefetch'],persistent_workers=True,
                       multiprocessing_context='spawn',timeout=300)
    return torch.utils.data.DataLoader(RankedData(data),batch_sampler=sampler,**options)


def wrap_policy(policy, device):
    cuda = str(device).startswith('cuda')
    return DistributedDataParallel(policy,device_ids=[device.index] if cuda else None,
        output_device=device.index if cuda else None,find_unused_parameters=True,
        broadcast_buffers=False,gradient_as_bucket_view=True)


def updates(model, optimizer, scheduler, loader, progress, precision, device):
    """Yield only after synchronized optimizer boundaries; no padded samples count."""
    world = dist.get_world_size()
    accumulated = torch.zeros(2,device=device,dtype=torch.float64)
    begin = time.monotonic()
    data_wait = 0.
    tick = time.monotonic()
    for batch in loader:
        data_wait += time.monotonic()-tick
        batch = to_device(batch,device)
        valid = int(batch['ddp.valid'].sum())
        total = int(batch['ddp.update_samples'][0])
        last = bool(batch['ddp.update_end'][0])
        with (contextlib.nullcontext() if last else model.no_sync()):
            with amp_context(precision,device):
                loss,_ = model(batch)
            (loss*(valid*world/total)).backward()
        accumulated[0] += loss.detach().double()*valid
        accumulated[1] += valid
        if not last:
            tick = time.monotonic()
            continue
        dist.all_reduce(accumulated)
        if int(accumulated[1]) != total:
            raise RuntimeError('Distributed update omitted or duplicated real samples')
        from core.dom_profiling import check_gradients
        check_gradients(model)
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        elapsed = torch.tensor(time.monotonic()-begin,device=device,dtype=torch.float64)
        dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
        progress['cursor'] += total
        progress['samples'] += total
        progress['updates'] += 1
        progress['training_seconds'] = progress.get('training_seconds',0.)+float(elapsed)
        record = {'loss':float(accumulated[0]/total),'samples_per_second':total/float(elapsed),
                  'update_latency_s':float(elapsed),'data_wait_s':data_wait,'samples':total}
        yield record
        accumulated.zero_()
        begin = tick = time.monotonic()
        data_wait = 0.


def stopped(flag, device):
    value = torch.tensor(int(flag['requested']),device=device)
    dist.all_reduce(value,op=dist.ReduceOp.MAX)
    return bool(value)


def restore_distributed(state):
    saved = state.get('distributed')
    if saved is None or saved['world_size'] != dist.get_world_size():
        raise ValueError('Exact resume requires the same distributed world size')
    restore_rng(saved['rng_by_rank'][dist.get_rank()])


def train_distributed(root, variant, runtime, recipe, device, resume_stopped=False):
    root = Path(root)
    manifest = load_family(root)
    simulator_evaluation_mode(root,recipe)
    max_epochs = recipe['epochs']
    rule = recipe.get('early_stopping')
    validate_rule(rule)
    if recipe['effective_batch'] != 352 or not 1 <= max_epochs <= 100:
        raise ValueError('The study requires batch 352 and at most 100 epochs')
    seed_all()
    destination = root/'runs'/variant
    fingerprint = digest({'manifest':manifest['fingerprint'],'recipe':recipe,
                          'variant':variant,'world_size':dist.get_world_size()})
    rank = dist.get_rank()
    with distributed_lock(destination):
        split = json.loads((root/'split.json').read_text())['splits']
        data = build_dataset(root,split['train'],training=True)
        if len(data) != manifest['train_frames']:
            raise ValueError('Training dataset differs from frozen frame count')
        val_loader = None
        if rank == 0:
            val = build_dataset(root,split['validation'])
            val_loader = make_loader(val,runtime,indices=json.loads((root/'validation_anchors.json').read_text()))
        policy = build_policy(root,variant,device)
        policy.config.optimizer_lr = recipe['lr']
        policy.config.scheduler_warmup_steps = 1000
        policy.config.scheduler_decay_steps = max_epochs*math.ceil(len(data)/352)
        optimizer = make_optimizer(policy,recipe['lr'],runtime['fused'])
        scheduler = make_scheduler(optimizer,policy.config.scheduler_decay_steps)
        progress = {'epoch':0,'cursor':0,'updates':0,'samples':0}
        best,wandb_id,state = None,None,None
        if (destination/'latest.json').exists():
            state = load_generation(destination,policy,optimizer,scheduler,fingerprint,restore_random=False)
            progress,best,wandb_id = state['progress'],state['best'],state['wandb_id']
        elif (destination/'provenance.json').exists():
            previous = json.loads((destination/'provenance.json').read_text())
            if previous['fingerprint'] != fingerprint:
                raise ValueError('Run directory belongs to a different distributed experiment')
            wandb_id = previous['wandb_id']
        model = wrap_policy(policy,device)
        if state is not None:
            restore_distributed(state)
            del state
        else:
            # Different denoising noise per rank; augmentation stays per sample.
            torch.manual_seed(42+rank)
            torch.cuda.manual_seed(42+rank)
        if progress.get('early_stopped') and resume_stopped and progress['epoch'] < max_epochs:
            progress['early_stopped'] = False
            progress.pop('completion_reason', None)
            progress['early_stopping']['bad_validations'] = 0
            progress.setdefault('resume_stopped_epochs', []).append(progress['epoch'])
            leader_call(lambda:(destination/'training_complete.json').unlink(missing_ok=True))
        elif (progress.get('early_stopped') or progress['epoch'] >= max_epochs) and not progress.get('pending_validation'):
            return progress
        writer = None
        def init_writer():
            nonlocal writer
            import wandb
            run_id = wandb_id or wandb.util.generate_id()
            writer = wandb.init(project='Kinematic-Residual-VLA',entity=os.environ.get('DVLA_WANDB_ENTITY','aparame-clemson-university'),
                name=variant,id=run_id,resume='allow',dir=str(destination),
                config={'manifest':manifest,'recipe':recipe,'variant':variant,'runtime':runtime,'world_size':dist.get_world_size()})
            atomic_json(destination/'provenance.json',{'fingerprint':fingerprint,'wandb_id':run_id,
                'variant':variant,'manifest':manifest['fingerprint'],'runtime':runtime,'recipe':recipe,
                'world_size':dist.get_world_size(),'backend':'DDP/NCCL'})
            return run_id
        wandb_id = leader_call(init_writer)
        stop = {'requested':False}
        def request_stop(signum,frame): stop['requested']=True
        handlers = {sig:signal.signal(sig,request_stop) for sig in (signal.SIGUSR1,signal.SIGTERM)}
        last_save = time.monotonic()
        def save(milestone=False, final_checkpoint=False):
            nonlocal last_save
            save_distributed(destination,policy,optimizer,scheduler,progress,fingerprint,wandb_id,best,milestone,device,final_checkpoint)
            last_save = time.monotonic()
        def finish_epoch():
            nonlocal best
            def validate():
                try:
                    return evaluate(policy,val_loader,runtime['precision'],device,should_stop=lambda:stop['requested'])
                except InterruptedError:
                    return None
            metrics = leader_call(validate)
            if metrics is None:
                stop['requested']=True
                return False
            progress['early_stopped'] = record_validation(progress,metrics,rule)
            finished = progress['early_stopped'] or progress['epoch'] >= max_epochs
            if finished:
                progress['completion_reason'] = 'early_stopping' if progress['early_stopped'] else 'epoch_cap'
            value = metrics['h20/action_mse']
            if best is None or value < best['value']:
                best = {'value':value,'epoch':progress['epoch'],'updates':progress['updates']}
            def publish():
                atomic_json(destination/f"validation-epoch-{progress['epoch']:03d}.json",metrics)
                report = {'Validation/'+key:value for key,value in metrics.items()}
                report.update({'Stop/bad_validations':progress.get('early_stopping',{}).get('bad_validations',0),
                               'Stop/early_stopped':progress['early_stopped'],'Stop/completed_epochs':progress['epoch']})
                writer.log(report,step=progress['updates'])
                logging.info('%s epoch=%d validation_mse=%.6g bad_validations=%d completion=%s',
                    variant,progress['epoch'],value,report['Stop/bad_validations'],progress.get('completion_reason','training'))
            leader_call(publish)
            progress['pending_validation']=False
            milestone = finished or progress['epoch'] in (1,5) or progress['epoch']%10==0
            save(milestone,final_checkpoint=finished)
            if milestone:
                leader_call(lambda:enqueue_evaluations(root,destination,variant,progress,best))
            return True
        try:
            if not (destination/'latest.json').exists(): save()
            if progress.get('pending_validation') and not finish_epoch():
                save();return {**progress,'interrupted':True}
            optimizer.zero_grad(set_to_none=True)
            while progress['epoch'] < max_epochs and not progress.get('early_stopped'):
                loader = distributed_loader(data,runtime,progress['epoch'],progress['cursor'],recipe['sampling'])
                for record in updates(model,optimizer,scheduler,loader,progress,runtime['precision'],device):
                    if rank == 0 and (progress['updates']%10==0 or progress['updates']==1):
                        logging.info('%s ranks=%d epoch=%.6f updates=%d samples=%d loss=%.6g rate=%.2f',
                            variant,dist.get_world_size(),progress['epoch']+progress['cursor']/len(data),progress['updates'],
                            progress['samples'],record['loss'],record['samples_per_second'])
                        writer.log({'Train/loss':record['loss'],'Train/lr':optimizer.param_groups[0]['lr'],
                            'Train/samples':progress['samples'],'Train/optimizer_updates':progress['updates'],
                            'Train/epoch':progress['epoch']+progress['cursor']/len(data),
                            'Train/samples_per_second':record['samples_per_second'],'Train/update_latency_s':record['update_latency_s'],
                            'Train/world_size':dist.get_world_size()},step=progress['updates'])
                    halt = stopped(stop,device)
                    due = torch.tensor(int(time.monotonic()-last_save>=1800),device=device)
                    dist.all_reduce(due,op=dist.ReduceOp.MAX)
                    if halt or bool(due):save()
                    if halt:return {**progress,'interrupted':True}
                del loader
                if progress['cursor'] != len(data):raise RuntimeError('Distributed epoch omitted samples')
                progress['epoch'] += 1;progress['cursor']=0;progress['pending_validation']=True
                save()
                if not finish_epoch():save();return {**progress,'interrupted':True}
            leader_call(lambda:enqueue_evaluations(root,destination,variant,progress,best))
            leader_call(lambda:atomic_json(destination/'training_complete.json',{**progress,'fingerprint':fingerprint,'best':best}))
            return progress
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)
            if writer is not None:writer.finish(exit_code=0 if progress.get('completion_reason') and not progress.get('pending_validation') else 1)


def distributed_smoke(root, variant, runtime, device):
    """Two real updates including an uneven tail, followed by exact RNG reload."""
    import psutil
    root = Path(root)
    manifest = load_family(root)
    world = dist.get_world_size()
    properties = torch.cuda.get_device_properties(device)
    hardware = {'name':properties.name,'total_memory':properties.total_memory,
                'torch':torch.__version__,'cuda':torch.version.cuda,'capability':[properties.major,properties.minor]}
    key = digest({'hardware':hardware,'world_size':world,'runtime':runtime,'variant':variant})[:16]
    destination = root/'distributed-smoke'/key
    fingerprint = digest({'manifest':manifest['fingerprint'],'key':key})
    with distributed_lock(destination):
        seed_all()
        data = build_dataset(root,json.loads((root/'pilot_episodes.json').read_text())[:8],training=True)
        policy = build_policy(root,variant,device)
        model = wrap_policy(policy,device)
        optimizer = make_optimizer(policy,3e-5,runtime['fused'])
        scheduler = make_scheduler(optimizer,10000,warmup=0)
        progress = {'epoch':0,'cursor':0,'updates':0,'samples':0}
        torch.cuda.reset_peak_memory_stats(device)
        loader = distributed_loader(data,runtime,lengths=[352+world-1])
        records = list(updates(model,optimizer,scheduler,loader,progress,runtime['precision'],device))
        del loader
        if progress['updates'] != 2 or progress['samples'] != 352+world-1:
            raise ValueError('Distributed smoke lost the uneven tail')
        # State projection must receive velocity gradients and synchronize.
        if variant in ('domv100','krv100','krvp100'):
            state = optimizer.state[policy.model.state_proj.weight]
            if not state['exp_avg'][:,6:9].abs().sum():
                raise ValueError('Velocity projection received no optimizer updates')
        probe = next(iter(distributed_loader(data,{**runtime,'microbatch':2,'workers':0},lengths=[world*2])))
        probe = to_device(probe,device)
        noise = fixed_noise(probe,policy,device)
        # Compare the same canonical inference state that rank zero saves.
        synchronize_checkpoint_buffers(policy)
        policy.eval()
        def predict_and_check():
            with torch.no_grad(),amp_context(runtime['precision'],device):
                policy.reset();prediction=policy.predict_action_chunk(probe,noise=noise)
                poisoned = {**probe,'dynamic_object.future_velocity_xyz':torch.full((len(prediction),20,3),999.,device=device),
                            'dynamic_object.future_position_xyz':torch.full((len(prediction),20,3),-999.,device=device)}
                policy.reset();torch.testing.assert_close(prediction,policy.predict_action_chunk(poisoned,noise=noise))
            if prediction.shape[1] != 20 or not torch.isfinite(prediction).all():raise ValueError('Invalid 20-step prediction')
            return prediction
        logging.info('rank=%d smoke stage=canonical_prediction',dist.get_rank())
        prediction = rank_check(predict_and_check)
        logging.info('rank=%d smoke stage=checkpoint_write',dist.get_rank())
        save_distributed(destination,policy,optimizer,scheduler,progress,fingerprint,None,device=device)
        expected_random = torch.rand(4,device=device)
        with torch.no_grad():policy.model.state_proj.weight.add_(1.)
        def reload_and_check():
            saved = load_generation(destination,policy,optimizer,scheduler,fingerprint,restore_random=False)
            restore_distributed(saved)
            torch.testing.assert_close(expected_random,torch.rand(4,device=device),rtol=0,atol=0)
            with torch.no_grad(),amp_context(runtime['precision'],device):
                policy.reset();torch.testing.assert_close(prediction,policy.predict_action_chunk(probe,noise=noise))
            return saved
        logging.info('rank=%d smoke stage=checkpoint_reload',dist.get_rank())
        saved = rank_check(reload_and_check)
        logging.info('rank=%d smoke stage=memory_headroom',dist.get_rank())
        peak = torch.cuda.max_memory_reserved(device)
        process = psutil.Process()
        ram = process.memory_info().rss+sum(p.memory_info().rss for p in process.children(recursive=True))
        from core.dom_distributed import init_control_group
        group = init_control_group()
        peaks = torch.tensor([peak,ram],dtype=torch.float64)
        dist.all_reduce(peaks,op=dist.ReduceOp.SUM,group=group)
        safe = torch.tensor(int(peak <= .85*properties.total_memory))
        dist.all_reduce(safe,op=dist.ReduceOp.MIN,group=group)
        host_limit = int(os.environ.get('SLURM_MEM_PER_NODE','524288'))*1024**2
        if not safe or float(peaks[1]) > .85*host_limit:raise ValueError('Distributed runtime lacks 15% memory headroom')
        report = {'complete':True,'manifest':manifest['fingerprint'],'world_size':world,'variant':variant,
            'hardware':hardware,'runtime':runtime,'global_effective_batch':352,'prediction_horizon':20,'execution_horizon':20,
            'strict_reload':True,'rank_rng_reload':True,'canonical_checkpoint_buffers':True,
            'future_inputs_ignored':True,'samples':progress['samples'],
            'optimizer_updates':progress['updates'],'full_update_measurement':records[0],
            'peak_gpu_bytes_this_rank':peak,'sum_gpu_peak_bytes':float(peaks[0]),'sum_ram_bytes':float(peaks[1]),'safe':True}
        report['fingerprint']=digest(report)
        leader_call(lambda:atomic_json(root/'distributed-profiles'/f'{key}.json',report))
        policy.train()
        del model,policy,optimizer,saved
        torch.cuda.empty_cache()
        return report
