"""Two-A100-80GB decoded profiling, fail-closed readiness, and resumable training."""
import argparse
import contextlib
import gc
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

# Slurm stores submitted script bodies. Resolve the required codec runtime in
# the versioned Python entry point as well, so all dependency-queued workers
# and their spawned data workers receive it before any native library import.
if __name__=='__main__' and os.environ.get('KRFM_CODEC_RUNTIME') != '1':
    cuda_lib='/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/lib64'
    if not Path(cuda_lib,'libnppicc.so.12').exists():
        raise RuntimeError('Required Palmetto CUDA12 codec runtime unavailable')
    environment=dict(os.environ,KRFM_CODEC_RUNTIME='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    environment['LD_LIBRARY_PATH']=cuda_lib+':'+environment.get('LD_LIBRARY_PATH','')
    os.execve(sys.executable,[sys.executable,*sys.argv],environment)

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from common import atomic, fingerprint, sha
from data import Batches, DOM
from policy import Policy
from runtime import Resources, synchronized_flags, validate_runtime


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}


def restore_rng(r):
    random.setstate(r['python']); np.random.set_state(r['numpy'])
    torch.set_rng_state(r['torch']); torch.cuda.set_rng_state(r['cuda'])


def save(path, policy, optimizer, scheduler, progress, identity):
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, rng_state())
    if dist.get_rank() == 0:
        temp = path.with_suffix('.tmp')
        torch.save({'model': policy.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'progress': progress,
                    'rng': ranks, 'identity': identity}, temp)
        os.replace(temp, path)
        atomic(path.with_suffix('.json'), {'progress': progress, 'identity': identity, 'file': str(path)})
    dist.barrier()


def load(path, policy, optimizer, scheduler, identity, rank, parent_identity=None):
    # Only load locally generated trusted training state.
    c = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if c['identity'] != (parent_identity or identity):
        raise ValueError('Checkpoint source/data/model/recipe/runtime identity differs')
    policy.load_state_dict(c['model'], strict=True)
    # Free existing GPU Adam moments before restoring CPU checkpoint moments.
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    gc.collect(); torch.cuda.empty_cache()
    optimizer.load_state_dict(c['optimizer']); scheduler.load_state_dict(c['scheduler'])
    restore_rng(c['rng'][rank])
    return c['progress']


def worker_init(_):
    torch.set_num_threads(1)


def loader(data, micro, global_batch, rank, world, epoch=0, cursor=0, workers=4, prefetch=2):
    return torch.utils.data.DataLoader(data, batch_sampler=Batches(len(data), epoch, cursor,
        global_batch, micro, rank, world), num_workers=workers,
        generator=torch.Generator().manual_seed(42+epoch), pin_memory=True,
        worker_init_fn=worker_init, **({'prefetch_factor': prefetch, 'persistent_workers': True,
        'multiprocessing_context': 'spawn'} if workers else {}))


def optimizer_for(policy, recipe):
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
        lr=recipe['lr'], betas=(.9,.95), eps=1e-8, weight_decay=recipe['weight_decay'],
        fused=(recipe['method']=='pi05'))
    def factor(step):
        if step < 1000:
            return (step+1)/1000
        t = min(1, (step-1000)/max(1, recipe['steps']-1000))
        floor = recipe['decay_lr']/recipe['lr']
        return floor+(1-floor)*.5*(1+math.cos(math.pi*t))
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def update(model, optimizer, scheduler, iterator, accumulation, clip):
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((),device=next(model.parameters()).device)
    for i in range(accumulation):
        batch = next(iterator)
        with model.no_sync() if i+1 < accumulation else contextlib.nullcontext():
            with torch.autocast('cuda', dtype=torch.bfloat16,cache_enabled=False):
                loss = model(batch)
            (loss/accumulation).backward()
        loss_sum += loss.detach() / accumulation
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    if not (torch.isfinite(norm) & torch.isfinite(loss_sum)):
        raise ValueError('Nonfinite gradient')
    optimizer.step(); scheduler.step()
    return loss_sum


@torch.no_grad()
def validate(policy, dataset, rank, world, limit):
    policy.eval()
    indices = np.random.default_rng(1234).choice(len(dataset), min(limit,len(dataset)), replace=False)[rank::world]
    data = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices.tolist()), batch_size=1, num_workers=0)
    total = torch.zeros(2, device='cuda')
    for batch in data:
        # Fixed per-anchor noise via RNG fork prevents validation from changing
        # the training noise stream. No official DOM test scores select weights.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(42 + int(batch['sample_index'][0]))
            with torch.autocast('cuda', dtype=torch.bfloat16,cache_enabled=False):
                prediction = policy.predict(batch)
        if prediction.shape[-2:] != (20,7) or not torch.isfinite(prediction).all():
            raise ValueError('Inference must return a finite, restored physical 20x7 chunk')
        mask = (~batch['action_is_pad']).to('cuda').unsqueeze(-1)
        error = (prediction.float()-batch['action'].to('cuda')).square()
        total[0] += (error*mask).sum(); total[1] += mask.sum()*7
    dist.all_reduce(total)
    policy.train()
    return float(total[0]/total[1].clamp_min(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['profile','train','benchmark'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--result', type=Path)
    args = parser.parse_args()
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        raise ValueError('This campaign requires exactly two GPUs')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    device = f"cuda:{os.environ['LOCAL_RANK']}"
    hardware = torch.cuda.get_device_properties(device)
    if 'A100' not in hardware.name or hardware.total_memory < 75 * 1024**3:
        raise ValueError('Readiness approval requires an actual A100 with 80 GB memory')
    random.seed(42+rank); np.random.seed(42+rank); torch.manual_seed(42+rank)
    root = args.root
    recipe = json.loads((root/'recipes'/f'{args.recipe}.json').read_text())
    prepared = json.loads((root/'prepared.json').read_text())
    for key, name in [('statistics_sha256','statistics.json'),('split_sha256','split.json'),('checkpoints_sha256','checkpoints.json'),
                      ('implementation_sha256','implementation-lock.json')]:
        if sha(root/name) != prepared[key]: raise ValueError('Frozen preparation changed')
    implementation=json.loads((root/'implementation-lock.json').read_text())
    for path, expected in implementation['files'].items():
        if sha(path)!=expected:raise ValueError('Pinned model implementation changed')
    source = {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}
    out = root/'runs'/args.recipe
    out.mkdir(parents=True, exist_ok=True)
    approval_file = root/'profiles'/f'{args.recipe}.json'
    approval = json.loads(approval_file.read_text()) if args.command=='train' else None
    runtime = (approval['runtime'] if approval else json.loads(args.runtime.read_text()))
    runtime={**runtime,'autocast_cache_enabled':False,'allocator':'expandable_segments:True'}
    if os.environ.get('PYTORCH_CUDA_ALLOC_CONF')!=runtime['allocator']:
        raise ValueError('Wrong frozen allocator runtime')
    validate_runtime(runtime,recipe['global_batch'],world)
    torch.set_num_threads(runtime['main_threads'])
    identity = fingerprint({'recipe':recipe,'prepared':prepared,'source':source,
        'hardware':hardware.name,'world':world,'torch':torch.__version__,'runtime':runtime})
    if approval and (not approval.get('approved') or approval['identity']!=identity):
        raise ValueError('No matching successful readiness approval')
    micro = runtime['microbatch']
    train = DOM(root, max_delay=recipe['max_delay'], training=True)
    val = DOM(root, split='validation', training=False)
    if recipe['method']=='pi05':
        from test_optimizer import verify_fused
        verify_fused(device)
        from test_autocast import verify_autocast
        verify_autocast(device)
    policy = Policy(root, recipe, device)
    model = DDP(policy, device_ids=[int(os.environ['LOCAL_RANK'])], find_unused_parameters=True,gradient_as_bucket_view=True)
    torch.cuda.empty_cache()
    optimizer, scheduler = optimizer_for(policy, recipe)
    accumulation = recipe['global_batch']//(micro*world)
    progress = {'updates':0, 'samples':0, 'epoch':0, 'cursor':0}
    if args.command in ('profile','benchmark'):
        # Compare decoded effective-batch updates, then prove the selected
        # runtime including coordinated saves and strict state/RNG reload.
        resources=Resources(gpu=rank==0);resources.__enter__()
        iterator = iter(loader(train,micro,recipe['global_batch'],rank,world,workers=runtime['workers'],prefetch=runtime['prefetch']))
        before = next(p for p in policy.parameters() if p.requires_grad).detach().clone()
        if args.command=='benchmark':
            for _ in range(2):update(model,optimizer,scheduler,iterator,accumulation,recipe['clip'])
        torch.cuda.synchronize()
        startup_gpu_peak=torch.cuda.max_memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        measured_updates=8 if args.command=='benchmark' else 20
        for step in range(measured_updates):
            loss = update(model, optimizer, scheduler, iterator, accumulation, recipe['clip'])
            progress['updates'] += 1; progress['samples'] += recipe['global_batch']; progress['cursor'] += recipe['global_batch']
            if args.command=='profile':
                _,due=synchronized_flags(False,step==9 and rank==0,device)
                if due:save(out/'readiness.pt',policy,optimizer,scheduler,progress,identity)
        if torch.equal(before, next(p for p in policy.parameters() if p.requires_grad)):
            raise ValueError('No optimizer weight change')
        del before
        torch.cuda.synchronize()
        seconds = time.monotonic()-started
        if args.command=='benchmark':
            resources.__exit__()
            peak=torch.tensor([torch.cuda.max_memory_reserved(device)],device=device,dtype=torch.float64);dist.all_reduce(peak,op=dist.ReduceOp.MAX)
            rss=torch.tensor([resources.peak_rss],device=device,dtype=torch.float64);dist.all_reduce(rss)
            if rank==0:atomic(args.result,{'runtime':runtime,'updates':measured_updates,'seconds':seconds,'samples_per_second':measured_updates*recipe['global_batch']/seconds,'peak_gpu_reserved_bytes':float(peak[0]),'total_gpu_bytes':hardware.total_memory,'conservative_node_rss_bytes':float(rss[0]),'mean_gpu_utilization_percent':resources.mean_gpu_utilization,'identity':identity})
            dist.barrier();dist.destroy_process_group();return
        checkpoint = out/'readiness.pt'
        save(checkpoint, policy, optimizer, scheduler, progress, identity)
        expected = torch.rand(16, device=device)
        loaded = load(checkpoint, policy, optimizer, scheduler, identity, rank)
        if loaded != progress or not torch.equal(expected, torch.rand(16,device=device)):
            raise ValueError('Checkpoint/RNG restoration failed')
        validation = validate(policy, val, rank, world, 8)
        # Prove inference equality across strict weight reload with identical RNG.
        check_batch = next(iter(torch.utils.data.DataLoader(val,batch_size=1)))
        state = rng_state()
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False): pred = policy.predict(check_batch)
        policy.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)['model'],strict=True)
        restore_rng(state)
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False): again = policy.predict(check_batch)
        if not torch.allclose(pred,again,rtol=1e-4,atol=1e-5):raise ValueError('Reload changed inference')
        peak = torch.tensor([torch.cuda.max_memory_reserved(device)],device=device,dtype=torch.float64)
        dist.all_reduce(peak,op=dist.ReduceOp.MAX)
        gpu_peak = max(float(peak[0]),startup_gpu_peak)
        lifetime_peak=torch.tensor([gpu_peak],device=device,dtype=torch.float64)
        dist.all_reduce(lifetime_peak,op=dist.ReduceOp.MAX)
        gpu_peak=float(lifetime_peak[0])
        headroom = gpu_peak/hardware.total_memory
        metrics = torch.tensor([headroom],device=device); dist.all_reduce(metrics,op=dist.ReduceOp.MAX)
        resources.__exit__()
        rss = resources.peak_rss
        node_rss = torch.tensor([rss],device=device,dtype=torch.float64);dist.all_reduce(node_rss)
        memory_limit = int(os.environ.get('SLURM_MEM_PER_NODE','262144'))*1024**2
        passed = float(metrics[0]) <= .85 and float(node_rss[0]) <= memory_limit*.85
        if rank == 0:
            atomic(approval_file,{'approved':passed,'identity':identity,'hardware':hardware.name,
                'runtime':runtime,'microbatch':micro,'global_batch':recipe['global_batch'],'accumulation':accumulation,
                'decoded_optimizer_updates':20,'seconds':seconds,'samples_per_second':20*recipe['global_batch']/seconds,
                'peak_gpu_reserved_bytes':gpu_peak,'node_process_peak_rss_bytes':float(node_rss[0]),
                'validation_h20_action_mse':validation,'strict_checkpoint_rng_reload':True,
                'strict_inference_reload':True,'coordinated_checkpoint_test':True,'mean_gpu_utilization_percent':resources.mean_gpu_utilization,'source':source})
        if not passed:raise ValueError('Less than15% memory headroom')
        dist.barrier();dist.destroy_process_group();return
    checkpoint = out/'latest.pt'
    if not checkpoint.exists() and (out/'latest.json').exists():
        raise ValueError('Expected production checkpoint is missing; refusing fresh start')
    if checkpoint.exists():
        progress = load(checkpoint, policy, optimizer, scheduler, identity, rank)
    elif (root/'resume'/f'{args.recipe}.json').exists():
        migration=json.loads((root/'resume'/f'{args.recipe}.json').read_text())
        if migration['recipe_fingerprint']!=fingerprint(recipe) or migration['preparation']!=prepared:
            raise ValueError('Parent recipe/data/model/normalizers differ')
        if migration['hardware']!=hardware.name or migration['world_size']!=world:
            raise ValueError('Parent hardware/rank count differs')
        path=Path(migration['checkpoint'])
        if sha(path)!=migration['checkpoint_sha256']:
            raise ValueError('Frozen parent checkpoint changed')
        progress=load(path,policy,optimizer,scheduler,identity,rank,migration['identity'])
        if rank==0:atomic(out/'resume-migration.json',{'parent':migration,
            'new_identity':identity,'restored_progress':progress,
            'optimizer_scheduler_rng_restored':True,
            'microbatch_changed':True,'data_cursor_preserved':True})
    # Logging initialization must not perturb a restored flow-time/noise stream.
    training_rng=rng_state()
    stop = [False]
    signal.signal(signal.SIGUSR1, lambda *_:stop.__setitem__(0,True))
    signal.signal(signal.SIGTERM, lambda *_:stop.__setitem__(0,True))
    writer = None
    if rank == 0:
        atomic(out/'provenance.json',{'recipe':recipe,'identity':identity,'source':source,
            'hardware':hardware.name,'job_id':os.environ.get('SLURM_JOB_ID'),'world_size':world,
            'readiness':approval,'preparation':prepared})
        try:
            import wandb
            writer=wandb.init(project='Kinematic-Residual-VLA',entity='aparame-clemson-university',
                name=args.recipe+'_a100',dir=str(out),config=recipe,resume='allow',
                id=(json.loads((out/'wandb.json').read_text())['id'] if (out/'wandb.json').exists() else None))
            atomic(out/'wandb.json',{'id':writer.id,'url':writer.url})
            provenance=json.loads((out/'provenance.json').read_text())
            provenance.update(wandb_id=writer.id,wandb_url=writer.url)
            atomic(out/'provenance.json',provenance)
        except Exception as e:
            atomic(out/'wandb-error.json',{'error':type(e).__name__})
    restore_rng(training_rng)
    last_save, last_log = time.monotonic(),time.monotonic()
    while progress['updates'] < recipe['steps']:
        batches=loader(train,micro,recipe['global_batch'],rank,world,progress['epoch'],progress['cursor'],workers=runtime['workers'],prefetch=runtime['prefetch'])
        iterator = iter(batches)
        for _ in range(len(batches)//accumulation):
            loss = update(model,optimizer,scheduler,iterator,accumulation,recipe['clip'])
            progress['updates']+=1;progress['samples']+=recipe['global_batch'];progress['cursor']+=recipe['global_batch']
            if progress['updates']%10==0 and rank==0:
                now=time.monotonic()
                metrics={'Train/loss':float(loss),'Train/optimizer_updates':progress['updates'],'Train/samples':progress['samples'],
                    'Train/epoch':progress['epoch']+progress['cursor']/len(train),
                    'Train/samples_per_second':10*recipe['global_batch']/(now-last_log),'Train/lr':scheduler.get_last_lr()[0]}
                last_log=now;print(json.dumps(metrics),flush=True)
                if writer:writer.log(metrics,step=progress['updates'])
            stopping,checkpoint_due=synchronized_flags(stop[0],time.monotonic()-last_save>1800 or progress['updates']==recipe['steps'],device)
            if stopping or checkpoint_due:
                save(checkpoint,policy,optimizer,scheduler,progress,identity);last_save=time.monotonic()
            if stopping:
                if rank==0:atomic(out/'stopped.json',{'progress':progress,'reason':'allocation_signal'})
                dist.barrier()
                dist.destroy_process_group();return 75
            if progress['updates']==recipe['steps']:break
        if progress['updates'] < recipe['steps']:
            progress['epoch']+=1;progress['cursor']=0
            mse=validate(policy,val,rank,world,256)
            if rank==0:atomic(out/f"validation-{progress['updates']}.json",{'action_mse':mse,'progress':progress})
            save(checkpoint,policy,optimizer,scheduler,progress,identity)
    mse=validate(policy,val,rank,world,256)
    if rank==0:
        atomic(out/'completed.json',{'progress':progress,'validation_h20_action_mse':mse,'stop_reason':'official_step_budget'})
        atomic(out/'training_complete.json',{'progress':progress,'validation_h20_action_mse':mse,'stop_reason':'official_step_budget'})
        if writer:writer.finish()
    dist.destroy_process_group()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
