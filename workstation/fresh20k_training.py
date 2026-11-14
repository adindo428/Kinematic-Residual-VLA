#!/usr/bin/env python3
"""Prepare and execute matched, fresh 20k fine-tuning after current evaluations."""
import argparse
from datetime import datetime,timezone,timedelta
import json
import logging
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid


def read(path):return json.loads(Path(path).read_text())
def now():return datetime.now(timezone.utc).isoformat()


def prepare(args):
    import numpy as np
    import pyarrow.parquet as pq
    from core.dom_training import load_family
    from utils.dom_experiment import atomic_json,digest,file_digest
    stage=args.stage.resolve();old=args.prior.resolve();family=stage/'family'
    if (stage/'plan.json').exists():
        verify(stage);print('Existing fresh family verified',flush=True);return
    parent=load_family(old/'family');family.mkdir(parents=True,exist_ok=True)
    # Reuse cached immutable data and exact committed split; no trained weights
    # or old optimizer/checkpoint state enter this new family.
    for name in parent['artifacts']:
        src=old/'family'/name;dst=family/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst)
    shutil.copyfile(old/'family/split.json',family/'split.json')
    (family/'phase_cache').mkdir(exist_ok=True)
    for src in (old/'family/phase_cache').iterdir():
        dst=family/'phase_cache'/src.name
        if not dst.exists():os.link(src,dst)
    weights=family/'initialization/model.safetensors'
    if not weights.exists():os.link(old/'family/initialization/model.safetensors',weights)
    if file_digest(weights)!=parent['weights_sha256']:raise ValueError('Released initialization changed')
    split=read(family/'split.json')['splits'];data=Path(parent['dataset_root']);info=read(data/'meta/info.json')
    hashes=read(family/'parquet_hashes.json');sums=np.zeros(3);squares=np.zeros(3);count=0
    for index,ep in enumerate(split['train']):
        path=data/info['data_path'].format(episode_chunk=ep//info['chunks_size'],episode_index=ep)
        if file_digest(path)!=hashes[str(ep)]:raise ValueError(f'Training parquet changed: {ep}')
        values=np.asarray(pq.read_table(path,columns=['observation.environment_state'])[0].to_pylist(),dtype=np.float64)[:,6:9]
        if not np.isfinite(values).all():raise ValueError('Non-finite training velocity')
        sums+=values.sum(0);squares+=np.square(values).sum(0);count+=len(values)
        if index%2500==0:print(json.dumps({'phase':'fitting_20k_velocity_statistics','episodes':index+1,'total':20000}),flush=True)
    if count!=parent['train_frames'] or len(split['train'])!=20000:raise ValueError('Unexpected subset size')
    provenance={'kind':'Fresh matched fine-tuning, not continuation of previous trained checkpoints',
        'data_parent':str(old/'family'),'data_parent_manifest':parent['fingerprint'],
        'split_sha256':file_digest(family/'split.json'),'training_frames_for_velocity_statistics':count,
        'normalization':'Shared released state/action statistics; velocity mean/std and analytic drift centering fit on 20k train only.',
        'prior_exposure':'Common released DynamicVLA initialization was DOM-pretrained. No local full-data epochs in these new runs.'}
    atomic_json(family/'fresh_provenance.json',provenance)
    manifest={**parent,'initialization':str(family/'initialization'),
        'velocity_mean':(sums/count).tolist(),
        'velocity_std':np.sqrt(np.maximum(squares/count-(sums/count)**2,1e-12)).tolist(),
        'dataset_selection_mode':'fresh_finetuning_on_committed_20k_split'}
    manifest.pop('fingerprint')
    manifest['artifacts']={name:file_digest(family/name) for name in [*parent['artifacts'],'fresh_provenance.json']}
    manifest['fingerprint']=digest(manifest);atomic_json(family/'manifest.json',manifest)
    shutil.copyfile(old/'family/evaluation_plan.json',family/'evaluation_plan.json')
    recipe={**read(old/'family/recipe.json'),'manifest':manifest['fingerprint'],'epochs':10,
        'early_stopping':{'enabled':True,'metric':'h20/action_mse','min_epochs':5,'patience':3,'relative_min_delta':0.005},
        'selection':'Fresh matched 20k fine-tuning from common released weights; reset optimizer/scheduler; validation-only selection.'}
    recipe.pop('continuation',None);recipe.pop('fingerprint');recipe['fingerprint']=digest(recipe)
    atomic_json(family/'recipe.json',recipe)
    controller=stage/'controller';controller.mkdir(exist_ok=True)
    for name in ('fresh20k_training.py','training_metadata.py','train_krv100.py'):
        shutil.copyfile(Path(__file__).with_name(name),controller/name)
    plan={'user_authorized':True,'created_at_utc':now(),'stage':str(stage),'family':str(family),
        'source':str(args.source.resolve()),'manifest':manifest['fingerprint'],
        'recipe_fingerprint':recipe['fingerprint'],'variants':args.variants.split(','),
        'wandb_ids':{v:uuid.uuid4().hex[:8] for v in args.variants.split(',')},
        'headroom_fraction':0.10,'memory_cap_fraction':0.90,'effective_batch':352,
        'profile_microbatches':[32,36,40,44],'runtime_base':{'workers':8,'prefetch':2,'pin_memory':True,'fused':True,'precision':'bf16'},
        'host_limit_mib':231673,'dependency_results':[
            str(Path(old).parents[1]/'results/krv100-best-epoch003-task-table1-20261025'),
            str(Path(old).parents[1]/'results/krv100-latest-epoch008p09-task-table1-20261025')],
        'controllers_sha256':{p.name:file_digest(p) for p in controller.iterdir()},
        'scheduler_total_updates':10*math.ceil(count/352)}
    if not set(plan['variants']).issubset({'dom100','domv100','krv100'}) or 'krv100' not in plan['variants']:
        raise ValueError('Unexpected comparison variants')
    for variant in plan['variants']:
        destination=family/'runs'/variant;destination.mkdir(parents=True,exist_ok=True)
        atomic_json(destination/'provenance.json',{'fingerprint':digest({'manifest':manifest['fingerprint'],
            'recipe':recipe,'variant':variant,'world_size':2}),'wandb_id':plan['wandb_ids'][variant],
            'variant':variant,'fresh':True,'initialization_sha256':manifest['weights_sha256']})
    atomic_json(stage/'plan.json',plan);verify(stage)
    atomic_json(stage/'status.json',{'phase':'prepared_waiting_for_existing_evaluations','variants':plan['variants'],
                                    'headroom_fraction':0.10,'updated_at_utc':now()})
    print(json.dumps({'phase':'fresh_family_frozen','manifest':manifest['fingerprint'],
                      'variants':plan['variants'],'scheduler_total_updates':plan['scheduler_total_updates']}),flush=True)


def verify(stage):
    from core.dom_training import load_family
    from utils.dom_experiment import digest,file_digest
    plan=read(Path(stage)/'plan.json');manifest=load_family(plan['family']);recipe=read(Path(plan['family'])/'recipe.json')
    body=dict(recipe);fp=body.pop('fingerprint')
    if (digest(body)!=fp or fp!=plan['recipe_fingerprint'] or manifest['fingerprint']!=plan['manifest']
            or recipe['manifest']!=manifest['fingerprint'] or recipe['epochs']!=10
            or manifest['counts']['train']!=20000):raise ValueError('Fresh family/recipe changed')
    for name,sha in plan['controllers_sha256'].items():
        if file_digest(Path(stage)/'controller'/name)!=sha:raise ValueError('Frozen controller changed')
    return plan,manifest,recipe


def gpu_worker(args):
    import torch
    import torch.distributed as dist
    from utils.dom_experiment import atomic_json,digest
    plan,manifest,recipe=verify(args.stage)
    if int(os.environ['WORLD_SIZE'])!=2:raise ValueError('Two GPUs required')
    local=int(os.environ['LOCAL_RANK']);torch.set_num_threads(1);torch.cuda.set_device(local)
    device=torch.device('cuda',local);dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    from training_metadata import install_metadata_compaction
    if args.command=='profile':
        import core.dom_distributed_profiling as profiling
        install_metadata_compaction(profiling)
        build=profiling.build_policy
        profiling.build_policy=lambda root,variant,device:build(root,args.variant,device)
        result=profiling.profile_trial(plan['family'],json.loads(args.runtime),'end_to_end',device,
                                       args.output,warmup=2,measured_updates=args.updates,epoch=args.epoch)
        result.pop('fingerprint');result['original_15_percent_headroom_safe']=result['safe']
        result.update(variant=args.variant,headroom_fraction=0.10,
            safe=all(r['telemetry']['peak_gpu_bytes']<=.90*r['hardware']['total_memory'] for r in result['ranks'])
                 and sum(r['telemetry']['peak_ram_bytes'] for r in result['ranks'])<=.85*result['host_limit_bytes'])
        result['fingerprint']=digest(result)
        if dist.get_rank()==0:atomic_json(args.output,result)
        dist.barrier();dist.destroy_process_group();return 0
    import core.dom_distributed_training as training
    selection=read(Path(args.stage)/'profiling/selected_runtime.json');body=dict(selection);fp=body.pop('fingerprint')
    if digest(body)!=fp or selection['manifest']!=plan['manifest']:raise ValueError('Runtime selection changed')
    runtime=selection['runtime'];props=torch.cuda.get_device_properties(device)
    actual={'name':props.name,'total_memory':props.total_memory,'torch':torch.__version__,
            'cuda':torch.version.cuda,'capability':[props.major,props.minor]}
    hardware=[None]*2;dist.all_gather_object(hardware,actual)
    for row in selection['confirmation_by_variant'][args.variant]:
        if not row['safe'] or row['hardware_by_rank']!=hardware or row['runtime']!=runtime:
            raise ValueError('Measured runtime does not match this model/hardware')
    destination=Path(plan['family'])/'runs'/args.variant
    install_metadata_compaction(training)
    from train_krv100 import install_checkpoint_cadence
    install_checkpoint_cadence(training,destination)
    original_updates=training.updates
    def updates(*positional,**keywords):
        for record in original_updates(*positional,**keywords):
            free,total=torch.cuda.mem_get_info(device)
            ratio=torch.tensor((total-free)/total,device=device,dtype=torch.float64)
            dist.all_reduce(ratio,op=dist.ReduceOp.MAX)
            if float(ratio)>.90:
                positional[4]['headroom_guard_triggered']=True
                logging.error('Memory headroom fell below 10%%; requesting checkpointed stop')
                os.kill(os.getpid(),signal.SIGUSR1)
            yield record
    training.updates=updates
    import wandb
    if not callable(getattr(wandb.util,'generate_id',None)):
        from wandb.sdk.lib.runid import generate_id
        wandb.util.generate_id=generate_id
    init=wandb.init
    def writer(*positional,**keywords):
        if keywords.get('id')!=plan['wandb_ids'][args.variant] or keywords['id']=='2qh0ijb1':
            raise ValueError('Fresh run must use its new W&B identity')
        keywords.update(name=f'fresh20k-{args.variant}-s42',group='fresh20k-20261025')
        run=init(*positional,**keywords)
        run.config.update({'fresh_20k_plan':plan,'memory_cap_fraction':0.90},allow_val_change=True)
        return run
    wandb.init=writer
    result=training.train_distributed(plan['family'],args.variant,runtime,recipe,device)
    if dist.get_rank()==0:print(json.dumps(result),flush=True)
    dist.destroy_process_group();return 75 if result.get('interrupted') else 0


def coordinate(args):
    from core.dom_checkpoint import run_lock
    from utils.dom_experiment import atomic_json,digest
    plan,manifest,recipe=verify(args.stage);stage=Path(args.stage)
    def status(phase,**details):
        atomic_json(stage/'status.json',{'phase':phase,'updated_at_utc':now(),**details})
    def execute(command,log,timeout=None):
        with log.open('a') as stream:
            child=subprocess.Popen([str(x) for x in command],cwd=plan['source'],
                                   stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=timeout)
            finally:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
            return code
    with run_lock(stage/'coordinator'):
        status('waiting_existing_evaluations',dependencies=plan['dependency_results'])
        while True:
            ready=all((Path(p)/'summary.json').is_file() and read(Path(p)/'summary.json').get('complete')
                      for p in plan['dependency_results'])
            busy=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                                capture_output=True,text=True,check=True).stdout.strip()
            if ready and not busy:break
            time.sleep(30)
        base=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
              stage/'controller/fresh20k_training.py']
        profile_root=stage/'profiling';profile_root.mkdir(exist_ok=True)
        def measure(variant,runtime,label,updates=5,epoch=0):
            folder=profile_root/label;folder.mkdir(exist_ok=True)
            path=folder/'measurement.json'
            if path.exists():
                result=read(path)
                if (result['manifest']!=plan['manifest'] or result['variant']!=variant
                        or result['runtime']!=runtime or result['measured_updates']!=updates):
                    raise ValueError('Cached profile differs from requested measurement')
                return result
            status('profiling',variant=variant,microbatch=runtime['microbatch'],headroom_fraction=0.10)
            code=execute([*base,'profile','--source',plan['source'],'--stage',stage,'--variant',variant,
                '--runtime',json.dumps(runtime),'--output',path,'--updates',updates,'--epoch',epoch],
                folder/'worker.log',timeout=1800)
            if code:
                if 'out of memory' in (folder/'worker.log').read_text().lower():return None
                raise RuntimeError(f'Profiling failed: {folder}/worker.log')
            return read(path)
        if (profile_root/'selected_runtime.json').exists():selection=read(profile_root/'selected_runtime.json')
        else:
            candidates=[]
            for micro in plan['profile_microbatches']:
                runtime={**plan['runtime_base'],'microbatch':micro}
                row=measure('krv100',runtime,f'capacity-m{micro}')
                if row and row['safe']:candidates.append(row)
                else:break
            selection=None
            for candidate in sorted(candidates,key=lambda r:r['samples_per_second'],reverse=True):
                runtime=candidate['runtime'];confirmations={};okay=True
                for variant in plan['variants']:
                    rows=[measure(variant,runtime,f'confirm-{variant}-m{runtime["microbatch"]}-{i}',updates=10,epoch=i)
                          for i in range(2)]
                    confirmations[variant]=rows
                    if not all(row and row['safe'] for row in rows):okay=False;break
                if okay:
                    selection={'manifest':plan['manifest'],'runtime':runtime,'headroom_fraction':0.10,
                        'confirmation_by_variant':confirmations,'capacity_candidates':candidates,
                        'selection':'Fastest measured common runtime satisfying >=10% free VRAM on both GPUs and host budget.'}
                    selection['fingerprint']=digest(selection);atomic_json(profile_root/'selected_runtime.json',selection);break
            if selection is None:raise RuntimeError('No common measured runtime retains 10% GPU memory headroom')
        for variant in plan['variants']:
            destination=Path(plan['family'])/'runs'/variant
            if (destination/'training_complete.json').exists():continue
            status('training',variant=variant,variants=plan['variants'],runtime=selection['runtime'],
                   wandb_id=plan['wandb_ids'][variant])
            code=execute([*base,'train','--source',plan['source'],'--stage',stage,'--variant',variant],
                         stage/f'training-{variant}.log')
            if code or not (destination/'training_complete.json').exists():
                raise RuntimeError(f'{variant} stopped before completion; saved checkpoints retained. Inspect training-{variant}.log')
        status('complete',variants=plan['variants'],wandb_ids=plan['wandb_ids'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','profile','train'))
    parser.add_argument('--stage',type=Path,required=True);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--prior',type=Path);parser.add_argument('--variants',default='dom100,krv100')
    parser.add_argument('--variant',choices=('dom100','domv100','krv100'))
    parser.add_argument('--runtime');parser.add_argument('--output',type=Path)
    parser.add_argument('--updates',type=int,default=5);parser.add_argument('--epoch',type=int,default=0)
    args=parser.parse_args();sys.path.insert(0,str(args.source.resolve()))
    if args.command=='prepare':prepare(args)
    elif args.command=='run':
        try:coordinate(args)
        except BaseException as error:
            from utils.dom_experiment import atomic_json
            atomic_json(args.stage/'status.json',{'phase':'failed','error':str(error),'updated_at_utc':now()});raise
    else:return gpu_worker(args)
    return 0


if __name__=='__main__':
    try:code=main()
    except Exception:
        import traceback
        traceback.print_exc();sys.stderr.flush();os._exit(1)
    sys.exit(code)
