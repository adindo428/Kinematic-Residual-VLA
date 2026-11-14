#!/usr/bin/env python3
"""Freeze the committed 20k split, then continue local KRV100 after epoch two.

This is a dataset migration of the existing run, not a new 20k-only experiment.
The old family stays immutable. Only the subset sampler and validation selection
restart; weights, Adam state, original LR schedule and rank RNG are inherited.
"""
import argparse
import copy
from datetime import datetime, timezone, timedelta
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def checked_digest(value):
    from utils.dom_experiment import digest
    body = dict(value)
    expected = body.pop('fingerprint')
    if digest(body) != expected:
        raise ValueError('Fingerprint mismatch')
    return value


def freeze(args):
    import numpy as np
    from core.dom_training import load_family
    from utils.dom_experiment import atomic_json, digest, file_digest
    old, stage = args.parent.resolve(), args.stage.resolve()
    out = stage/'family'
    if (stage/'plan.json').exists():
        verify(stage)
        print('Existing frozen 20k continuation reused', flush=True)
        return
    prior = load_family(old/'family')
    origin = read(stage/'github-source/github-provenance.json')
    for name, expected in origin['files_sha256'].items():
        if file_digest(stage/'github-source'/name) != expected:
            raise ValueError(f'Committed artifact changed: {name}')
    committed = stage/'github-source/experiments/main8_20k_10epochs'
    split_path = committed/'split.json'
    split = read(split_path)['splits']
    selected = read(committed/'selected-episodes.json')
    records = read(old/'family/episodes.json')
    by_id = {r['episode_index']:r for r in records}
    memberships = {k:set(v) for k,v in split.items()}
    if set(split) != {'train','validation','test'} or len(split['train']) != 20000:
        raise ValueError('Expected the exact 20k train/validation/test split')
    owners, groups = {}, {}
    for part, ids in split.items():
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate episode ID inside partition')
        for ep in ids:
            if ep in owners or ep not in by_id:
                raise ValueError(f'Overlapping or unavailable episode: {ep}')
            owners[ep] = part
            group = by_id[ep]['duplicate_id']
            if groups.setdefault(group,part) != part:
                raise ValueError('Duplicate demonstration group crosses partitions')
    if {r['episode_index'] for r in selected} != memberships['train'] or len(selected) != 20000:
        raise ValueError('Committed selected records differ from train IDs')
    for record in selected:
        if record != by_id[record['episode_index']]:
            raise ValueError('Committed metadata differs from locally audited metadata')
    tasks = {task:sum(r['task']==task for r in selected) for task in ('pick','place','long_horizon')}
    if tasks != {'pick':9000,'place':9000,'long_horizon':2000}:
        raise ValueError('Unexpected task quotas')
    counts = {k:len(ids) for k,ids in split.items()}
    frames = {k:sum(by_id[ep]['length'] for ep in ids) for k,ids in split.items()}
    if frames['train'] != 2406773:
        raise ValueError('Unexpected train frame count')
    old_split = read(old/'family/split.json')['splits']
    overlaps = {k:{p:len(ids.intersection(previous)) for p,previous in old_split.items()}
                for k,ids in memberships.items()}
    confirmation = {'confirmed_at_utc':now(),'github':origin,'split_sha256':file_digest(split_path),
        'dataset_revision':prior['dataset_revision'],'episodes':counts,'frames':frames,'task_quotas':tasks,
        'partition_episode_overlap':False,'partition_duplicate_group_overlap':False,
        'all_selected_episodes_locally_audited':True,'overlap_with_prior_workstation_partitions':overlaps,
        'interpretation':'Continuation after two full-data epochs. New validation/test contain prior training exposure; not pristine held-out estimates.',
        'validation_selection':'New fixed 8192 anchors; reset best and stopping comparisons for this cohort only.'}
    atomic_json(stage/'split_confirmation.json',confirmation)
    out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(split_path,out/'split.json')  # Preserve exact committed bytes/order.
    retained = sorted(owners)
    local_records = [by_id[ep] for ep in retained]
    parquet = read(old/'family/parquet_hashes.json')
    phases = read(old/'family/phase_hashes.json')
    video = read(old/'family/video_hashes.json')
    data = Path(prior['dataset_root'])
    info = read(data/'meta/info.json')
    spec = importlib.util.spec_from_file_location('frozen_preparation',args.source/'scripts/prepare_dom100.py')
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    subset_video = {}
    (out/'phase_cache').mkdir(exist_ok=True)
    for ep in retained:
        for path in prep.episode_paths(data,info,ep):
            if not path.is_file() or not path.stat().st_size:
                raise FileNotFoundError(path)
            rel = str(path.relative_to(data))
            if rel in video:
                if path.stat().st_size != video[rel]['bytes']:
                    raise ValueError(f'Cached video size changed: {rel}')
                subset_video[rel] = video[rel]
        cache = old/'family/phase_cache'/f'{ep:06d}.npz'
        if file_digest(cache) != phases[str(ep)]:
            raise ValueError(f'Phase cache changed: {ep}')
        target = out/'phase_cache'/cache.name
        if not target.exists():
            os.link(cache,target)
    coverage = {**prior['coverage'],'retained_episodes':len(retained),
        'retained_frames':sum(frames.values()),'episode_fraction':len(retained)/prior['published_episodes'],
        'frame_fraction':sum(frames.values())/prior['published_frames'],
        'not_selected_valid_episodes':prior['episodes']-len(retained),
        'task_speed_coverage':{'policy':'See exact committed selected-episodes and split records'}}
    selection = {**read(old/'family/dataset_selection.json'),'episode_ids':retained,
        'mode':'committed_20k_train_plus_committed_validation_test',
        'reason':'User requested committed tracking_controller split starting at global epoch 3'}
    artifacts = {
        'episodes.json':local_records,'parquet_hashes.json':{str(e):parquet[str(e)] for e in retained},
        'phase_hashes.json':{str(e):phases[str(e)] for e in retained},'video_hashes.json':subset_video,
        'dataset_selection.json':selection,'dataset_coverage.json':coverage,
        'pilot_episodes.json':prep.balanced_sample(selected,4096),
        'validation_anchors.json':sorted(np.random.default_rng(42).choice(frames['validation'],8192,replace=False).tolist()),
        'validation_cases.json':prep.balanced_sample([by_id[e] for e in split['validation']],64),
        'test_cases.json':prep.balanced_sample([by_id[e] for e in split['test']],256),
        'revision_provenance.json':{'parent_family':str(old/'family'),'parent_manifest':prior['fingerprint'],
            'github_commit':origin['commit'],'split_confirmation_sha256':file_digest(stage/'split_confirmation.json'),
            'statistics':'Retain parent velocity normalization and drift centering; no refit for continuation.',
            'cached_raw_hashes':'Inherited audited hashes; frozen loader verifies raw files on access.'}}
    for name,value in artifacts.items():
        atomic_json(out/name,value)
    shutil.copyfile(old/'family/approved_exclusions.json',out/'approved_exclusions.json')
    (out/'initialization').mkdir(exist_ok=True)
    shutil.copyfile(old/'family/initialization/config.json',out/'initialization/config.json')
    weights = out/'initialization/model.safetensors'
    if not weights.exists():
        os.link(old/'family/initialization/model.safetensors',weights)
    if file_digest(weights) != prior['weights_sha256']:
        raise ValueError('Initialization weights changed')
    manifest = {**prior,'initialization':str(out/'initialization'),'episodes':len(retained),
        'frames':sum(frames.values()),'train_frames':frames['train'],'counts':counts,'coverage':coverage,
        'dataset_selection_mode':'committed_20k_continuation_after_full_data_epoch2',
        'split_sha256':file_digest(out/'split.json'),
        'parquets_sha256':digest(artifacts['parquet_hashes.json'])}
    manifest.pop('fingerprint')
    manifest['artifacts'] = {name:file_digest(out/name) for name in [*artifacts,'approved_exclusions.json','initialization/config.json']}
    manifest['fingerprint'] = digest(manifest)
    atomic_json(out/'manifest.json',manifest)
    shutil.copyfile(old/'family/evaluation_plan.json',out/'evaluation_plan.json')
    recipe = {**read(old/'family/recipe.json'),'epochs':10,'manifest':manifest['fingerprint'],
        'selection':'User authorized epochs 3 through 10 on exact committed 20k split; retain previous early-stopping rule.',
        'continuation':{'first_global_epoch':3,'subset_epochs_maximum':8,
            'parent_manifest':prior['fingerprint'],'scheduler_total_updates':100*math.ceil(prior['train_frames']/352),
            'sampler_epoch_offset':2,'validation_best_reset':True}}
    recipe.pop('fingerprint')
    recipe['fingerprint'] = digest(recipe)
    atomic_json(out/'recipe.json',recipe)
    controllers = stage/'controller'
    controllers.mkdir(exist_ok=True)
    for name in ('continue_krv100_20k.py','train_krv100.py','training_metadata.py'):
        shutil.copyfile(Path(__file__).with_name(name),controllers/name)
    profile_path = old/'profiling/selected_runtime.json'
    profile = checked_digest(read(profile_path))
    if profile['manifest'] != prior['fingerprint'] or not all(profile[k] for k in ('complete','safe','correctness_checks_passed')):
        raise ValueError('Missing parent runtime approval')
    plan = {'user_authorized':True,'created_at_utc':now(),'source':str(args.source.resolve()),
        'parent':str(old),'stage':str(stage),'family':str(out),'manifest':manifest['fingerprint'],
        'recipe_fingerprint':recipe['fingerprint'],
        'fingerprint':digest({'manifest':manifest['fingerprint'],'recipe':recipe,'variant':'krv100','world_size':2}),
        'parent_manifest':prior['fingerprint'],
        'parent_fingerprint':read(old/'family/runs/krv100/provenance.json')['fingerprint'],
        'parent_service':'krvla-krv100-after-eval-20261010.service',
        'parent_epoch':2,'maximum_total_epochs':10,'wandb_id':'2qh0ijb1','github_commit':origin['commit'],
        'split_sha256':manifest['split_sha256'],'runtime':profile['runtime'],
        'parent_performance_profile':str(profile_path),'parent_performance_sha256':file_digest(profile_path),
        'performance_provenance':'Reuse parent measurements with same numerical model, batch and hardware; no new throughput measurement claimed.',
        'scheduler_total_updates':recipe['continuation']['scheduler_total_updates'],
        'controllers_sha256':{p.name:file_digest(p) for p in controllers.iterdir()},
        'split_confirmation_sha256':file_digest(stage/'split_confirmation.json')}
    atomic_json(stage/'plan.json',plan)
    verify(stage)
    request_path = old/'dataset_transition_request.json'
    request = read(request_path)
    request.update(new_split_confirmed=True,phase='20k_split_confirmed_waiting_for_epoch_two',
                   github_commit=origin['commit'],split_sha256=manifest['split_sha256'],
                   next_stage=str(stage),plan_sha256=file_digest(stage/'plan.json'))
    request.pop('required_input',None)
    atomic_json(request_path,request)
    print(json.dumps({'phase':'20k_family_frozen','manifest':manifest['fingerprint'],
                      'counts':counts,'frames':frames,'wandb_id':plan['wandb_id']}),flush=True)


def verify(stage):
    from core.dom_training import load_family
    from utils.dom_experiment import digest,file_digest
    stage = Path(stage)
    plan = read(stage/'plan.json')
    for name,expected in plan['controllers_sha256'].items():
        if file_digest(stage/'controller'/name) != expected:
            raise ValueError('Frozen continuation controller changed')
    if file_digest(stage/'split_confirmation.json') != plan['split_confirmation_sha256']:
        raise ValueError('Split confirmation changed')
    manifest = load_family(plan['family'])
    recipe = checked_digest(read(Path(plan['family'])/'recipe.json'))
    if (manifest['fingerprint'] != plan['manifest'] or recipe['manifest'] != plan['manifest']
            or recipe['fingerprint'] != plan['recipe_fingerprint'] or recipe['epochs'] != 10
            or manifest['counts']['train'] != 20000 or manifest['split_sha256'] != plan['split_sha256']
            or digest({'manifest':manifest['fingerprint'],'recipe':recipe,'variant':'krv100','world_size':2}) != plan['fingerprint']):
        raise ValueError('Continuation plan differs from frozen family')
    if file_digest(plan['parent_performance_profile']) != plan['parent_performance_sha256']:
        raise ValueError('Parent performance record changed')
    parent = load_family(Path(plan['parent'])/'family')
    for key in ('source_sha256','velocity_mean','velocity_std','normalization','weights_sha256',
                'phase_config','prediction_horizon','execution_horizon','dataset_revision'):
        if manifest[key] != parent[key]:
            raise ValueError(f'Continuation changed parent numerical setting: {key}')
    return plan,manifest,recipe


def publish_status(stage,phase,**details):
    from utils.dom_experiment import atomic_json
    value = {'phase':phase,'updated_at_utc':now(),**details}
    atomic_json(Path(stage)/'training_status.json',value)
    return value


def migrate(plan):
    """Publish a new identity with unchanged numerical state at a validated boundary."""
    import torch
    from core.dom_checkpoint import run_lock
    from utils.dom_experiment import atomic_json,file_digest
    parent = Path(plan['parent'])/'family/runs/krv100'
    dest = Path(plan['family'])/'runs/krv100'
    dest.mkdir(parents=True,exist_ok=True)
    with run_lock(parent), run_lock(dest):
        if (dest/'latest.json').exists():
            pointer = read(dest/'latest.json')
            if pointer['fingerprint'] != plan['fingerprint'] or not (dest/'migration.json').exists():
                raise ValueError('Incomplete or unrelated migrated run')
            return
        pointer = read(parent/'epoch-002.json')
        complete = read(parent/'training_complete.json')
        if (pointer != read(parent/'final.json') or pointer != read(parent/'latest.json')
                or pointer['fingerprint'] != plan['parent_fingerprint']
                or complete['epoch'] != 2 or complete.get('pending_validation') or complete['cursor'] != 0
                or complete['fingerprint'] != plan['parent_fingerprint']
                or not (parent/'validation-epoch-002.json').exists()):
            raise ValueError('Parent is not at the exact validated epoch-two final checkpoint')
        origin = parent/'generations'/pointer['generation']
        state = torch.load(origin/'training_state.pt',map_location='cpu',weights_only=False)
        if (state['progress'] != pointer['progress'] or state['progress']['epoch'] != 2
                or state['fingerprint'] != plan['parent_fingerprint'] or state['wandb_id'] != plan['wandb_id']
                or state['distributed']['world_size'] != 2 or len(state['distributed']['rng_by_rank']) != 2
                or state['scheduler']['last_epoch'] != state['progress']['updates']
                or state['progress']['samples'] != 2*23080253):
            raise ValueError('Parent numerical state or cumulative counters differ')
        historical = {'progress':copy.deepcopy(state['progress']),'best':state['best'],
                      'provenance':state['provenance']}
        progress = state['progress']
        for key in ('completion_reason','user_epoch_limit','epoch_limit_control_sha256','early_stopping'):
            progress.pop(key,None)
        progress.update(epoch=2,cursor=0,pending_validation=False,early_stopped=False,
                        dataset_transition={'first_global_epoch':3,'sampler_epoch_offset':2,
                                            'github_commit':plan['github_commit'],'train_episodes':20000})
        state.update(fingerprint=plan['fingerprint'],best=None)
        provenance = {'fingerprint':plan['fingerprint'],'wandb_id':plan['wandb_id'],
            'manifest':plan['manifest'],'variant':'krv100','world_size':2,'backend':'DDP/NCCL',
            'runtime':plan['runtime'],'recipe':read(Path(plan['family'])/'recipe.json'),
            'continuation_parent':str(origin),'github_split_commit':plan['github_commit']}
        state['provenance'] = provenance
        receipt = {'created_at_utc':now(),'parent_generation':str(origin),
            'parent_files_sha256':{p.name:file_digest(p) for p in origin.iterdir() if p.is_file()},
            'parent_history':historical,'new_fingerprint':plan['fingerprint'],'new_progress':progress,
            'scheduler_total_updates':plan['scheduler_total_updates'],
            'preserved':['model weights','optimizer including Adam moments and LR','scheduler state',
                         'per-rank RNG','global update/sample counters','action/velocity normalization','drift centering'],
            'reset':['subset sampler cursor at boundary','best checkpoint selection for new validation cohort',
                     'early-stopping comparisons for new validation cohort'],
            'prior_exposure':'Two full-data epochs. Validation/test membership changed; prior training exposure retained.'}
        generations = dest/'generations'
        generations.mkdir(exist_ok=True)
        name = f"u{progress['updates']:09d}-subset-{uuid.uuid4().hex[:8]}"
        temp = generations/('.tmp-'+name)
        temp.mkdir()
        os.link(origin/'model.safetensors',temp/'model.safetensors')
        config = read(origin/'config.json')
        config.update(experiment_fingerprint=plan['fingerprint'],completed_epochs=2,epoch=2,
                      completion_reason=None,early_stopping=None)
        atomic_json(temp/'config.json',config)
        atomic_json(temp/'provenance.json',provenance)
        torch.save(state,temp/'training_state.pt')
        for path in temp.iterdir():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
        temp.rename(generations/name)
        receipt['migrated_generation'] = str(generations/name)
        receipt['new_files_sha256'] = {p.name:file_digest(p) for p in (generations/name).iterdir()}
        if receipt['new_files_sha256']['model.safetensors'] != receipt['parent_files_sha256']['model.safetensors']:
            raise ValueError('Migration changed model weights')
        atomic_json(dest/'migration.json',receipt)
        atomic_json(dest/'provenance.json',provenance)
        atomic_json(dest/'latest.json',{'generation':name,'progress':progress,'fingerprint':plan['fingerprint']})


def wait_and_train(args):
    from core.dom_checkpoint import run_lock
    from utils.dom_experiment import atomic_json
    plan,_,_ = verify(args.stage)
    stage, parent = Path(plan['stage']),Path(plan['parent'])
    with run_lock(stage/'coordinator'):
        last_phase = None
        while True:
            verify(stage)
            parent_status = read(parent/'training_status.json')
            active = subprocess.run(['systemctl','--user','is-active',plan['parent_service']],
                                    text=True,capture_output=True,check=False).stdout.strip()
            if parent_status['phase'] == 'failed' or active == 'failed':
                raise RuntimeError('Epoch-two trainer failed; retaining queued split without starting a second trainer')
            if active not in ('active','activating','deactivating') and parent_status['phase']=='complete':
                break
            phase = 'waiting_for_validated_epoch_two'
            publish_status(stage,phase,parent_service=active,wandb_id=plan['wandb_id'],
                           maximum_total_epochs=10,train_episodes=20000)
            if last_phase != phase:
                print(json.dumps(read(stage/'training_status.json')),flush=True)
                last_phase = phase
            time.sleep(30)
        # Systemd being inactive plus the run lock establishes single ownership.
        # Also wait for residual CUDA cleanup or unrelated GPU jobs to finish.
        while True:
            check = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                                   text=True,capture_output=True,check=True)
            if not check.stdout.strip():
                break
            publish_status(stage,'waiting_for_gpu_cleanup',wandb_id=plan['wandb_id'])
            time.sleep(30)
        publish_status(stage,'migrating_epoch_two_checkpoint',wandb_id=plan['wandb_id'])
        migrate(plan)
        request_path = parent/'dataset_transition_request.json'
        request = read(request_path)
        request.update(phase='20k_training_active',activated_at_utc=now(),new_split_confirmed=True)
        atomic_json(request_path,request)
        publish_status(stage,'training',family_root=plan['family'],wandb_id=plan['wandb_id'],
                       first_global_epoch=3,maximum_total_epochs=10,train_episodes=20000)
        command = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                   stage/'controller/continue_krv100_20k.py','train','--source',plan['source'],'--stage',stage]
        with (stage/'training.log').open('a') as log:
            code = subprocess.run([str(x) for x in command],cwd=plan['source'],
                                  stdout=log,stderr=subprocess.STDOUT,check=False).returncode
        if code:
            raise RuntimeError(f'20k training exited {code}; inspect {stage}/training.log')
        complete_path = Path(plan['family'])/'runs/krv100/training_complete.json'
        if not complete_path.exists():
            raise RuntimeError('20k trainer exited without a completion record')
        complete = read(complete_path)
        if complete['epoch'] > 10 or complete['fingerprint'] != plan['fingerprint']:
            raise ValueError('Invalid 20k completion')
        publish_status(stage,'complete',completed_epochs=complete['epoch'],wandb_id=plan['wandb_id'])
        request.update(phase='20k_training_complete',completed_at_utc=now(),completed_epochs=complete['epoch'])
        atomic_json(request_path,request)


def train(args):
    import torch
    import torch.distributed as dist
    import core.dom_distributed_training as training
    from utils.dom_experiment import digest,file_digest
    plan,manifest,recipe = verify(args.stage)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    world,local = int(os.environ['WORLD_SIZE']),int(os.environ['LOCAL_RANK'])
    if world != 2 or int(os.environ['LOCAL_WORLD_SIZE']) != 2:
        raise ValueError('Continuation requires the original two ranks')
    torch.set_num_threads(1)
    torch.cuda.set_device(local)
    device = torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    props = torch.cuda.get_device_properties(device)
    hardware = {'name':props.name,'total_memory':props.total_memory,'torch':torch.__version__,
                'cuda':torch.version.cuda,'capability':[props.major,props.minor]}
    actual = [None]*world
    dist.all_gather_object(actual,hardware)
    profile = checked_digest(read(plan['parent_performance_profile']))
    metadata_sha = file_digest(Path(__file__).with_name('training_metadata.py'))
    if (profile['hardware_by_rank'] != actual or profile['runtime'] != plan['runtime']
            or profile['manifest'] != plan['parent_manifest'] or profile['variant'] != 'krv100'
            or profile['world_size'] != 2 or not all(profile[k] for k in ('complete','safe','correctness_checks_passed'))
            or profile['execution_controller']['metadata_adapter_sha256'] != metadata_sha):
        raise ValueError('Parent runtime approval does not transfer to this hardware/model configuration')
    key = digest({'hardware':actual[0],'world_size':world,'runtime':plan['runtime'],'variant':'krv100'})[:16]
    readiness = checked_digest(read(Path(plan['parent'])/'family/distributed-profiles'/f'{key}.json'))
    if (readiness['hardware_by_rank'] != actual or readiness['manifest'] != plan['parent_manifest']
            or readiness['runtime'] != plan['runtime'] or not all(readiness[k] for k in
                ('complete','safe','strict_reload','rank_rng_reload','future_inputs_ignored'))):
        raise ValueError('Parent distributed readiness is inconsistent')
    destination = Path(plan['family'])/'runs/krv100'
    receipt = read(destination/'migration.json')
    # The parent epoch-two generation is pinned. The migrated initial generation
    # can be garbage-collected after new periodic checkpoints are published.
    expected_config = read(Path(receipt['parent_generation'])/'config.json')
    original_scheduler = training.make_scheduler
    training.make_scheduler = lambda optimizer,total_updates,warmup=1000: original_scheduler(
        optimizer,plan['scheduler_total_updates'],warmup)
    original_wrap = training.wrap_policy
    def wrap(policy,device):
        policy.config.scheduler_decay_steps = plan['scheduler_total_updates']
        for name in ('velocity_input_mean','velocity_input_std','residual_drift_mean',
                     'residual_flow_formulation','residual_center_drift','residual_normalize_drift',
                     'condition_on_object_velocity','enable_residual_flow','use_checkpoint_normalization'):
            actual_value = getattr(policy.config,name)
            if json.loads(json.dumps(actual_value)) != expected_config[name]:
                raise ValueError(f'Continuation changed learned action/velocity semantics: {name}')
        return original_wrap(policy,device)
    training.wrap_policy = wrap
    original_loader = training.distributed_loader
    def loader(data,runtime,epoch=0,cursor=0,sampling='frames',lengths=None):
        if epoch < 2 or epoch >= 10:
            raise ValueError('Subset sampler requested outside global epochs 3 through 10')
        return original_loader(data,runtime,epoch-2,cursor,sampling,lengths)
    training.distributed_loader = loader
    original_load = training.load_generation
    def load(*args,**kwargs):
        state = original_load(*args,**kwargs)
        if state['wandb_id'] != plan['wandb_id'] or state['progress']['epoch'] < 2:
            raise ValueError('Lost original W&B run or global epoch accounting')
        return state
    training.load_generation = load
    from train_krv100 import install_checkpoint_cadence
    from training_metadata import install_metadata_compaction
    install_checkpoint_cadence(training,destination)
    install_metadata_compaction(training)
    import wandb
    if not callable(getattr(wandb.util,'generate_id',None)):
        from wandb.sdk.lib.runid import generate_id
        wandb.util.generate_id = generate_id
    init = wandb.init
    def init_writer(*args,**kwargs):
        if kwargs.get('id') != plan['wandb_id']:
            raise ValueError('A new W&B run is forbidden for this continuation')
        # Keep original run configuration/history and attach a named stage.
        stage_config = kwargs.pop('config')
        kwargs['resume'] = 'must'
        writer = init(*args,**kwargs)
        writer.config.update({'dataset_transition_epoch3_20k':{
            'github_commit':plan['github_commit'],'split_sha256':plan['split_sha256'],
            'first_global_epoch':3,'maximum_total_epochs':10,'train_episodes':20000,
            'prior_exposure':'Two full-data epochs; new validation/test have prior training exposure.',
            'scheduler_total_updates':plan['scheduler_total_updates'],'new_stage':stage_config}},allow_val_change=True)
        return writer
    wandb.init = init_writer
    logging.info('Continuing same W&B %s, global epochs 3–10, 20k IDs; original scheduler horizon %d',
                 plan['wandb_id'],plan['scheduler_total_updates'])
    result = training.train_distributed(Path(plan['family']),'krv100',plan['runtime'],recipe,device)
    if dist.get_rank()==0:
        print(json.dumps(result),flush=True)
    dist.destroy_process_group()
    return 75 if result.get('interrupted') else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','wait','train'))
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--parent',type=Path)
    args = parser.parse_args()
    sys.path.insert(0,str(args.source.resolve()))
    if args.command=='prepare':
        if args.parent is None:
            parser.error('--parent required for prepare')
        freeze(args)
    elif args.command=='wait':
        try:
            wait_and_train(args)
        except BaseException as error:
            publish_status(args.stage,'failed',error=str(error))
            raise
    else:
        return train(args)
    return 0


if __name__=='__main__':
    try:
        code = main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    sys.exit(code)
