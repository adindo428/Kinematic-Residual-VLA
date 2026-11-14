"""Prepare independent original-DOM initialization on every released real task."""
import argparse
from collections import Counter,defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from real_robot.dom_dataset import jsonl_read,json_write


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def prepare(output):
    output.mkdir(parents=True,exist_ok=False)
    parent=PROJECT/'scratch/real_robot/finetuning-runs/static-continuation-constant-20261112'
    c=json.loads((parent/'dom-contract.json').read_text())
    original=json.loads((PROJECT/'scratch/real_robot/robust-finetuning-v4-grasp-common-dom/photometric.json').read_text())
    preference=original['checkpoint_preferences'][0]
    if preference['source']['repo']!='hzxie/dynamic-vla-DOM' or preference['source']['real_robot_finetuning']:raise ValueError('Original upstream base required')
    for name,expected in preference['sha256'].items():
        if sha(Path(preference['path'])/name)!=expected:raise ValueError('Original base checksum mismatch')
    data=Path(c['local_verified_dataset']);latest=json.loads((PROJECT/'scratch/real_robot/training_releases/latest.json').read_text())
    episodes=jsonl_read(data/'meta/episodes.jsonl');sources=jsonl_read(data/'meta/sources.jsonl')
    released=jsonl_read(Path(latest['dataset_path'])/'meta/episodes.jsonl')
    if len(episodes)!=len(released):raise ValueError('Prepared data does not cover the latest release')
    splits=json.loads((data/'meta/splits.json').read_text());quality=json.loads((data/'meta/quality_policy.json').read_text())
    if sorted(x for ids in splits.values() for x in ids)!=list(range(len(episodes))):raise ValueError('Splits must partition all episodes')
    source_keys=[row['source_episode'] for row in sources]
    if len(set(source_keys))!=len(source_keys):raise ValueError('Duplicate source episode across splits')
    keys=[];groups=Counter();validation_buckets=defaultdict(list);val_index=0;parquet_hashes={};tasks=Counter();timing=[]
    for split,ids in splits.items():
        for ep in ids:
            record=episodes[ep];task=json.loads(record['tasks'][0]);tasks[(split,task['objects'])]+=1
            if not task.get('instruction'):raise ValueError('Recorded prompt missing')
            path=data/f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet'
            parquet_hashes[str(ep)]=sha(path)
            table=pq.read_table(path,columns=['timestamp','observation.state','observation.environment_state','action'])
            arrays={key:np.asarray(table[key].to_pylist()) for key in table.column_names}
            if any(not np.isfinite(array).all() for array in arrays.values()):raise ValueError(f'Nonfinite labels: {ep}')
            timestamps=arrays['timestamp'].reshape(-1)
            if len(timestamps)!=record['length'] or np.any(np.diff(timestamps)<=0):raise ValueError(f'Bad episode clock: {ep}')
            timing.extend(np.diff(timestamps).tolist())
            rows=jsonl_read(data/f'quality/episode_{ep:06d}.jsonl');usable=set(quality['usable_frames'][str(ep)])
            if len(rows)!=len(timestamps):raise ValueError('Quality length mismatch')
            if split=='test':continue
            for frame,row in enumerate(rows):
                if frame not in usable or max(0,frame-2) not in usable:continue
                key=(task['objects'],bool(sources[ep].get('static_object_collection')),row.get('manipulation_phase','unknown'))
                if split=='train':keys.append(key);groups[key]+=1
                else:validation_buckets[key].append(val_index);val_index+=1
    weights=np.asarray([1/np.sqrt(groups[key]) for key in keys]);weights=np.clip(weights/weights.mean(),.25,4.)
    weights=np.asarray([min(w,1.) if key[-1]=='unknown' else w for key,w in zip(keys,weights)])
    np.save(output/'sampling_weights.npy',weights)
    rng=np.random.default_rng(12345)
    for bucket in validation_buckets.values():rng.shuffle(bucket)
    selected=[]
    while len(selected)<min(512,val_index):
        for bucket in validation_buckets.values():
            if bucket and len(selected)<512:selected.append(bucket.pop())
    for key in ['source_hashes','checkpoint_smoke_reports','checkpoint_architecture_verification','audit_reports','parent_run','batch_profile','checkpoint_selection_reason','profiling_note']:c.pop(key,None)
    c.update(recipe_version='fresh-dom-posttraining-v1',families=['krv100','domv'],training_started=False,status='prepared',
        initialization='Independent strict loading of original hzxie/dynamic-vla-DOM; no local fine-tunes, optimizer or EMA state',
        checkpoint_preferences=[preference],checkpoint={f:preference['checkpoint_id'] for f in ['krv100','domv']},
        instruction_mode='recorded',microbatch=8,gradient_accumulation=8,effective_batch_size=64,
        num_workers=4,prefetch_factor=2,decoder_cache_bytes=512*1024**2,persistent_workers=True,
        learning_rate=3e-5,backbone_learning_rate=3e-6,weight_decay=1e-10,optimizer_eps=1e-8,
        max_epochs=60,min_epochs=15,early_stopping_patience=12,ema_decay=.999,
        scheduler=dict(type='warmup_cosine',warmup_steps=1000,min_lr_ratio=.1),
        sampling_weights=str((output/'sampling_weights.npy').resolve()),sampling_weights_sha256=sha(output/'sampling_weights.npy'),
        action_validation_indices=sorted(selected),checkpoint_keep_last=2,validation_flow_max_batches=None,
        dataset_quality_policy_sha256=sha(data/'meta/quality_policy.json'))
    c['methods']['krv100']['residual_phase_gate_mode']='causal_rules'
    c['methods']['krv100']['residual_phase_gate_config']={}
    c['action_objective'].update(early_steps=4,early_weight=1.,auxiliary_weight=0.,validation_anchors=512,
        selection='Mean physical XYZ/SO(3)/gripper score over identical fixed validation anchors and three noise seeds; compare raw and EMA; not best-of-N')
    c['training_objective']='Native flow matching, uniform valid horizon, zero extra sampled reconstruction loss, as in upstream post-training'
    c['execution_policy']=dict(velocity_innovation={'krv100':'on','domv':'on'},continuous_inference=True,latency_aware_action_streaming=True,
        innovation_is_training_free=True,gate_input='Causal current/past observations and previous applied gripper command only')
    c['augmentation']['photometric']=dict(enabled=True,brightness=[.8,1.2],contrast=[.8,1.2],saturation=[.8,1.2],shared_across_history_and_cameras=True)
    c['augmentation']['proprioception']['enabled']=False
    c['wandb']['run_names']={f:'fresh-full-real-causal-'+f for f in c['families']}
    c['reference_sources']=dict(paper='https://arxiv.org/html/2601.22153v1#A2',
        code='https://github.com/hzxie/DynamicVLA/blob/master/core/train.py',
        config='https://github.com/hzxie/DynamicVLA/blob/master/configs/dynamicvla.yaml',
        departures='Smaller effective batch and discriminative LR for one 24GB GPU; EMA and held-out physical selection; capped balancing; stronger validation; conservative saturation; no geometric image augmentation')
    json_write(output/'launch_contract.json',c)
    audit=dict(episodes=len(episodes),split_counts={k:len(v) for k,v in splits.items()},train_anchors=len(keys),validation_anchors=val_index,
        task_episodes=[dict(split=s,task=t,count=n) for (s,t),n in sorted(tasks.items())],
        duplicate_source_episodes=0,finite_labels=True,monotonic_clocks=True,frame_interval_quantiles_s=np.quantile(timing,[0,.5,.95,1]).tolist(),
        original_base=preference,historical_holdouts_preserved=True,test_used_for_training_or_selection=False,
        robot_motion_started=False,previous_local_finetunes_used=False)
    json_write(output/'data_audit.json',audit);json_write(output/'parquet_hashes.json',parquet_hashes)
    print(json.dumps(audit,indent=2))


def prepare_phases(output):
    sys.path.insert(0,str(PROJECT/'third_party/DynamicVLA'))
    from proactive_vla.phase_gate import replay_prefix_states,PHASES
    contract=json.loads((output/'launch_contract.json').read_text());data=Path(contract['local_verified_dataset'])
    splits=json.loads((data/'meta/splits.json').read_text());folder=output/'phase_cache';folder.mkdir(exist_ok=True)
    hashes={};counts=Counter()
    for ep in splits['train']+splits['validation']:
        path=data/f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet'
        if sha(path)!=json.loads((output/'parquet_hashes.json').read_text())[str(ep)]:raise ValueError('Changed labels before gate replay')
        table=pq.read_table(path,columns=['timestamp','observation.state','observation.environment_state','action'])
        values={key:np.asarray(table[key].to_pylist()) for key in table.column_names}
        state,phases=replay_prefix_states(values['observation.state'],values['observation.environment_state'],values['action'],values['timestamp'].reshape(-1))
        target=folder/f'{ep:06d}.npz';np.savez(target,state=state,phase=phases);hashes[str(ep)]=sha(target)
        counts.update(PHASES[int(i)] for i in phases)
    json_write(output/'phase_hashes.json',hashes);json_write(output/'phase_audit.json',dict(episodes=len(hashes),phase_frames=dict(counts),
        uses_current_or_future_commands=False,uses_future_observations=False,source='Same CausalPhaseGate implementation used by live controller'))
    contract['phase_cache']=str(folder.resolve());json_write(output/'launch_contract.json',contract)
    print(json.dumps(dict(phase_cache_episodes=len(hashes),phase_frames=dict(counts))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase-cache-only',action='store_true')
    a=p.parse_args()
    if a.phase_cache_only:prepare_phases(a.output.resolve())
    else:prepare(a.output.resolve())
