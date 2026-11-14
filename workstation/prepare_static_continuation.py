"""Freeze completed collection for architecture-preserving continuation."""
import argparse
from collections import Counter,defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pyarrow.parquet as pq

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from real_robot.dom_dataset import jsonl_read,jsonl_write,json_write
from real_robot.calibration_acceptance import dataset_acceptance


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def retain(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(source,target)
    except OSError:shutil.copy2(source,target)


def freeze(latest,previous,output):
    if (output/'continuation_dataset.json').exists():return json.loads((output/'continuation_dataset.json').read_text())
    output.mkdir(parents=True,exist_ok=False);shutil.copytree(latest/'meta',output/'meta')
    (output/'videos').symlink_to((latest/'videos').resolve(),target_is_directory=True)
    current={k:jsonl_read(latest/f'meta/{k}.jsonl') for k in ['sources','episodes','camera','episodes_stats']}
    old={k:jsonl_read(previous/f'meta/{k}.jsonl') for k in current}
    old_n=len(old['episodes']);n=len(current['episodes'])
    assert n>=old_n
    qnew=json.loads((latest/'meta/quality_policy.json').read_text())
    qold=json.loads((previous/'meta/quality_policy.json').read_text())
    splits=json.loads((previous/'meta/splits.json').read_text())
    for ep in range(n):
        name=f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet'
        if ep<old_n:
            assert current['sources'][ep]['source_episode']==old['sources'][ep]['source_episode']
            assert current['episodes'][ep]==old['episodes'][ep]
            for key in ['action','observation.state']:
                a=np.asarray(pq.read_table(latest/name,columns=[key])[key].to_pylist())
                b=np.asarray(pq.read_table(previous/name,columns=[key])[key].to_pylist())
                if not np.array_equal(a,b):raise ValueError(f'Expert robot labels changed: episode {ep} {key}')
            source=previous
            for key in current:current[key][ep]=copy.deepcopy(old[key][ep])
            qnew['usable_frames'][str(ep)]=qold['usable_frames'][str(ep)]
        else:
            source=latest;splits[current['sources'][ep]['split']].append(ep)
        retain(source/name,output/name)
        retain(source/f'quality/episode_{ep:06d}.jsonl',output/f'quality/episode_{ep:06d}.jsonl')
    assert sorted(ep for group in splits.values() for ep in group)==list(range(n))
    for split,ids in splits.items():
        for ep in ids:current['sources'][ep]['split']=split
    for key,value in current.items():jsonl_write(output/f'meta/{key}.jsonl',value)
    info=json.loads((latest/'meta/info.json').read_text());info['splits']=splits
    json_write(output/'meta/info.json',info);json_write(output/'meta/splits.json',splits)
    qnew['base_calibration_acceptance']=dataset_acceptance(output,current['camera'])
    json_write(output/'meta/quality_policy.json',qnew)
    from lerobot.common.datasets.compute_stats import aggregate_stats
    agg=aggregate_stats([{k:{name:np.asarray(v) for name,v in values.items()} for k,values in current['episodes_stats'][ep]['stats'].items()} for ep in splits['train']])
    json_write(output/'meta/train_stats.json',{k:{name:v.tolist() for name,v in values.items()} for k,values in agg.items()})
    report=dict(episodes=n,previous_episodes=old_n,new_episodes=n-old_n,
        split_counts={k:len(v) for k,v in splits.items()},previous=str(previous),latest=str(latest),
        old_expert_labels_verified_unchanged=True,old_processed_geometry_and_held_labels_retained=True,
        historical_holdouts_preserved=True,test_used_for_training=False)
    json_write(output/'continuation_dataset.json',report);return report


def prepare(output):
    output.mkdir(parents=True,exist_ok=True)
    old_run=PROJECT/'scratch/real_robot/finetuning-runs/action-v5-fullsampler-20261110'
    old_contract=json.loads((old_run/'launch_contract.json').read_text())
    parent=Path(old_contract['local_verified_dataset'])
    latest=json.loads((PROJECT/'scratch/real_robot/training_releases/latest.json').read_text())
    data=output/'dataset';report=freeze(Path(latest['dataset_path']),parent,data)
    c=copy.deepcopy(old_contract)
    for key in ['source_hashes','checkpoint_smoke_reports','training_started','checkpoint_architecture_verification','audit_reports']:c.pop(key,None)
    c.update(recipe_version='static-continuation-constant-v1',repo_id='ailab-Clemson/KRVLA-dataset',
        revision=latest['commit'],dataset_parent_revision=latest['commit'],local_verified_dataset=str(data.resolve()),
        dataset_source='Local derivative of pinned latest release; preserves earlier holdouts and processed prefix',
        families=['krv100','domv','dom'],microbatch=16,gradient_accumulation=1,num_workers=4,prefetch_factor=2,
        batched_decode=True,decoder_cache_bytes=512*1024**2,persistent_workers=True,vectorized_photometric=True,
        max_epochs=15,early_stopping_patience=5,learning_rate=3e-6,weight_decay=.01,
        dataset_quality_policy_sha256=sha(data/'meta/quality_policy.json'),training_started=False,
        checkpoint={},checkpoint_preferences=[],comparison_note='Each family continues its own best real-robot checkpoint; architecture unchanged; constant KRV gate; same frozen anchors, splits and physical metrics.')
    c['methods']['krv100']['residual_phase_gate_mode']='constant'
    sources=jsonl_read(data/'meta/sources.jsonl');metadata=jsonl_read(data/'meta/episodes.jsonl')
    splits=json.loads((data/'meta/splits.json').read_text());q=json.loads((data/'meta/quality_policy.json').read_text())
    counts=Counter();keys=[];buckets=defaultdict(list);val_index=0
    for split in ['train','validation']:
        for ep in splits[split]:
            quality=jsonl_read(data/f'quality/episode_{ep:06d}.jsonl');usable=set(q['usable_frames'][str(ep)])
            for i,row in enumerate(quality):
                if i not in usable or max(0,i-2) not in usable:continue
                key=(metadata[ep]['tasks'][0],bool(sources[ep].get('static_object_collection')),row.get('manipulation_phase','unknown'))
                if split=='train':keys.append(key);counts[key]+=1
                else:buckets[key].append(val_index);val_index+=1
    weights=np.array([1/np.sqrt(counts[key]) for key in keys]);weights=np.clip(weights/weights.mean(),.25,4.)
    weights=np.array([min(w,1.) if key[-1]=='unknown' else w for w,key in zip(weights,keys)])
    np.save(output/'sampling_weights.npy',weights)
    c.update(sampling_weights=str((output/'sampling_weights.npy').resolve()),sampling_weights_sha256=sha(output/'sampling_weights.npy'))
    rng=np.random.default_rng(12345)
    for bucket in buckets.values():rng.shuffle(bucket)
    selected=[]
    while len(selected)<min(256,val_index):
        for bucket in buckets.values():
            if bucket and len(selected)<256:selected.append(bucket.pop())
    c['action_validation_indices']=sorted(selected)
    c['wandb']['run_names']={f:'static-continuation-constant-'+f for f in c['families']}
    for family in c['families']:
        selected_checkpoint=json.loads((old_run/family/'best_checkpoint.json').read_text());path=Path(selected_checkpoint['path'])
        identity=f'{family}-best-real-epoch-{selected_checkpoint["epoch"]:03d}'
        c['checkpoint'][family]=identity
        c['checkpoint_preferences'].append(dict(checkpoint_id=identity,path=str(path),sha256={name:sha(path/name) for name in ['config.json','model.safetensors']}))
    json_write(output/'dom-contract.json',c)
    legacy=PROJECT/'scratch/real_robot/family-finetuning-v6'
    campaign=copy.deepcopy(json.loads((legacy/'campaign.json').read_text()))
    campaign['entries']=[entry for entry in campaign['entries'] if entry['family'] in ['smolvla_baseline','smolvla_kr']]
    for entry in campaign['entries']:
        best=json.loads((legacy/'runs'/entry['family']/'best_checkpoint.json').read_text())
        entry['continuation_checkpoint']=dict(path=best['path'],sha256=sha(best['path']),epoch=best['epoch'],weights=best['weights'])
    recipe=campaign['training_recipe']
    for key in ['repo_id','revision','local_verified_dataset','dataset_quality_policy_sha256','sampling_weights','sampling_weights_sha256','action_validation_indices','max_epochs','early_stopping_patience','learning_rate','weight_decay']:
        recipe[key]=c[key]
    campaign.update(benchmark_before_training=False,source_hashes={},parent_campaign=str(legacy/'campaign.json'),
        continuation_from_best_real_checkpoints=True,profile_reuse='v6 selected flow=16 auxiliary=4, workers=4, prefetch=2, decoder cache=8, vectorized photometric; readiness rechecked per new checkpoint/data')
    for name in ['hub','dependencies']:
        if not (output/name).exists():(output/name).symlink_to((legacy/name).resolve(),target_is_directory=True)
    shutil.copy2(legacy/'downloads.json',output/'downloads.json')
    json_write(output/'family-campaign.json',campaign)
    json_write(output/'preparation.json',dict(dataset=report,train_anchors=len(weights),validation_anchors=val_index,
        selected_validation_anchors=len(selected),order=['krv100','domv','dom','smolvla_baseline','smolvla_kr'],
        gating='constant',latest_training_commit=latest['commit']))
    print(json.dumps(json.loads((output/'preparation.json').read_text()),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    prepare(p.parse_args().output.resolve())
