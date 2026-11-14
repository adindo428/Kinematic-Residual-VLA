"""Audit frozen labels/timing and prepare continuation runs; never controls hardware."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def quantiles(values):
    return dict(zip(['min','p50','p95','p99','max'],np.quantile(values,[0,.5,.95,.99,1]).tolist())) if values else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    c=json.loads((a.parent/'dom/run_config.json').read_text());root=Path(c['local_verified_dataset'])
    splits=json.loads((root/'meta/splits.json').read_text())
    quality=json.loads((root/'meta/quality_policy.json').read_text())
    episodes={e['episode_index']:e for e in map(json.loads,(root/'meta/episodes.jsonl').read_text().splitlines())}
    ages=defaultdict(list);violations=[];groups=Counter();weights=[];anchor_keys=[]
    training=set(splits['train']); rows_by_ep={}
    if splits['train']!=sorted(splits['train']):raise ValueError('Expected training split in episode order for sampler weights')
    for ep in sorted(episodes):
        rows=[json.loads(x) for x in (root/f'quality/episode_{ep:06d}.jsonl').read_text().splitlines()]
        parquet=root/f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet'
        table=pq.read_table(parquet,columns=['timestamp','observation.state','action'])
        times=np.array(table['timestamp'].to_pylist());mono=np.array([r['anchor_monotonic_s'] for r in rows])
        if len(rows)!=len(times) or len(rows)!=episodes[ep]['length']:violations.append([ep,'length mismatch']);continue
        if not np.isfinite(times).all() or not np.isfinite(mono).all() or np.any(np.diff(times)<=0) or np.any(np.diff(mono)<=0):violations.append([ep,'nonfinite/nonmonotonic timestamps'])
        ages['export_interval_s'].extend(np.diff(times).tolist())
        ages['capture_minus_export_elapsed_s'].extend(((mono-mono[0])-(times-times[0])).tolist())
        states=np.array(table['observation.state'].to_pylist());actions=np.array(table['action'].to_pylist())
        if states.shape!=(len(rows),6) or actions.shape!=(len(rows),7) or not np.isfinite(states).all() or not np.isfinite(actions).all():violations.append([ep,'invalid pose/action'])
        for r in rows:
            for k in ['robot_age_s','action_delay_s','wrist_archive_age_s','position_age_s','velocity_age_s']:
                v=r.get(k)
                if v is not None and np.isfinite(v):ages[k].append(v)
            for k,v in r.get('camera_ages_s',{}).items():
                if v is not None and np.isfinite(v):ages['camera_'+k+'_age_s'].append(v)
        if ep in training:
            valid=set(quality['usable_frames'][str(ep)])
            task=episodes[ep]['tasks'][0]
            if isinstance(task,str):task=json.loads(task)
            task=task.get('objects',str(task))
            if isinstance(task,list):task='|'.join(task)
            for i,r in enumerate(rows):
                if all(max(0,min(len(rows)-1,i+offset)) in valid for offset in c['observation_history']):
                    key=(task,r.get('manipulation_phase','unknown'))
                    groups[key]+=1;anchor_keys.append(key)
        rows_by_ep[ep]=(rows,states,actions)
    # Partial task/phase balancing, capped to avoid repeatedly sampling rare noisy labels.
    raw=np.array([1/np.sqrt(groups[key]) for key in anchor_keys]);raw/=raw.mean();raw=np.clip(raw,.25,4.)
    # Unknown phases are not made more authoritative by being rare.
    raw=np.array([min(w,1.) if key[1]=='unknown' else w for w,key in zip(raw,anchor_keys)])
    np.save(a.output/'sampling_weights.npy',raw)
    write(a.output/'timing_audit.json',dict(episodes=len(episodes),violations=violations,
        timing={k:quantiles(v) for k,v in ages.items()},
        interpretation='Metadata/timeline checks only; native visual-to-command latency and physical calibration are not established by these checks. No labels or timestamps changed.'))
    # Reuse the already requested ten-episode results to audit the hold baseline and phases.
    old=PROJECT/'scratch/real_robot/action-evaluation/validation-10-seed12345'
    breakdown={}
    for family in ['dom','domv','krv100']:
        data=defaultdict(list);baseline=[]
        for sample in map(json.loads,(old/f'{family}-predictions.jsonl').read_text().splitlines()):
            ep,fr=sample['episode'],sample['frame'];rows,states,actions=rows_by_ep[ep]
            target=np.asarray(sample['target']);pred=np.asarray(sample['prediction']);valid=np.asarray(sample['valid'])
            # Verify decoded targets against the actual absolute recorded command FK.
            future=actions[np.minimum(fr+np.arange(len(target)),len(actions)-1)]
            if not np.allclose(target[:,:3],future[:,:3],atol=1e-6):violations.append([ep,fr,'absolute XYZ label mismatch'])
            phase=rows[fr].get('manipulation_phase','unknown')
            err=np.linalg.norm(pred[:,:3]-target[:,:3],axis=1)*1000
            data[phase].extend(err[valid].tolist())
            baseline.append(float(np.linalg.norm(states[fr,:3]-target[0,:3])*1000))
        breakdown[family]=dict(phase_mean_distance_mm={k:dict(targets=len(v),mean=float(np.mean(v))) for k,v in data.items()},
            hold_current_first_target_distance_mm=float(np.mean(baseline)))
    write(a.output/'phase_error_audit.json',breakdown)
    write(a.output/'sampling_audit.json',dict(anchors=len(raw),groups=[dict(task=t,phase=p,anchors=n) for (t,p),n in groups.items()],
        method='inverse square-root task/phase frequency; normalize mean=1 then cap [0.25,4]; unknown phase capped at 1; replacement; exactly one training-anchor count per epoch',
        min_weight=float(raw.min()),max_weight=float(raw.max()),phase_labels='candidate/inferred labels retained as such; no placement or recovery ground truth invented'))
    if violations:raise RuntimeError(f'Audit violations: {violations[:10]}')
    buckets=defaultdict(list);validation_index=0
    for ep in splits['validation']:
        rows,_,_=rows_by_ep[ep];usable=set(quality['usable_frames'][str(ep)])
        for fr,r in enumerate(rows):
            if all(max(0,min(len(rows)-1,fr+offset)) in usable for offset in c['observation_history']):
                buckets[(episodes[ep]['tasks'][0],r.get('manipulation_phase','unknown'))].append(validation_index)
                validation_index+=1
    rng=np.random.default_rng(12345)
    for bucket in buckets.values():rng.shuffle(bucket)
    selected=[]
    while len(selected)<min(256,validation_index):
        for bucket in buckets.values():
            if bucket and len(selected)<256:selected.append(bucket.pop())
    write(a.output/'validation_probe.json',dict(indices=sorted(selected),eligible_validation_anchors=validation_index,
        method='Round-robin task/phase buckets, shuffled with seed 12345; selection probe is balanced, full validation flow remains naturally distributed.'))
    # Remove stale parent-run execution and verification claims.
    for key in ['source_hashes','checkpoint_smoke_reports','checkpoint_architecture_verification','completed_epoch_policy','audit_reports','checkpoint_selection_reason','profiling_note']:
        c.pop(key,None)
    c.update(recipe_version='real_action_v5',parent_run=str(a.parent.resolve()),families=['dom','domv','krv100'],
        training_started=False,status='prepared',microbatch=16,
        sampling_weights=str((a.output/'sampling_weights.npy').resolve()),
        sampling_weights_sha256=hashlib.sha256((a.output/'sampling_weights.npy').read_bytes()).hexdigest(),
        sampling_audit=str((a.output/'sampling_audit.json').resolve()),
        max_epochs=15,early_stopping_patience=5,
        action_objective=dict(early_steps=4,early_weight=4.,position_scale_m=.02,rotation_scale_deg=10.,gripper_scale=.2,
            position_weight=1.,rotation_weight=1.,gripper_weight=1.,auxiliary_weight=.1,
            auxiliary_every_steps=4,auxiliary_batch_size=2,validation_anchors=256,
            validation_noise_seeds=[12345,23456,34567],validation_seed=12345,
            selection='weighted decoded physical action score; scales and horizon weights identical to auxiliary loss',
            reconstruction='full differentiable deployment sampler; no endpoint approximation'),
        action_validation_indices=sorted(selected),
        comparison_note='Each method continues from its own previously selected best flow checkpoint. New optimizer; same frozen data, statistics, balancing and action objective. Not a common-initialization ablation.',
        audit_reports={k:str((a.output/(k+'.json')).resolve()) for k in ['timing_audit','phase_error_audit','sampling_audit']})
    c['checkpoint_preferences']=[];c['checkpoint']={}
    for family in c['families']:
        b=json.loads((a.parent/family/'best_checkpoint.json').read_text());directory=Path(b['path'])
        checkpoint_id=f'{family}-parent-epoch-{b["epoch"]:03d}'
        c['checkpoint'][family]=checkpoint_id
        c['checkpoint_preferences'].append(dict(checkpoint_id=checkpoint_id,path=str(directory),
            sha256={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ['config.json','model.safetensors']}))
    c['wandb']['run_names']={f:'real-v5-action-'+f for f in c['families']}
    c['recovery_data']=dict(status='not collected',required='Expert-corrected actions paired with observations from actual policy deviations; use existing collector, retain session provenance and separate collection sessions across splits.',
        synthetic_recovery_labels=False,robot_motion_started=False)
    c['source_hashes']={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in [
        'workstation/train_real_finetune.py','workstation/smoke_real_finetune.py','workstation/prepare_action_retraining.py',
        'real_robot/action_objective.py','real_robot/training_augmentation.py',
        'third_party/DynamicVLA/policies/dynamicvla/modeling_dynamicvla.py']}
    write(a.output/'contract.json',c)
    print(json.dumps(dict(anchors=len(raw),timing_violations=violations,phase_errors=breakdown,contract=str(a.output/'contract.json')),indent=2))


if __name__=='__main__':main()
