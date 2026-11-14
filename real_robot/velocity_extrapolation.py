"""Bounded causal velocity extrapolation, evaluated against withheld tracker values.

Preserves measured/repaired values, positions, actions, timing and phase labels.
Predictions never seed further predictions. No hardware or Hub writes.
"""
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq

from .dom_dataset import DATA_PATH, json_write, jsonl_read, jsonl_write, numeric_stats, sha256, write_parquet
from .static_object import SCHEMA as STATIC_SCHEMA,quality_valid as static_quality_valid


def measured(q):
    return bool(q['velocity_estimated'] and not q['velocity_zero_filled']
                and not q.get('velocity_repaired',False) and not q.get('velocity_extrapolated',False))


def position_motion(rows,quality,target,cutoff,horizon):
    """Extrapolate a well-supported straight-line position fit, not stale rows."""
    q=quality[target];now=q['anchor_monotonic_s'];samples=[];seen=set()
    for j in range(cutoff,-1,-1):
        old=quality[j]
        if now-old['anchor_monotonic_s']>horizon+.35:break
        if (not old['velocity_required'] or old['manipulation_phase']!=q['manipulation_phase']
                or old.get('phase_confirmed')):break
        grip=rows[j]['action'][-1];current=rows[target]['action'][-1]
        if (grip>.25 and current<=0) or (grip<=0 and current>.25):break
        if (not old['position_current'] or old.get('position_repaired')
                or not old.get('depth_position_corrected',True)):continue
        stamp=old.get('position_measurement_monotonic_s',old['anchor_monotonic_s']-old.get('position_age_s',0))
        if stamp>now+1e-8 or stamp in seen:continue
        seen.add(stamp);samples.append((stamp,j,np.asarray(rows[j]['observation.environment_state'][:3],float)))
        if len(samples)==5:break
    if len(samples)<5:return None
    samples.sort(key=lambda x:x[0]);t=np.array([x[0] for x in samples]);positions=np.array([x[2] for x in samples])
    age=now-t[-1]
    if not 0<=age<=horizon+1e-8 or not .12<=t[-1]-t[0]<=.30:return None
    centered=t-t.mean();denom=float(centered@centered)
    velocity=centered@(positions-positions.mean(0))/denom
    residual=positions-positions.mean(0)-centered[:,None]*velocity
    rms=float(np.sqrt(np.mean(np.sum(residual**2,axis=1))))
    uncertainty=float(np.sqrt(np.sum(residual**2)/(len(samples)-2)/denom))
    if not np.isfinite(velocity).all() or np.linalg.norm(velocity)>2. or rms>.004 or uncertainty>.05:return None
    return velocity,dict(horizon_s=float(age),source_frames=[x[1] for x in samples],
        measurement_times_s=t.tolist(),source_measurement_kind='positions',
        position_fit_rms_m=rms,fit_velocity_uncertainty_mps=uncertainty)


def predict_velocity(rows, quality, target, cutoff, horizon=.20):
    """Hold recent measured velocity constant only across a short stable interval."""
    now = quality[target]['anchor_monotonic_s']
    q = quality[target]
    if not q['velocity_required'] or q.get('phase_confirmed'):
        return None
    phase = q['manipulation_phase']
    candidates=[];seen=set()
    for j in range(cutoff,-1,-1):
        old=quality[j]
        if now-old['anchor_monotonic_s'] > horizon+.35:break
        if (not old['velocity_required'] or old['manipulation_phase']!=phase
                or old.get('phase_confirmed')):break
        grip=rows[j]['action'][-1];current=rows[target]['action'][-1]
        if (grip>.25 and current<=0) or (grip<=0 and current>.25):break
        if not measured(old):continue
        stamp=old.get('velocity_measurement_monotonic_s',old['anchor_monotonic_s']-old['velocity_age_s'])
        if stamp>now+1e-8 or stamp in seen:continue
        seen.add(stamp)
        candidates.append((stamp,j,np.asarray(rows[j]['observation.environment_state'][6:9],float)))
        if len(candidates)==3:break
    if len(candidates)<3:return position_motion(rows,quality,target,cutoff,horizon)
    candidates.sort(key=lambda x:x[0]);times=np.array([x[0] for x in candidates]);vel=np.array([x[2] for x in candidates])
    age=now-times[-1]
    if not 0<=age<=horizon+1e-8 or not .06<=times[-1]-times[0]<=.30:
        return position_motion(rows,quality,target,cutoff,horizon)
    if not np.isfinite(vel).all() or np.linalg.norm(vel,axis=1).max()>2.:return None
    # Stable recent motion supports a short constant-velocity continuation.
    spread=float(np.linalg.norm(vel-vel[-1],axis=1).max())
    if spread>.08:return None
    return vel[-1].copy(),dict(horizon_s=float(age),source_frames=[x[1] for x in candidates],
        measurement_times_s=times.tolist(),source_measurement_kind='velocities',recent_velocity_spread_mps=spread)


def extrapolate_episode(rows,quality,horizon=.20):
    result,flags=copy.deepcopy(rows),copy.deepcopy(quality)
    for i,q in enumerate(flags):
        if (q.get('static_collection_schema')==STATIC_SCHEMA or not q['velocity_required'] or measured(q) or q.get('velocity_repaired')
                or q.get('velocity_extrapolated')):continue
        prediction=predict_velocity(rows,quality,i,i-1,horizon)
        if prediction is None:continue
        velocity,provenance=prediction
        result[i]['observation.environment_state'][6:9]=velocity.astype(np.float32).tolist()
        q.update(velocity_repaired=True,velocity_extrapolated=True,
                 velocity_input_source='bounded_causal_constant_velocity_extrapolation',
                 velocity_extrapolation_uses_future=False,velocity_extrapolation=provenance)
        # A velocity estimate does not rehabilitate an invalid object position.
        position_ok=q['position_current'] or q.get('position_repaired',False)
        depth_ok=q.get('depth_position_corrected',True)
        q['tracking_usable']=bool(position_ok and depth_ok)
    return result,flags


def evaluate(root,horizons=(.08,.12,.20)):
    """Pseudo-gaps with no future support; references are tracker estimates, not GT."""
    splits=json.loads((root/'meta/splits.json').read_text());groups={ep:s for s,ids in splits.items() for ep in ids}
    errors=defaultdict(list);position_errors=defaultdict(list)
    for ep,split in groups.items():
        rows=pq.read_table(root/DATA_PATH.format(episode_chunk=ep//1000,episode_index=ep)).to_pylist()
        quality=jsonl_read(root/f'quality/episode_{ep:06d}.jsonl')
        for i in range(12,len(rows),7):
            if not measured(quality[i]):continue
            reference=np.asarray(rows[i]['observation.environment_state'][6:9])
            for horizon in horizons:
                # Hide the complete gap, including any interpolated velocities.
                cutoff=i-int(round(horizon*25))
                estimate=predict_velocity(rows,quality,i,cutoff,horizon+.06)
                if estimate is not None:
                    errors[(split,str(horizon))].append(float(np.linalg.norm(estimate[0]-reference)))
                positional=position_motion(rows,quality,i,cutoff,horizon+.06)
                if positional is not None:
                    position_errors[(split,str(horizon))].append(float(np.linalg.norm(positional[0]-reference)))
    result={}
    for (split,horizon),values in errors.items():
        result.setdefault(split,{})[horizon]=dict(samples=len(values),median_error_mps=float(np.median(values)),
            p95_error_mps=float(np.percentile(values,95)),max_error_mps=float(max(values)))
    fit_result={}
    for (split,horizon),values in position_errors.items():
        fit_result.setdefault(split,{})[horizon]=dict(samples=len(values),median_error_mps=float(np.median(values)),
            p95_error_mps=float(np.percentile(values,95)),max_error_mps=float(max(values)))
    return dict(reference='withheld original tracker velocity; not independent physical ground truth',
                selection='Select horizon from training pseudo-gaps only; held-out results are diagnostic',
                results=result,position_fit_results=fit_result)


def process(root,output,horizon):
    if not 0<horizon<=.20:raise ValueError('Maximum supported horizon is 0.20 seconds')
    if output.exists() or output==root:raise ValueError('Choose a new output path')
    output.mkdir(parents=True)
    shutil.copytree(root/'meta',output/'meta')
    (output/'videos').symlink_to((root/'videos').resolve(),target_is_directory=True)
    info=json.loads((root/'meta/info.json').read_text());splits=json.loads((root/'meta/splits.json').read_text())
    policy=json.loads((root/'meta/quality_policy.json').read_text());stats=jsonl_read(root/'meta/episodes_stats.jsonl')
    counts=Counter();episodes=[]
    for ep in range(info['total_episodes']):
        name=DATA_PATH.format(episode_chunk=ep//1000,episode_index=ep)
        rows=pq.read_table(root/name).to_pylist();quality=jsonl_read(root/f'quality/episode_{ep:06d}.jsonl')
        changed,flags=extrapolate_episode(rows,quality,horizon)
        added=sum(q.get('velocity_extrapolated',False) and not old.get('velocity_extrapolated',False) for q,old in zip(flags,quality))
        before=sum(q['tracking_usable'] for q in quality);after=sum(q['tracking_usable'] for q in flags)
        for a,b,old,q in zip(rows,changed,quality,flags):
            for key in a:
                if key!='observation.environment_state':assert a[key]==b[key],key
            assert a['observation.environment_state'][:6]==b['observation.environment_state'][:6]
            if static_quality_valid(old) or measured(old) or old.get('velocity_repaired') or not old['velocity_required']:
                assert a['observation.environment_state']==b['observation.environment_state']
            assert old.get('phase_confirmed')==q.get('phase_confirmed')
        write_parquet(changed,output/name);jsonl_write(output/f'quality/episode_{ep:06d}.jsonl',flags)
        policy['usable_frames'][str(ep)]=[i for i,q in enumerate(flags) if q['tracking_usable']]
        stats[ep]['stats'].pop('observation.environment_state',None)
        selected=[r['observation.environment_state'] for r,q in zip(changed,flags) if q['tracking_usable']]
        if selected:stats[ep]['stats']['observation.environment_state']=numeric_stats(selected)
        counts.update(frames=len(rows),velocities_extrapolated=added,usable_before=before,usable_after=after,
                      missing_velocity_remaining=sum(q['velocity_required'] and not measured(q) and not q.get('velocity_repaired') and not static_quality_valid(q) for q in flags))
        episodes.append(dict(episode=ep,extrapolated=added,usable_before=before,usable_after=after))
    from lerobot.common.datasets.compute_stats import aggregate_stats
    train=[{k:{n:np.asarray(v) for n,v in d.items()} for k,d in stats[ep]['stats'].items()} for ep in splits['train']]
    agg=aggregate_stats(train);json_write(output/'meta/train_stats.json',{k:{n:v.tolist() for n,v in d.items()} for k,d in agg.items()})
    jsonl_write(output/'meta/episodes_stats.jsonl',stats)
    policy['velocity_extrapolation']=dict(method='causal_constant_velocity_or_stable_position_fit',max_horizon_s=horizon,
        min_distinct_measurements=3,max_recent_variation_mps=.08,max_speed_mps=2.,uses_future=False,
        position_fit_min_measurements=5,position_fit_min_span_s=.12,position_fit_max_span_s=.30,
        position_fit_max_rms_m=.004,position_fit_max_velocity_uncertainty_mps=.05,
        excludes_gripper_and_phase_transitions=True,extrapolated_samples_never_seed_predictions=True)
    json_write(output/'meta/quality_policy.json',policy)
    report=dict(complete=True,source_dataset=str(root),source_policy_sha256=sha256(root/'meta/quality_policy.json'),
        code_sha256=sha256(Path(__file__)),settings=policy['velocity_extrapolation'],totals=dict(counts),episodes=episodes)
    json_write(output/'velocity_extrapolation_report.json',report)
    json_write(output/'export_report.json',dict(complete=True,episodes=info['total_episodes'],frames=info['total_frames'],
        source_dataset=str(root),velocity_extrapolation='velocity_extrapolation_report.json'))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evaluate',action='store_true');p.add_argument('--max-horizon-s',type=float,default=.20)
    a=p.parse_args()
    if a.evaluate:
        r=evaluate(a.dataset.resolve());json_write(a.output,r);print(json.dumps(r,indent=2))
    else:
        r=process(a.dataset.resolve(),a.output.resolve(),a.max_horizon_s);print(json.dumps(r['totals'],indent=2))
