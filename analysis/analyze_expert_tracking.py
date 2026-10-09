#!/usr/bin/env python3
"""Measure the recorded expert's current EE position, not its future action label."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    v=np.asarray(values,dtype=float)
    return {'mean':float(v.mean()),'min':float(v.min()),
            'p05':float(np.quantile(v,.05)),'p10':float(np.quantile(v,.1)),
            'median':float(np.median(v)),'p90':float(np.quantile(v,.9)),
            'max':float(v.max())}


def analyze(dataset,comparison,output):
    dataset=Path(dataset);comparison=Path(comparison);output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    collection=json.load(open(dataset/'meta/collection.json'))
    cfg=collection['provenance']['config'];layout=collection['provenance']['layout']
    manifest=json.load(open(comparison/'manifest.json'))
    for k in ('controller','trajectory'):assert cfg[k]==manifest['config'][k]
    for k in ('robot_position','robot_quaternion'):np.testing.assert_allclose(layout[k],manifest['layout'][k],rtol=0,atol=1e-12)
    for k in ('scene_usd_sha256','object_usd_sha256'):assert layout[k]==manifest['layout'][k]
    fps=cfg['collection']['fps'];seconds=cfg['collection']['seconds']
    assert fps==25 and seconds==10
    splits=json.load(open(dataset/'meta/split_assignments.json'))
    membership={ep:split for split,ids in splits.items() for ep in ids}
    assert len(membership)==100
    # Original observation-tool hover offset is constant for the fixed apple mesh.
    # All saved paired trajectories must independently agree with this offset.
    offset=None;max_offset_deviation=0.;initial_checks=0
    for ep in splits['test']:
        trace=comparison/'baseline'/'iid'/f'episode_{ep:06d}'/'steps.jsonl'
        first=None
        for line in trace.open():
            r=json.loads(line)
            if first is None:first=r
            current=np.asarray(r['hover_target'])-r['object_position']
            if offset is None:offset=current
            max_offset_deviation=max(max_offset_deviation,float(np.linalg.norm(current-offset)))
        state=pq.read_table(dataset/'data/chunk-000'/f'episode_{ep:06d}.parquet',columns=['observation.state','observation.environment_state']).slice(0,1).to_pylist()[0]
        np.testing.assert_allclose(state['observation.state'][:3],first['ee_pose'][:3],rtol=0,atol=1e-7)
        np.testing.assert_allclose(state['observation.environment_state'][:3],first['object_position'],rtol=0,atol=1e-7)
        initial_checks+=1
    assert max_offset_deviation<1e-10
    records=[];errors_all=[];source_hashes={};boundary_samples=0
    for path in sorted((dataset/'data/chunk-000').glob('episode_*.parquet')):
        ep=int(path.stem.split('_')[-1]);assert ep in membership
        table=pq.read_table(path,columns=['observation.state','observation.environment_state','timestamp','frame_index'])
        state=np.array(table['observation.state'].to_pylist(),dtype=float)
        env=np.array(table['observation.environment_state'].to_pylist(),dtype=float)
        timestamps=np.array(table['timestamp'].to_pylist(),dtype=float)
        frames=np.array(table['frame_index'].to_pylist())
        assert len(state)==250 and np.isfinite(state).all() and np.isfinite(env).all()
        np.testing.assert_array_equal(frames,np.arange(250))
        np.testing.assert_allclose(timestamps,np.arange(250)/fps,rtol=0,atol=1e-6)
        target=env[:,:3]+offset
        errors=np.linalg.norm(state[:,:3]-target,axis=1)
        inside=errors<=.02;post=frames>=fps
        boundary_samples+=int(np.sum(np.abs(errors-.02)<2e-7))
        errors_all.extend(errors.tolist())
        records.append({'episode':ep,'split':membership[ep],'samples':len(state),
            'whole_episode_within_2cm_percent':100*float(inside.mean()),
            'after_first_second_within_2cm_percent':100*float(inside[post].mean()),
            'hover_rmse_mm':1000*float(np.sqrt(np.mean(errors**2))),
            'hover_max_mm':1000*float(errors.max()),
            'mean_object_speed_mps':float(np.linalg.norm(env[:,6:9],axis=1).mean())})
        source_hashes[str(path.relative_to(dataset))]=digest(path)
    assert len(records)==100
    def group(rows):
        return {'episodes':len(rows),
            'whole_episode_coverage_percent':distribution([r['whole_episode_within_2cm_percent'] for r in rows]),
            'post_1s_coverage_percent':distribution([r['after_first_second_within_2cm_percent'] for r in rows]),
            'per_episode_rmse_mm':distribution([r['hover_rmse_mm'] for r in rows]),
            'episodes_above_coverage_threshold':{str(t):sum(r['whole_episode_within_2cm_percent']>t for r in rows) for t in (50,70,80,90,95,99)},
            'perfect_coverage_episodes':sum(r['whole_episode_within_2cm_percent']==100 for r in rows)}
    report={'dataset':str(dataset),'evaluation_reference':str(comparison),
        'dataset_revision_from_evaluation_manifest':manifest['dataset_revision'],
        'sampling_hz':fps,'episode_seconds':seconds,'tolerance_m':.02,
        'warmup_excluded':True,'future_action_labels_excluded':True,
        'target_offset_from_object_origin_m':offset.tolist(),
        'target_offset_max_deviation_across_reference_traces_m':max_offset_deviation,
        'expert_and_evaluation_initial_observation_matches':initial_checks,
        'samples_within_0p2_micrometers_of_threshold':boundary_samples,
        'scope':'100 accepted original training-collection episodes; not newly collected diverse data and not expert rollouts under the nine new stress conditions.',
        'all_episodes':group(records),'splits':{k:group([r for r in records if r['split']==k]) for k in splits},
        'pooled_rmse_mm_at_25hz':1000*float(np.sqrt(np.mean(np.asarray(errors_all)**2))),
        'original_collection_pooled_rmse_mm_at_200hz':1000*collection['report']['rmse'],
        'original_collection_max_error_mm_at_200hz':1000*collection['report']['max_error'],
        'limitation':'Coverage is measured at the 25 Hz recorded observation timestamps. Raw 200 Hz expert telemetry is unavailable in this cached dataset; brief inter-frame errors may be missed.'}
    # A close match to the independent original collection report checks reconstruction.
    assert abs(report['pooled_rmse_mm_at_25hz']-report['original_collection_pooled_rmse_mm_at_200hz'])<.2
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (output/'source_hashes.json').write_text(json.dumps(source_hashes,indent=2))
    with (output/'per_episode.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    ids=[r['episode'] for r in records]
    axes[0].plot(ids,[r['whole_episode_within_2cm_percent'] for r in records],'.',label='Whole episode',markersize=6)
    axes[0].plot(ids,[r['after_first_second_within_2cm_percent'] for r in records],'+',label='After first second')
    axes[0].set(xlabel='Expert episode',ylabel='Time within 2 cm (%)',title='Recorded expert tracking coverage')
    lo=min(r['after_first_second_within_2cm_percent'] for r in records)
    axes[0].set_ylim(max(0,lo-1),100.2);axes[0].legend(frameon=False);axes[0].grid(alpha=.2)
    axes[1].plot(ids,[r['hover_rmse_mm'] for r in records],'.',color='#297764')
    axes[1].set(xlabel='Expert episode',ylabel='Position RMSE (mm)',title='Distance to the same moving hover target')
    axes[1].grid(alpha=.2)
    fig.suptitle('Original expert data: 100 episodes, 10 seconds each')
    fig.text(.05,.01,'Measured current EE pose, 25 Hz observations; warmup and future-action continuation excluded.',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(output/'expert_tracking_coverage.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,type=Path);p.add_argument('--comparison',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();analyze(a.dataset,a.comparison,a.output)
