#!/usr/bin/env python3
"""Three-model live comparison; policy terminations remain visible in every report."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from tracking_online_common import write_json
from tracking_variants import VARIANT_LABELS


def joint_match(left,right):
    a=json.loads((Path(left['videos']['opst_cam']).parent/'initial_state.json').read_text())
    b=json.loads((Path(right['videos']['opst_cam']).parent/'initial_state.json').read_text())
    return left['trajectory']==right['trajectory'] and np.allclose(a['all_joint_position'],b['all_joint_position'],atol=1e-4,rtol=0)


def comparison_video(records, models, destination, fps, frames):
    """Persistent method/scenario labels, including after a rollout terminates."""
    import av
    from contextlib import ExitStack
    from tracking_visuals import PanelRenderer, WIDTH, HEIGHT
    with ExitStack() as stack:
        decoders={}; renderers={}
        for i,model in enumerate(models):
            row=dict(records.get(model,{'metrics':{'completed':False},'failure_type':'missing'}),model=model)
            renderers[model]=PanelRenderer(row,position=('LEFT','MIDDLE','RIGHT')[i] if len(models)==3 else str(i+1))
            if row.get('video_frames',{}).get('opst_cam',0)>0:
                container=stack.enter_context(av.open(str(row['videos']['opst_cam'])))
                assert float(container.streams.video[0].average_rate)==fps
                decoders[model]=iter(container.decode(video=0))
        out=stack.enter_context(av.open(str(destination),'w'))
        stream=out.add_stream('libx264',rate=fps)
        stream.width=WIDTH*len(models);stream.height=HEIGHT;stream.pix_fmt='yuv420p';stream.thread_count=2
        stream.options={'crf':'18','preset':'fast'}
        for index in range(frames):
            panels=[]
            for model in models:
                frame=next(decoders[model],None) if model in decoders else None
                panels.append(renderers[model].render(None if frame is None else frame.to_ndarray(format='rgb24'),index,fps))
            video_frame=av.VideoFrame.from_ndarray(np.concatenate(panels,axis=1),format='rgb24')
            for packet in stream.encode(video_frame):out.mux(packet)
        for packet in stream.encode():out.mux(packet)
    return frames


def main(args):
    manifest=json.loads((args.output/'manifest.json').read_text())
    models=list(manifest['models']);episodes=[r['episode'] for r in manifest['episodes']];speeds=manifest['speed_multipliers']
    rows=[json.loads(p.read_text()) for p in sorted(args.output.glob('*/*/episode_*/metrics.json'))]
    expected=len(models)*len(episodes)*len(speeds)
    keys=[(r['model'],r['speed_multiplier'],r['episode']) for r in rows]
    expected_keys=set(itertools.product(models,speeds,episodes))
    assert len(keys)==len(set(keys)) and set(keys)<=expected_keys,'Unexpected/duplicate rollout'
    summary={'evaluation':'live closed-loop tracking in simulation time','models':{m:VARIANT_LABELS[m] for m in models},
        'expected_rollouts':expected,'recorded_rollouts':len(rows),
        'complete':len(rows)==expected and all(r['metrics']['completed'] or r.get('failure_type')=='policy_termination' for r in rows),
        'all_rollouts_completed':len(rows)==expected and all(r['metrics']['completed'] for r in rows),
        'policy_terminations':sum(r.get('failure_type')=='policy_termination' for r in rows),
        'groups':{},'pairwise':{},'pairing_issues':[],'comparison_videos':[],
        'notes':['All policy terminations count as failed rollouts. Missing rollouts remain missing.',
          'Mean errors and paired RMSE differences use completed rollouts only; compare termination/success rates alongside them.',
          'A complete benchmark may contain terminated rollouts; all_rollouts_completed distinguishes this.',
          'Simulation pauses for inference. Speed multipliers also scale acceleration by their square.',
          'Video panels follow model order; termination cards fill the remaining requested horizon.']}
    flat=[]
    for row in rows:
        flat.append({k:row.get(k) for k in ('model','scenario','speed_multiplier','episode','failure_type','error','requested_seconds','recorded_seconds')} |
                    {k:v for k,v in row['metrics'].items() if not isinstance(v,dict)})
    if flat:
        with (args.output/'per_episode.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in flat for k in r)))
            writer.writeheader();writer.writerows(flat)
    for speed in speeds:
        for model in models:
            group=[r for r in rows if r['model']==model and r['speed_multiplier']==speed]
            completed=[r for r in group if r['metrics']['completed']]
            g={'expected':len(episodes),'recorded':len(group),'completed':len(completed),
               'policy_terminations':sum(r.get('failure_type')=='policy_termination' for r in group),
               'success_count':sum(r['metrics']['success'] for r in group),
               'success_rate_over_expected':sum(r['metrics']['success'] for r in group)/len(episodes)}
            for metric in ('hover_rmse_m','hover_p95_m','orientation_rmse_deg','position_within_2cm_fraction','fraction_above_train_max_speed','mean_inference_seconds'):
                values=[r['metrics'][metric] for r in completed if r['metrics'].get(metric) is not None]
                g['completed_mean_'+metric]=float(np.mean(values)) if values else None
            summary['groups'][f'{model}_{speed:g}x']=g
        for left,right in itertools.combinations(models,2):
            deltas=[]
            for ep in episodes:
                pair={r['model']:r for r in rows if r['episode']==ep and r['speed_multiplier']==speed and r['model'] in (left,right)}
                if len(pair)!=2:continue
                if not joint_match(pair[left],pair[right]):
                    summary['pairing_issues'].append({'left':left,'right':right,'episode':ep,'speed':speed,'error':'Initial joints or trajectory differ'})
                    continue
                if all(r['metrics']['completed'] for r in pair.values()):
                    deltas.append(pair[right]['metrics']['hover_rmse_m']-pair[left]['metrics']['hover_rmse_m'])
            stat={'completed_matched_pairs':len(deltas),'expected_pairs':len(episodes),'right_minus_left_hover_rmse_m':None,'episode_bootstrap_ci95':None}
            if deltas:
                delta=np.array(deltas);rng=np.random.default_rng(42)
                boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
                stat.update(right_minus_left_hover_rmse_m=float(delta.mean()),episode_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist(),episodes_right_better=int((delta<0).sum()))
            summary['pairwise'][f'{right}_minus_{left}_{speed:g}x']=stat
        if not args.no_videos:
            for ep in episodes:
                records={r['model']:r for r in rows if r['episode']==ep and r['speed_multiplier']==speed}
                if not records:continue
                dest=args.output/'comparison_videos'/f'{speed:g}x_episode_{ep:06d}.mp4';dest.parent.mkdir(exist_ok=True)
                frames=round(max(r['requested_seconds'] for r in records.values())*25)
                count=comparison_video(records,models,dest,25,frames)
                summary['comparison_videos'].append({'path':str(dest),'frames':count,'model_order':models})
    summary['pairing_valid']=not summary['pairing_issues']
    write_json(args.output/'summary.json',summary)
    from tracking_visuals import plot_live
    plot_live(summary,speeds,args.output/'tracking_comparison.png')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--no-videos',action='store_true');main(p.parse_args())
