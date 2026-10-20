#!/usr/bin/env python3
"""Summarize all live rollouts, including failures, and create paired videos."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from tracking_online_common import write_json


def paired_video(left,right,dest):
    import av
    dest.parent.mkdir(parents=True,exist_ok=True)
    with av.open(str(left)) as a,av.open(str(right)) as b,av.open(str(dest),'w') as out:
        sa,sb=a.streams.video[0],b.streams.video[0]
        if sa.average_rate!=sb.average_rate:raise ValueError('Video frame rate mismatch')
        stream=out.add_stream('libx264',rate=sa.average_rate)
        stream.width=sa.width+sb.width;stream.height=sa.height;stream.pix_fmt='yuv420p'
        stream.thread_count=2
        if sa.height!=sb.height:raise ValueError('Video dimensions differ')
        from itertools import zip_longest
        count=0
        for fa,fb in zip_longest(a.decode(video=0),b.decode(video=0)):
            if fa is None or fb is None:raise ValueError('Paired videos have unequal frame counts')
            frame=av.VideoFrame.from_ndarray(np.concatenate([fa.to_ndarray(format='rgb24'),fb.to_ndarray(format='rgb24')],axis=1),format='rgb24')
            for packet in stream.encode(frame):out.mux(packet)
            count+=1
        for packet in stream.encode():out.mux(packet)
        return count


def main(args):
    rows=[json.loads(p.read_text()) for p in sorted(args.output.glob('*/*/episode_*/metrics.json'))]
    manifest=json.loads((args.output/'manifest.json').read_text())
    expected=2*len(manifest['episodes'])*len(manifest['speed_multipliers'])
    summary={'evaluation':'live closed-loop tracking in simulation time',
        'expected_rollouts':expected,'recorded_rollouts':len(rows),
        'complete':len(rows)==expected and all(r['metrics']['completed'] for r in rows),
        'groups':{},'paired':{},'pairing_issues':[],'side_by_side_videos':[],
        'notes':['Left video: baseline; right video: residual.',
                 'Each model observes its own robot and camera feedback; object trajectories and initial conditions are paired.',
                 'Simulation pauses for inference; latency is measured but not injected into physics.',
                 'Higher target speed comes from time scaling, which also increases acceleration.',
                 'All failed/incomplete rollouts count as unsuccessful; absent rollouts are reported as missing.']}
    flat=[]
    for r in rows:
        m=r['metrics'];flat.append({k:r[k] for k in ('model','scenario','speed_multiplier','episode')} |
            {k:v for k,v in m.items() if not isinstance(v,dict)} | {'error':r['error']})
    if flat:
        keys=list(dict.fromkeys(k for r in flat for k in r))
        with (args.output/'per_episode.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(flat)
    for speed in manifest['speed_multipliers']:
        for model in ('baseline','residual'):
            group=[r for r in rows if r['model']==model and r['speed_multiplier']==speed]
            g={'expected':len(manifest['episodes']),'recorded':len(group),
               'completed':sum(r['metrics']['completed'] for r in group),
               'success_count':sum(r['metrics']['success'] for r in group),
               'success_rate_over_expected':sum(r['metrics']['success'] for r in group)/len(manifest['episodes'])}
            for metric in ('hover_rmse_m','hover_p95_m','orientation_rmse_deg','position_within_2cm_fraction','fraction_above_train_max_speed','mean_inference_seconds'):
                values=[r['metrics'][metric] for r in group if r['metrics'].get(metric) is not None]
                g['mean_'+metric]=float(np.mean(values)) if values else None
            summary['groups'][f'{model}_{speed:g}x']=g
        paired=[]
        for ep in [e['episode'] for e in manifest['episodes']]:
            pair={r['model']:r for r in rows if r['episode']==ep and r['speed_multiplier']==speed}
            if set(pair)=={'baseline','residual'} and all(r['metrics']['completed'] for r in pair.values()):
                left,right=pair['baseline'],pair['residual']
                # Verify reset reproducibility and exact same physical target path.
                if left['trajectory']!=right['trajectory']:
                    summary['pairing_issues'].append({'episode':ep,'speed':speed,'error':'Mismatched target trajectories'})
                    continue
                a=json.loads((Path(left['videos']['opst_cam']).parent/'initial_state.json').read_text())
                b=json.loads((Path(right['videos']['opst_cam']).parent/'initial_state.json').read_text())
                if not np.allclose(a['all_joint_position'],b['all_joint_position'],atol=1e-4,rtol=0):
                    summary['pairing_issues'].append({'episode':ep,'speed':speed,'error':'Initial robot states differ beyond 1e-4 rad'})
                    continue
                paired.append(right['metrics']['hover_rmse_m']-left['metrics']['hover_rmse_m'])
                if not args.no_videos:
                    dest=args.output/'paired_videos'/f'{speed:g}x_episode_{ep:06d}.mp4'
                    count=paired_video(left['videos']['opst_cam'],right['videos']['opst_cam'],dest)
                    summary['side_by_side_videos'].append({'path':str(dest),'frames':count})
        if paired:
            delta=np.asarray(paired);rng=np.random.default_rng(42)
            bootstrap=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
            summary['paired'][f'{speed:g}x']={'completed_pairs':len(delta),'residual_minus_baseline_hover_rmse_m':float(delta.mean()),
                'episode_bootstrap_ci95':np.quantile(bootstrap,[.025,.975]).tolist(),'residual_better':int((delta<0).sum())}
    summary['pairing_valid']=not summary['pairing_issues']
    write_json(args.output/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for model in ('baseline','residual'):
        ys=[summary['groups'][f'{model}_{s:g}x'].get('mean_hover_rmse_m') for s in manifest['speed_multipliers']]
        ax.plot(manifest['speed_multipliers'],[1000*y if y is not None else np.nan for y in ys],marker='o',label=model)
    ax.set(xlabel='Object trajectory speed multiplier',ylabel='Mean episode hover RMSE (mm)',title='Live learned-policy tracking: IID and faster OOD')
    ax.legend();fig.tight_layout();fig.savefig(args.output/'tracking_comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--no-videos',action='store_true');main(p.parse_args())
