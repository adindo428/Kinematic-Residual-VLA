"""Labeled review videos and mode/velocity coverage, outside clean training videos."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from .storage import atomic_json, read_jsonl


def make_previews(dataset, output):
    dataset, output = Path(dataset), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    info = json.loads((dataset/'meta/info.json').read_text())
    rows = read_jsonl(dataset/'meta/episode_groups.jsonl')
    groups = defaultdict(list)
    for row in rows:
        groups[row['pair_group']].append(row)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import findfont
    font = ImageFont.truetype(findfont('DejaVu Sans'), 19)
    small = ImageFont.truetype(findfont('DejaVu Sans'), 15)
    selected, review = set(), []
    for pair in groups.values():
        obj = pair[0]['object']
        if obj in selected:
            continue
        selected.add(obj)
        pair = sorted(pair, key=lambda r:r['mode'])
        assert [r['mode'] for r in pair] == [-45,45]
        videos, tables = [], []
        for row in pair:
            eid = row['episode_index']
            videos.append(av.open(str(dataset/info['video_path'].format(
                episode_chunk=eid//info['chunks_size'], episode_index=eid,
                video_key='observation.images.opst_cam'))))
            tables.append(pq.read_table(dataset/info['data_path'].format(
                episode_chunk=eid//info['chunks_size'], episode_index=eid)).to_pylist())
        target = output/f'{obj}_minus45_vs_plus45.mp4'
        try:
            with av.open(str(target), 'w') as container:
                stream = container.add_stream('libx264', rate=info['fps'])
                stream.width, stream.height, stream.pix_fmt = 960, 460, 'yuv420p'
                stream.options = {'crf':'20','preset':'fast'}
                count = 0
                for frames in zip(*(v.decode(video=0) for v in videos)):
                    canvas = Image.new('RGB', (960,460), (20,24,30))
                    draw = ImageDraw.Draw(canvas)
                    for side,(row,frame,table,color) in enumerate(zip(pair,frames,tables,('#69d5b0','#eea6cf'))):
                        x=side*480
                        canvas.paste(frame.to_image().convert('RGB').resize((480,360)), (x,64))
                        draw.text((x+10,8),f"{obj} | fixed {row['mode']:+d} degrees",font=font,fill=color)
                        draw.text((x+10,35),f"Paired expert demonstration | episode {row['episode_index']}",font=small,fill='white')
                        record=table[count]
                        vy=record['observation.environment_state'][7]
                        draw.text((x+10,432),f"t={record['timestamp']:.2f}s | base v_y={vy:+.3f} m/s | overhead",font=small,fill='white')
                    for packet in stream.encode(av.VideoFrame.from_image(canvas)):
                        container.mux(packet)
                    count+=1
                assert count == len(tables[0]) == len(tables[1]), 'Paired preview length mismatch'
                for packet in stream.encode():
                    container.mux(packet)
        finally:
            for video in videos:
                video.close()
        review.append(dict(object=obj,pair_group=pair[0]['pair_group'],episodes=[r['episode_index'] for r in pair],video=target.name,frames=count))
    bins=np.array([-.151,-.10,-.05,-.01,.01,.05,.10,.151])
    counts=np.zeros((2,len(bins)-1),dtype=int)
    errors=defaultdict(list)
    for row in rows:
        eid=row['episode_index']
        table=pq.read_table(dataset/info['data_path'].format(episode_chunk=eid//info['chunks_size'],episode_index=eid), columns=['observation.environment_state'])
        velocity=np.asarray(table.column(0).to_pylist())[:,7]
        counts[0 if row['mode']==-45 else 1]+=np.histogram(velocity,bins)[0]
        metadata=json.loads((dataset/f'diagnostics/episode_{eid:06d}/episode.json').read_text())
        errors[f"{row['object']} / {row['mode']:+d}"] .append(metadata['metrics'])
    labels=[f'{a:+.2f} to {b:+.2f}' for a,b in zip(bins[:-1],bins[1:])]
    fig,ax=plt.subplots(figsize=(11,5))
    x=np.arange(len(labels))
    for i,(name,color) in enumerate((('Fixed -45 degrees','#269f7a'),('Fixed +45 degrees','#bb5c96'))):
        ax.bar(x+(i-.5)*.36,counts[i],.36,label=name,color=color)
    ax.set_xticks(x,labels,rotation=20)
    ax.set_xlabel('Signed lateral velocity in robot-base axes (m/s)')
    ax.set_ylabel('Accepted demonstration frames')
    ax.set_title('Paired training demonstrations: both modes at both velocity signs')
    ax.legend();fig.tight_layout();fig.savefig(output/'mode_velocity_coverage.png',dpi=160);plt.close(fig)
    atomic_json(output/'summary.json',dict(dataset=str(dataset.resolve()),previews=review,velocity_bin_edges=bins.tolist(),mode_order=[-45,45],frame_counts=counts.tolist(),accepted_episode_metrics=dict(errors),scope='Accepted expert demonstrations only; failed pairs remain in raw runs. No learned-policy evaluation.'))
    (output/'README.md').write_text('# Paired orientation demonstrations\n\n'
        'These are expert-controller demonstrations, not DOM-VLA, KR-VLA-Field, or KR-VLA-Coordinate policy evaluations. '
        'Each video shows the same object trajectory twice: fixed -45 degrees on the left and fixed +45 degrees on the right. '
        'Both trajectories contain positive and negative lateral velocities. Raw training videos have no overlays.\n\n'
        'The initial dataset is balanced at every signed velocity because both modes are collected for every path. '
        'The desired future mode selector is 50/50 near zero lateral velocity, favoring +45 degrees increasingly to the right and -45 increasingly to the left. '
        'That preference is not imposed on these labels. Use observed tracking errors and failures to investigate it; these accepted-only previews cannot establish the best mode or OOD performance. '
        'Positive/negative base Y is shown explicitly; camera-screen left/right is not a coordinate definition.\n\n'
        'The coverage plot counts accepted frames. summary.json records preview episode IDs, velocity counts and per-episode quality metrics. '
        'Trajectory pairs remain together in the supplied 80/10/10 splits.\n')
    return review


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(make_previews(args.dataset,args.output),indent=2))
