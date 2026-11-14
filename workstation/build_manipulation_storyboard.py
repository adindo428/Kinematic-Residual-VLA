#!/usr/bin/env python3
"""Compose a six-row scientific storyboard from exact, traceable video frames."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(config,out):
    assets=out/'frames';assets.mkdir(exist_ok=True)
    for i,row in enumerate(config['sources']):
        path=REPO/row['path']
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(path)],text=True))['streams'][0]
        fps=Fraction(probe['avg_frame_rate']);indices=row['frame_indices']
        assert indices==sorted(set(indices)) and len(indices)==6
        assert indices[-1]<int(probe['nb_frames'])
        x0,y0,x1,y1=row['camera_crop'];width,height=x1-x0,y1-y0
        select='+'.join(f'eq(n\\,{j})' for j in indices)
        pixels=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(path),
            '-vf',f'select={select},crop={width}:{height}:{x0}:{y0}','-vsync','0',
            '-f','rawvideo','-pix_fmt','rgb24','-threads','1','-'])
        array=np.frombuffer(pixels,np.uint8).reshape(-1,height,width,3)
        assert len(array)==len(indices)
        row.update(source_sha256=sha(path),encoded_fps=str(fps),source_frame_count=int(probe['nb_frames']),frames=[])
        for j,(index,image) in enumerate(zip(indices,array)):
            frame=assets/f'row{i+1:02d}_{row["object"].lower()}_{row["phase_labels"][j].lower()}_f{index:04d}.png'
            Image.fromarray(image).save(frame)
            row['frames'].append({'path':str(frame.relative_to(out)),'frame_index_zero_based':index,
                'clip_time_s':float(Fraction(index)/fps),'phase':row['phase_labels'][j],
                'sha256':sha(frame),'size':[width,height]})
    return config


def render(config,out):
    dpi=180;W=2520;F=360;IH=270;row_h=316
    left=244;within=12;between=24;group_w=2*F+within
    group_a=182;start_a=244;group_b=start_a+3*row_h+18;start_b=group_b+62
    footer=start_b+3*row_h+8;H=footer+112
    fig=plt.figure(figsize=(W/dpi,H/dpi),dpi=dpi,facecolor='white')
    colors=['#2866A7','#7352A0','#20866F'];dark='#183044';muted='#5C6C79'
    def rect(x,y,w,h,color,edge=None,lw=.5):
        fig.add_artist(Rectangle((x/W,1-(y+h)/H),w/W,h/H,transform=fig.transFigure,
            facecolor=color,edgecolor=edge,linewidth=lw,zorder=0))
    def text(x,y,s,px=26,color=dark,weight='normal',ha='left'):
        fig.text(x/W,1-y/H,s,fontsize=px*72/dpi,color=color,fontweight=weight,
                 va='top',ha=ha,fontfamily='DejaVu Sans')
    def line(x,y1,y2,color='#C8D1DB',lw=1):
        fig.add_artist(Line2D([x/W,x/W],[1-y1/H,1-y2/H],transform=fig.transFigure,color=color,linewidth=lw,zorder=0))
    text(32,23,config['title'],48,weight='bold')
    text(34,84,'Three object–container tasks, shown through successive manipulation stages',26,color=muted)
    text(34,137,'Task',27,weight='bold')
    for g,phase in enumerate(['Approach','Pick','Place']):
        x=left+g*(group_w+between)
        rect(x,127,group_w,43,['#EDF4FB','#F3EFF8','#ECF6F2'][g])
        rect(x,127,group_w,5,colors[g]);text(x+group_w/2,135,phase,31,color=colors[g],weight='bold',ha='center')
        if g<2:line(x+group_w+between/2,127,footer-8)
    for y,heading,sub,color in [
        (group_a,'A   DOM training demonstrations','Hugging Face · hzxie/DOM · 20k training split','#EDF1F5'),
        (group_b,'B   DynamicVLA + KR','Isaac Sim · selected successful evaluation rollouts','#E9F3F9')]:
        rect(30,y,W-60,46,color)
        text(44,y+8,heading,30,weight='bold')
        text(W-47,y+12,sub,23,color=muted,ha='right')
    for i,row in enumerate(config['sources']):
        y=(start_a if i<3 else start_b)+(i%3)*row_h
        text(38,y+69,row['object'],34,weight='bold')
        text(38,y+116,'→ '+row['container'],25,color=muted)
        small=f'episode {row["episode_index"]:06d}' if i<3 else row.get('dimension', 'CR' if i<5 else 'SR')+' condition'
        text(39,y+163,small,21,color=muted)
        for j,frame in enumerate(row['frames']):
            g=j//2;col=j%2;x=left+g*(group_w+between)+col*(F+within)
            ax=fig.add_axes([x/W,1-(y+IH)/H,F/W,IH/H])
            ax.imshow(Image.open(out/frame['path']),interpolation='none');ax.set_axis_off()
            rect(x,y+IH+2,F,31,'#F4F6F8')
            text(x+F/2,y+IH+6,f't = {frame["clip_time_s"]:.2f} s',23,color=muted,ha='center')
        if i%3<2:
            fig.add_artist(Line2D([30/W,(W-30)/W],[1-(y+row_h-10)/H]*2,transform=fig.transFigure,color='#E4E9EE',linewidth=.5))
    text(34,footer,'Top: expert demonstrations. Bottom: velocity-conditioned DynamicVLA + KR, validation-best epoch 5.',23,color=muted)
    text(34,footer+35,'Independent scenes; examples illustrate successful behavior. Timestamps are elapsed video time.',23,color=muted)
    for ext in ['png','pdf','svg']:
        fig.savefig(out/f'manipulation_dynamicvla_kr_six_rows.{ext}',dpi=dpi,facecolor='white')
    plt.close(fig)
    Image.open(out/'manipulation_dynamicvla_kr_six_rows.png').resize((1260,H//2),Image.Resampling.LANCZOS).save(out/'preview.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=REPO/'scratch/dom45/figures/manipulation-six-rows-20261110')
    args=parser.parse_args();out=args.root.resolve();config=json.loads((out/'storyboard_selection.json').read_text())
    split=REPO/'scratch/dom45/training/krv100-fresh20k-h200-20261029/family/split.json'
    train=set(json.loads(split.read_text())['splits']['train']);assert len(train)==20000
    config['training_split_provenance']={'path':str(split.relative_to(REPO)),'sha256':sha(split),'training_episode_count':len(train)}
    for row in config['sources']:
        if row['kind']=='training_demonstration':
            assert row['episode_index'] in train
            row['confirmed_in_20k_training_split']=True
            row['source_url']=f'https://huggingface.co/datasets/{config["dataset"]}/resolve/{config["dataset_revision"]}/{row["huggingface_path"]}'
        else:
            path=REPO/row['path'];phase=path.parents[3];protocol=path.parents[5]/'protocol.json'
            checkpoint=json.loads(protocol.read_text())['checkpoint']
            assert checkpoint['progress']['epoch']==5 and 'KRV100' in checkpoint['label']
            scenario=path.stem.rsplit('-',3)[0]
            videos=sorted((phase/'videos').rglob(scenario+'-*.mp4'))
            assert len(videos)==20
            trial=videos.index(path)+1
            matches=[json.loads(s) for s in (phase/'trial_results.jsonl').read_text().splitlines() if s.strip()]
            result=next(r for r in matches if r['env_name']==scenario and r['trial']==trial)
            assert result['success'] and result['status']==0
            row.update(evaluation_scenario=scenario,global_trial=trial,recorded_result=result,
                checkpoint_provenance={'protocol':str(protocol.relative_to(REPO)),'protocol_sha256':sha(protocol),
                    'label':checkpoint['label'],'checkpoint_sha256':checkpoint['checkpoint_sha256'],
                    'config_sha256':checkpoint['config_sha256'],'progress':checkpoint['progress']})
    config=extract(config,out);render(config,out)
    config.update(builder=str(Path(__file__).relative_to(REPO)),builder_sha256=sha(__file__),
        image_processing='Exact decoded RGB frames; opposite-view tile selection for simulator videos. No synthesized or retouched scene content.',
        outputs={f'manipulation_dynamicvla_kr_six_rows.{ext}':sha(out/f'manipulation_dynamicvla_kr_six_rows.{ext}') for ext in ['png','pdf','svg']})
    (out/'frame_manifest.json').write_text(json.dumps(config,indent=2)+'\n')
    names=', '.join(row['object'].lower() for row in config['sources'][:3])
    (out/'caption.txt').write_text('Dynamic manipulation with DynamicVLA + KR. Three DOM training demonstrations from Hugging Face (top) and three selected successful Isaac Sim policy rollouts (bottom) illustrate approach, pick and place for '+names+' tasks. Dataset examples belong to the recorded 20k training split; policy examples use the fresh-20k, validation-best epoch-5 KRV100 checkpoint, with neural velocity conditioning and privileged simulator-state velocity. Rows show independent scenes, not matched initial configurations. Each stage contains two unaltered video frames. Timestamps are elapsed encoded video time, not inference latency or wall-clock task duration. Selected successes are qualitative illustrations and do not summarize success rates.\n')
    print(json.dumps({'rows':6,'frames':36,'output':str(out/'manipulation_dynamicvla_kr_six_rows.png')},indent=2))


if __name__=='__main__':main()
