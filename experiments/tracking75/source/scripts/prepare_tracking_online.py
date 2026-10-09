#!/usr/bin/env python3
"""Verify exact collection assets and prepare a paired live-tracking manifest."""
import argparse
import json
from pathlib import Path
import subprocess
import urllib.request
import numpy as np
import pyarrow.parquet as pq
from tracking_online_common import sha256, write_json
from tracking_checkpoint import checkpoint_files
from tracking_variants import VARIANT_LABELS, checkpoints_from_args


def main(a):
    collection=json.loads((a.dataset/'meta/collection.json').read_text())
    cfg=collection['provenance']['config'];layout=collection['provenance']['layout']
    scene=a.scene_dir/Path(layout['scene_usd']).name
    obj=a.object_dir/'apple/apple01.usd'
    for label,path in [('scene',scene),('object',obj)]:
        if not path.is_file():
            raise FileNotFoundError(f'Missing exact tracking {label}: {path}. Set SCENE_DIR / OBJECT_DIR to the original assets; benchmark scene substitution is not IID.')
        expected=layout[f'{label}_usd_sha256']
        if sha256(path)!=expected:
            raise ValueError(f'{label} hash differs from dataset provenance: {path}')
        layout[f'{label}_usd']=str(path.resolve())
    cfg['scene'].update(device='cuda:0',headless=True)
    cfg['collection']['purpose']='evaluation'
    if a.robot_usd:cfg['scene']['robot_usd']=str(a.robot_usd.resolve())
    split=json.loads((a.dataset/'meta/split_assignments.json').read_text())
    ids=[ep for part in split.values() for ep in part]
    if len(ids)!=len(set(ids)) or not split['test']:
        raise ValueError('Overlapping or empty evaluation split')
    episodes=split['test'][:a.max_episodes] if a.max_episodes else split['test']
    records=[]
    for ep in episodes:
        path=a.dataset/f'diagnostics/episode_{ep:06d}/episode.json'
        if path.is_file():data=json.loads(path.read_text())
        else:
            url=f'https://huggingface.co/datasets/adi2440/dom-franka-tracking/raw/{a.dataset_revision}/diagnostics/episode_{ep:06d}/episode.json'
            data=json.load(urllib.request.urlopen(url))
        records.append({'episode':ep,'trajectory':data['trajectory']})
    speeds=[]
    for ep in split['train']:
        tab=pq.read_table(a.dataset/f'data/chunk-000/episode_{ep:06d}.parquet',columns=['observation.environment_state']).to_pandas()
        speeds.append(np.linalg.norm(np.stack(tab.iloc[:,0])[:,6:9],axis=-1).max())
    models=checkpoints_from_args(a)
    from types import SimpleNamespace
    from tracking_variants import validate_variant
    checkpoint_metadata={}
    for variant,root in models.items():
        resolved,config,weights,payload=checkpoint_files(root,a.require_complete_training)
        validate_variant(SimpleNamespace(**payload),variant)
        if payload['chunk_size'] not in (20,50) or payload['n_action_steps']!=20:
            raise ValueError('Expected 20 execution steps and 20/50 predicted actions')
        checkpoint_metadata[variant]={'weights_sha256':sha256(weights),'config_sha256':sha256(config),
            'chunk_size':payload['chunk_size'],'selected_weights':str(weights)}
    commit=subprocess.check_output(['git','-C',str(a.collector),'rev-parse','HEAD'],text=True).strip()
    result={'dataset_revision':a.dataset_revision,'config':cfg,'layout':layout,'episodes':records,
        'speed_multipliers':[1.,1.5,2.],'train_max_speed_mps':float(max(speeds)),
        'baseline':str(a.baseline.resolve()),'residual':str(a.residual.resolve()),'collector_commit':commit,
        'models':{name:str(path.resolve()) for name,path in models.items()},'model_labels':VARIANT_LABELS,
        'checkpoint_metadata':checkpoint_metadata,
        'note':'OOD time scaling increases velocity by s and acceleration by s squared; identical spatial paths.'}
    write_json(a.output,result)
    guide=Path(__file__).resolve().parents[1]/('slurm/README_tracking_four.md' if 'velocity' in models else 'slurm/README_tracking_three_online.md')
    (a.output.parent/'README.md').write_text('# Live tracking evaluation artifacts\n\n'
        'Exact checkpoint paths, hashes, held-out trajectories and speed multipliers are in `manifest.json`. '
        'This directory is complete only after its simulation/summary stages finish.\n\n'+guide.read_text())
    print(f'PREPARED {len(episodes)} paired episodes x 3 speeds x {len(models)} models: {a.output}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','scene-dir','object-dir','collector','baseline','residual','output'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--coordinate',type=Path)
    p.add_argument('--velocity',type=Path)
    p.add_argument('--require-complete-training',action='store_true')
    p.add_argument('--robot-usd',type=Path);p.add_argument('--max-episodes',type=int,default=0)
    p.add_argument('--dataset-revision',default='5c46a9f26b5f85edafa53885187cc03fa8706219')
    a=p.parse_args()
    if a.max_episodes<0:p.error('--max-episodes must be >=0')
    main(a)
