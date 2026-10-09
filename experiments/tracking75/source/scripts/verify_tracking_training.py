#!/usr/bin/env python3
"""Verify completed training and validation reload before permitting evaluation."""
import argparse,json,os
from pathlib import Path
import torch
from tracking_checkpoint import load_checkpoint,checkpoint_files
from tracking_online_common import sha256,write_json
from tracking_variants import validate_variant,VARIANT_LABELS
from tracking_recorded import DEFAULT_DATASET,read_episode,make_batch,noise_for

def verify(root,variant,dataset=DEFAULT_DATASET,require_neural_velocity=False):
    root,config,weights,cfg=checkpoint_files(root)
    from types import SimpleNamespace
    validate_variant(SimpleNamespace(**cfg),variant,require_neural_velocity=require_neural_velocity)
    assert cfg.get('epoch')==75, f'{variant}: training not complete'
    assert cfg['chunk_size']==cfg['n_action_steps']==20
    assert all(cfg[k] is False for k in ('freeze_vision_model','freeze_connector','freeze_text_model'))
    state=torch.load(root/'training_state.pt',map_location='cpu',mmap=True,weights_only=False)
    assert state['epoch']==75;del state
    best=json.loads((root/'best_eval.json').read_text())
    assert best['split']=='validation' and best['metric']=='action_mse' and 1<=best['epoch']<=75
    assert best['weights']==weights.name and torch.isfinite(torch.tensor(best['value']))
    policy=load_checkpoint(root)
    for name,value in policy.state_dict().items():
        assert torch.isfinite(value).all(), f'nonfinite checkpoint tensor {name}'
    split=json.loads((dataset/'meta/split_assignments.json').read_text())
    ep=split['validation'][0];batch=make_batch(read_episode(dataset,ep),[10,30])
    with torch.inference_mode():pred=policy.predict_action_chunk(batch,noise=noise_for(ep,[10,30]))
    assert pred.shape==(2,20,7) and torch.isfinite(pred).all()
    marker={'status':'ready','variant':variant,'name':VARIANT_LABELS[variant],'checkpoint':str(root),
        'trained_epochs':75,'selection':best,'weights_sha256':sha256(weights),'config_sha256':sha256(config),
        'job_id':os.environ.get('SLURM_JOB_ID'),'validation_reload_shape':list(pred.shape),
        'neural_velocity_conditioning':bool(cfg.get('condition_on_object_velocity',False)),
        'checks':['75 epochs in config and optimizer state','validation-selected weights','full unfreezing',
                  'finite state tensors','strict checkpoint reload','finite validation inference'],
        'velocity_source':'recorded current velocity for validation; current simulator velocity for live evaluation'}
    write_json(root/'training_ready.json',marker)
    print('TRAINING VERIFIED',variant,root,flush=True)
    del policy;torch.cuda.empty_cache()
    return marker

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--variant',choices=VARIANT_LABELS,required=True);p.add_argument('--dataset',type=Path,default=DEFAULT_DATASET)
    p.add_argument('--require-neural-velocity',action='store_true')
    a=p.parse_args();verify(a.checkpoint,a.variant,a.dataset,a.require_neural_velocity)
