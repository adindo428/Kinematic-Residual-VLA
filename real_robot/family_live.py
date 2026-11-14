"""Live input/selection contract for the native real-data family checkpoints."""
import json
import hashlib
from pathlib import Path
import numpy as np

FAMILIES=('smolvla_baseline','smolvla_kr','vlash_baseline','vlash_kr','pi05_baseline','pi05_kr')


def select_family_checkpoint(campaign_path,family,selection='best'):
    campaign_path=Path(campaign_path).resolve();campaign=json.loads(campaign_path.read_text())
    entry=next(e for e in campaign['entries'] if e['family']==family)
    training=campaign_path.parent/'runs'/family
    status=json.loads((training/'status.json').read_text())
    stopped=status.get('state')=='stopped'
    release=None
    if stopped:
        path=training/'evaluation_release.json'
        release=json.loads(path.read_text()) if path.exists() else {}
        if not (release.get('user_requested') and release.get('training_process_exited') and release.get('snapshots_verified')):
            raise ValueError('Stopped training requires a verified evaluation release: '+family)
    elif status.get('state')!='completed':raise ValueError('Real experiments require completed training and final evaluation: '+family)
    if selection not in ('best','last'):raise ValueError('Unknown checkpoint selection')
    pointer=json.loads((training/('best_checkpoint.json' if selection=='best' else 'latest_checkpoint.json')).read_text())
    source=Path(pointer['path']).resolve()
    if source.parent!=training.resolve() or not source.is_file():raise ValueError('Invalid selected checkpoint location')
    if selection=='best' and source.name not in ('best.pt','epoch-000-initial.pt'):
        raise ValueError('Best selection must reference the selected inference checkpoint')
    if selection=='last' and not pointer.get('completed_epoch'):raise ValueError('Partial recovery is not an evaluation checkpoint')
    if release is not None:
        frozen=release.get('checkpoints',{}).get(selection,{})
        if frozen.get('path')!=str(source) or frozen.get('epoch')!=pointer['epoch']:
            raise ValueError('Stopped-run checkpoint selection changed after release')
        digest=hashlib.sha256()
        with source.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
        if digest.hexdigest()!=frozen.get('sha256'):raise ValueError('Stopped-run checkpoint bytes changed after release')
    record=dict(pointer,selection=selection,source=str(source),family=family,
                velocity_conditioning=entry.get('velocity_conditioning'),analytical_kr=family.endswith('_kr'),
                object_inputs_used_by_policy=family_object_inputs_required(family,entry.get('velocity_conditioning')),
                neural_observation_offsets=[0],action_horizon=20,action_hz=25)
    return source,record,campaign


def family_object_inputs_required(family,velocity_conditioning=None):
    if family not in FAMILIES:raise ValueError('Unknown native family')
    return family.endswith('_kr') or bool(velocity_conditioning)


def current_family_inputs(payload,object_inputs_required=True):
    """Select only the current measured observation from two-frame transport."""
    state=np.asarray(payload['state'])
    if state.shape!=(2,6) or not np.isfinite(state).all():raise ValueError('Expected two finite robot-state observations')
    if object_inputs_required:
        environment=np.asarray(payload['environment']);required=np.asarray(payload['required']);valid=np.asarray(payload['valid'])
        if environment.shape!=(2,9) or required.shape!=(2,) or valid.shape!=(2,):
            raise ValueError('Expected two object observations and quality masks')
        if required.dtype!=np.bool_ or valid.dtype!=np.bool_:raise ValueError('Velocity quality masks must be boolean')
        if not np.isfinite(environment).all():raise ValueError('Nonfinite live object observation')
        if required[-1] and not valid[-1]:raise ValueError('Current required object velocity is invalid')
    for role in ['front','wrist']:
        image=np.asarray(payload[role])
        if image.ndim!=4 or image.shape[0]!=2 or image.shape[-1]!=3 or image.dtype!=np.uint8:
            raise ValueError('Expected two uint8 BGR camera observations')
    result=dict(state=state[-1].copy(),required=False)
    if object_inputs_required:
        result.update(velocity=environment[-1,6:9].copy() if required[-1] else np.zeros(3,dtype=environment.dtype),required=bool(required[-1]))
    return result


def family_batch(payload,meta,spec,device='cuda',object_inputs_required=True):
    import torch
    from .image_preprocessing import transform
    current=current_family_inputs(payload,object_inputs_required)
    state=torch.as_tensor(current['state'],dtype=torch.float32,device=device)[None]
    result={'observation.state':state,'reference_state':state.clone(),
            'task':[meta['instruction']],'offset':torch.zeros(1,dtype=torch.long,device=device),
            # Native preprocessing expects these fields; no target is read by sampling.
            'action':torch.zeros(1,20,7,device=device),'action_is_pad':torch.zeros(1,20,dtype=torch.bool,device=device)}
    if object_inputs_required:
        result.update(velocity=torch.as_tensor(current['velocity'],dtype=torch.float32,device=device)[None],
            velocity_required=torch.tensor([current['required']],dtype=torch.bool,device=device))
    for role,key in [('front','observation.images.opst_cam'),('wrist','observation.images.wrist_cam')]:
        image=transform(payload[role][-1],spec,role)[:,:,::-1].copy()
        result[key]=torch.from_numpy(image).permute(2,0,1)[None].to(device=device,dtype=torch.float32)/255
    return result
