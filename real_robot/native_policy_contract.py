"""Verify native DOM training/evaluation semantics before opening devices."""
import hashlib
import json
from pathlib import Path
import random

import numpy as np


OBJECTS={'orange':'rolling orange','striped_ball':'moving orange and white striped ball','marker':'moving green marker'}
# These only select the outer worker and restore normalization from saved weights.
RUNTIME_FIELDS={'enable_streaming','use_checkpoint_normalization','delta_timestamps'}


def read(path):return json.loads(Path(path).read_text())


def training_instruction(dataset, task, mode='generated'):
    from third_party.DynamicVLA.utils.instruction_generator import InstructionGenerator
    metadata=None
    for row in map(json.loads,(Path(dataset)/'meta/tasks.jsonl').read_text().splitlines()):
        candidate=row['task'];candidate=json.loads(candidate) if isinstance(candidate,str) else candidate
        if candidate.get('objects')==OBJECTS[task]:metadata=candidate;break
    if metadata is None:raise ValueError('Task absent from checkpoint training metadata: '+task)
    if mode=='recorded':
        if not metadata.get('instruction'):raise ValueError('Recorded instruction is missing')
        return metadata['instruction']
    if mode!='generated':raise ValueError('Unknown instruction mode')
    # Select an actual training template reproducibly, without changing global RNG.
    state=random.getstate()
    try:
        for seed in range(256):
            random.seed(seed);text=InstructionGenerator.generate_instruction(metadata)
            if text.startswith('pick up ') and ' and deposit in ' in text:return text
    finally:random.setstate(state)
    raise ValueError('No matching native place instruction')


def verify_native_contract(checkpoint, family, dataset, contract):
    from safetensors import safe_open
    checkpoint=Path(checkpoint).resolve();dataset=Path(dataset).resolve();cfg=read(checkpoint/'config.json')
    selection=checkpoint/'selection.json'
    source=Path(read(selection)['source']).resolve() if selection.exists() else checkpoint
    original=read(source/'config.json');provenance=read(source/'provenance.json')
    changed={key for key in set(cfg)|set(original) if cfg.get(key)!=original.get(key)}-RUNTIME_FIELDS
    if changed:raise ValueError('Evaluation changed trained configuration: '+', '.join(sorted(changed)))
    if (checkpoint/'model.safetensors').resolve()!=(source/'model.safetensors').resolve():
        raise ValueError('Evaluation does not use selected native checkpoint weights')
    if contract.get('instruction_mode','generated')!=provenance.get('instruction_mode','generated'):
        raise ValueError('Evaluation instruction mode differs from training checkpoint')
    if not provenance.get('completed_epoch'):raise ValueError('Selected checkpoint is not a completed epoch')
    if Path(provenance['dataset']).resolve()!=dataset:raise ValueError('Checkpoint dataset differs from live contract')
    quality_hash=hashlib.sha256((dataset/'meta/quality_policy.json').read_bytes()).hexdigest()
    if provenance['quality_policy_sha256']!=quality_hash:raise ValueError('Checkpoint dataset quality policy changed')
    expected={'dom':(False,False),'domv':(True,False),'krv100':(True,True)}[family]
    if (cfg['condition_on_object_velocity'],cfg['enable_residual_flow'])!=expected:
        raise ValueError('Native velocity/residual method mismatch')
    requirements=dict(n_obs_steps=2,chunk_size=20,use_delta_action=True,adapt_to_pi_aloha=False,
        euler_delta_convention='wrapped_xyz_v1',require_tracking_quality=True,residual_phase_gate_mode=contract.get('methods',{}).get(family,{}).get('residual_phase_gate_mode','constant'))
    if requirements['residual_phase_gate_mode'] not in ('constant','causal_rules'):raise ValueError('Unsupported trained gate mode')
    if requirements['residual_phase_gate_mode']=='causal_rules':
        if not expected[1]:raise ValueError('Causal residual gating requires KR')
        if cfg.get('residual_phase_gate_config',{})!=contract.get('methods',{}).get(family,{}).get('residual_phase_gate_config',{}):raise ValueError('Trained gate thresholds differ from deployment contract')
    for key,value in requirements.items():
        if cfg.get(key)!=value:raise ValueError('Unsupported trained/live contract: '+key)
    if contract['observation_history']!=[-2,0]:raise ValueError('Live observation history differs from training')
    if cfg.get('delta_timestamps')!={'observation':[-2,0],'action':list(range(20))}:
        raise ValueError('Live timestamp configuration differs from training')
    controller=contract.get('controller_contract',{})
    if controller and (controller['tcp_offset_m']!=[.16,0,0] or controller['fps']!=25):
        raise ValueError('Unsupported trained TCP or action clock')
    stats={k:{n:np.asarray(v,dtype=np.float32) for n,v in values.items()} for k,values in read(dataset/'meta/train_stats.json').items()}
    for value in stats.values():
        mask=(np.abs(value['mean'])<1e-6)&(np.abs(value['std'])<1e-6);value['std'][mask]=1.
    differences={}
    with safe_open(checkpoint/'model.safetensors',framework='np') as tensors:
        for prefix,key in [('normalize_inputs','observation.state'),('normalize_targets','action'),('unnormalize_outputs','action')]:
            for moment in ['mean','std']:
                name=prefix+'.buffer_'+key.replace('.','_')+'.'+moment
                actual=tensors.get_tensor(name);target=stats[key][moment]
                if actual.shape!=target.shape or not np.allclose(actual,target,atol=1e-7,rtol=1e-6):
                    raise ValueError('Saved train normalization mismatch: '+name)
                differences[name]=float(np.max(np.abs(actual-target)))
    if expected[0]:
        np.testing.assert_allclose(cfg['velocity_input_mean'],stats['observation.environment_state']['mean'][6:9],rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(cfg['velocity_input_std'],np.maximum(stats['observation.environment_state']['std'][6:9],1e-6),rtol=1e-6,atol=1e-7)
    return dict(passed=True,family=family,epoch=provenance['epoch'],source=str(source),dataset=str(dataset),
        normalization_max_abs_differences=differences,constant_gate=cfg['residual_phase_gate_mode']=='constant',phase_gate_mode=cfg['residual_phase_gate_mode'],observation_offsets=[-2,0],
        action_hz=25,action_horizon=20,velocity_conditioning=expected[0],residual_flow=expected[1],
        interpretation='Model input/decode contract; physical teleoperation and IK actuation are distinct.')


def verify_live_geometry(dataset, rig, contract):
    """Check stored coordinate identities; does not measure physical calibration."""
    dataset=Path(dataset);spec=read(dataset/'meta/image_preprocessing.json')
    if spec['front_serial']!=rig['front_serial']:raise ValueError('Live front camera differs from training')
    trained_tcp=contract.get('controller_contract',{}).get('tcp_offset_m',[.16,0,0])
    live_tcp=rig.get('validated_tcp_offset_m',[.16,0,0])
    if not np.array_equal(trained_tcp,live_tcp):raise ValueError('Live TCP offset differs from training')
    matched=0;total=0
    for row in map(json.loads,(dataset/'meta/camera.jsonl').read_text().splitlines()):
        front=next(c for c in row['cameras'] if c['source_role']=='front')
        total+=1
        if front['calibration'].get('T_base_front_candidate')==rig.get('T_base_front_candidate'):matched+=1
    if not total or matched!=total:raise ValueError('Live camera/base transform differs from training camera metadata')
    return dict(passed=True,front_serial=rig['front_serial'],tcp_offset_m=live_tcp,
        matching_camera_base_transforms=matched,training_camera_records=total,
        physical_calibration_accuracy_verified=False)
