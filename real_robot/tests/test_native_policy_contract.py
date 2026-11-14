import json
from pathlib import Path
import random
import numpy as np
import pytest
from safetensors.numpy import save_file
from real_robot.native_policy_contract import training_instruction,verify_native_contract


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_live_instruction_is_an_actual_training_template_and_preserves_rng(tmp_path):
    write(tmp_path/'meta/tasks.jsonl',{'task':json.dumps(dict(task='place',objects='rolling orange',containers='box'))})
    state=random.getstate()
    assert training_instruction(tmp_path,'orange')=='pick up the rolling orange and deposit in the box.'
    assert random.getstate()==state
    with pytest.raises(ValueError,match='absent'):training_instruction(tmp_path,'marker')


def setup_contract(tmp_path):
    import hashlib
    dataset=tmp_path/'dataset';source=tmp_path/'domv/epoch-008';evaluation=tmp_path/'evaluation/domv'
    write(dataset/'meta/quality_policy.json',{'test':'policy'})
    stats={'observation.state':{'mean':[0.]*6,'std':[1.]*6},'action':{'mean':[0.]*7,'std':[1.]*7},
        'observation.environment_state':{'mean':[0.]*9,'std':[1.]*9}}
    write(dataset/'meta/train_stats.json',stats)
    cfg=dict(condition_on_object_velocity=True,enable_residual_flow=False,n_obs_steps=2,chunk_size=20,
        use_delta_action=True,adapt_to_pi_aloha=False,euler_delta_convention='wrapped_xyz_v1',require_tracking_quality=True,
        residual_phase_gate_mode='constant',delta_timestamps={'observation':[-2,0],'action':list(range(20))},
        velocity_input_mean=[0.]*3,velocity_input_std=[1.]*3,use_checkpoint_normalization=False)
    write(source/'config.json',cfg);write(evaluation/'config.json',dict(cfg,use_checkpoint_normalization=True))
    write(source/'provenance.json',dict(completed_epoch=True,epoch=8,dataset=str(dataset),
        quality_policy_sha256=hashlib.sha256((dataset/'meta/quality_policy.json').read_bytes()).hexdigest()))
    buffers={}
    for prefix,key in [('normalize_inputs','observation.state'),('normalize_targets','action'),('unnormalize_outputs','action')]:
        for moment in ['mean','std']:buffers[prefix+'.buffer_'+key.replace('.','_')+'.'+moment]=np.array(stats[key][moment],dtype=np.float32)
    save_file(buffers,source/'model.safetensors');(evaluation/'model.safetensors').symlink_to(source/'model.safetensors')
    write(evaluation/'selection.json',dict(source=str(source)))
    return evaluation,dataset,dict(observation_history=[-2,0])


def test_saved_method_and_normalization_are_enforced(tmp_path):
    checkpoint,dataset,contract=setup_contract(tmp_path)
    result=verify_native_contract(checkpoint,'domv',dataset,contract)
    assert result['passed'] and max(result['normalization_max_abs_differences'].values())==0
    cfg=json.loads((checkpoint/'config.json').read_text());cfg['residual_phase_gate_mode']='causal_rules';write(checkpoint/'config.json',cfg)
    with pytest.raises(ValueError,match='changed trained configuration'):verify_native_contract(checkpoint,'domv',dataset,contract)


def test_causal_kr_contract_accepts_only_the_trained_gate_and_thresholds(tmp_path):
    checkpoint,dataset,contract=setup_contract(tmp_path)
    source=Path(json.loads((checkpoint/'selection.json').read_text())['source'])
    for directory in [source,checkpoint]:
        cfg=json.loads((directory/'config.json').read_text())
        cfg.update(enable_residual_flow=True,residual_phase_gate_mode='causal_rules',residual_phase_gate_config={})
        write(directory/'config.json',cfg)
    contract['methods']={'krv100':dict(residual_phase_gate_mode='causal_rules',residual_phase_gate_config={})}
    report=verify_native_contract(checkpoint,'krv100',dataset,contract)
    assert report['phase_gate_mode']=='causal_rules' and not report['constant_gate']
    contract['methods']['krv100']['residual_phase_gate_config']={'fade_s':.5}
    with pytest.raises(ValueError,match='thresholds'):verify_native_contract(checkpoint,'krv100',dataset,contract)


def test_normalization_mismatch_cannot_launch(tmp_path):
    checkpoint,dataset,contract=setup_contract(tmp_path)
    stats=json.loads((dataset/'meta/train_stats.json').read_text());stats['action']['mean'][0]=.07
    write(dataset/'meta/train_stats.json',stats)
    with pytest.raises(ValueError,match='normalization mismatch'):verify_native_contract(checkpoint,'domv',dataset,contract)


def test_geometry_check_rejects_different_camera_base_or_tcp(tmp_path):
    from real_robot.native_policy_contract import verify_live_geometry
    transform=np.eye(4).tolist();rig=dict(front_serial='front',T_base_front_candidate=transform)
    write(tmp_path/'meta/image_preprocessing.json',dict(front_serial='front'))
    write(tmp_path/'meta/camera.jsonl',dict(cameras=[dict(source_role='front',calibration=dict(T_base_front_candidate=transform))]))
    contract=dict(controller_contract=dict(tcp_offset_m=[.16,0,0]))
    assert verify_live_geometry(tmp_path,rig,contract)['passed']
    with pytest.raises(ValueError,match='TCP'):verify_live_geometry(tmp_path,dict(rig,validated_tcp_offset_m=[.2,0,0]),contract)
    with pytest.raises(ValueError,match='camera/base'):verify_live_geometry(tmp_path,dict(rig,T_base_front_candidate=None),contract)


def test_recorded_instruction_keeps_case_and_wording(tmp_path):
    text='Pick up the rolling orange and place it in the box.'
    write(tmp_path/'meta/tasks.jsonl',{'task':json.dumps(dict(task='place',objects='rolling orange',containers='box',instruction=text))})
    assert training_instruction(tmp_path,'orange',mode='recorded')==text
