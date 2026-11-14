import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workstation'))
from real_robot.policy_camera import publish_frame
from real_robot.policy_stream import live_camera,live_object
from real_robot.evaluate import perception_ready,policy_object_inputs_required


def test_plain_dom_checkpoint_requires_no_object_inputs_but_conditioned_policies_do(tmp_path):
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(condition_on_object_velocity=False,enable_residual_flow=False)))
    assert not policy_object_inputs_required('dom',tmp_path)
    assert policy_object_inputs_required('domv',tmp_path) and policy_object_inputs_required('krv100',tmp_path)
    config.write_text(json.dumps(dict(condition_on_object_velocity=True,enable_residual_flow=False)))
    assert policy_object_inputs_required('dom',tmp_path)


def test_rgb_front_feed_is_ready_without_track_or_velocity_and_never_fabricates_geometry(tmp_path,monkeypatch):
    from real_robot import evaluate
    image=np.zeros((480,640,3),np.uint8)
    record=publish_frame(tmp_path,image,dict(host_arrival_monotonic_s=1.),'rig',1.02)
    monkeypatch.setattr(evaluate.time,'monotonic',lambda:1.05)
    assert perception_ready(tmp_path,'rig',object_tracking_required=False)
    assert not perception_ready(tmp_path,'rig',object_tracking_required=True)
    assert record['track'] is None and record['velocity_report']['status']=='not_used_by_policy'
    assert 'position_m' not in record and 'velocity_mps' not in record['velocity_report']
    with np.load(tmp_path/'policy_front.npz',allow_pickle=False) as saved:np.testing.assert_array_equal(saved['bgr'],image)
    with pytest.raises(ValueError):live_object(record,1.05,'rig')
    for stamp,rig in [(1.25,'rig'),(1.05,'wrong'),(.99,'rig')]:
        with pytest.raises(ValueError):live_camera(record,stamp,rig)


def test_plain_dom_live_batch_uses_only_rgb_and_robot_state():
    from real_policy_worker import prepare_live_batch
    payload=dict(front=np.zeros((2,480,640,3),np.uint8),wrist=np.zeros((2,480,640,3),np.uint8),state=np.ones((2,6),np.float32))
    spec=dict(schema='krvla_image_preprocessing_v1',source_size_wh=[640,480],target_size_wh=[384,384],padding='left_top',pad_value=0,front_crop_xywh=[0,0,640,480])
    cfg=SimpleNamespace(condition_on_object_velocity=False,enable_residual_flow=False)
    batch=prepare_live_batch(cfg,payload,{'instruction':'pick the marker'},spec,device='cpu')
    assert set(batch)=={'observation.images.opst_cam','observation.images.wrist_cam','observation.state','task'}
    assert batch['observation.state'].shape==(1,2,6)
    # Even invalid object fields cannot affect the unconditioned model input.
    payload.update(environment=np.full((2,9),np.nan),required=np.ones(2,bool),valid=np.zeros(2,bool))
    again=prepare_live_batch(cfg,payload,{'instruction':'pick the marker'},spec,device='cpu')
    for key in batch:
        if torch.is_tensor(batch[key]):torch.testing.assert_close(batch[key],again[key])
    del payload['environment']
    with pytest.raises(KeyError):prepare_live_batch(SimpleNamespace(condition_on_object_velocity=True,enable_residual_flow=False),payload,{'instruction':'pick'},spec,device='cpu')


def test_live_causal_kr_requires_the_controller_gate_without_replacing_it():
    from real_policy_worker import prepare_live_batch
    cfg=SimpleNamespace(condition_on_object_velocity=True,enable_residual_flow=True,
        residual_phase_gate_mode='causal_rules',residual_phase_gate_key='dynamic_object.phase_gate',chunk_size=20)
    payload=dict(front=np.zeros((2,480,640,3),np.uint8),wrist=np.zeros((2,480,640,3),np.uint8),
        state=np.ones((2,6),np.float32),environment=np.zeros((2,9)),required=np.ones(2,bool),valid=np.ones(2,bool),
        phase_gate=np.linspace(1,0,20,dtype=np.float32)[:,None])
    spec=dict(schema='krvla_image_preprocessing_v1',source_size_wh=[640,480],target_size_wh=[384,384],padding='left_top',pad_value=0,front_crop_xywh=[0,0,640,480])
    batch=prepare_live_batch(cfg,payload,{'instruction':'same prompt'},spec,device='cpu')
    np.testing.assert_array_equal(batch['dynamic_object.phase_gate'][0].numpy(),payload['phase_gate'])
    payload['phase_gate'][2]=float('nan')
    with pytest.raises(ValueError,match='causal phase gate'):prepare_live_batch(cfg,payload,{'instruction':'same prompt'},spec,device='cpu')


def test_rgb_camera_process_publishes_without_depth_or_tracker(monkeypatch,tmp_path):
    from real_robot import policy_camera as mod
    rig=tmp_path/'rig.json';rig.write_text(json.dumps({'front_serial':'front-test'}));output=tmp_path/'feed'
    closed=[]
    class Camera:
        def __init__(self,serial,**kwargs):assert serial=='front-test'
        def read(self):
            return np.zeros((480,640,3),np.uint8),np.full((480,640),np.nan),dict(
                host_arrival_monotonic_s=1.,color_timestamp_s=1.,timestamp_domain='test'),None
        def close(self):closed.append(True)
    monkeypatch.setattr(mod,'RGBDCamera',Camera)
    monkeypatch.setattr(mod.time,'monotonic',lambda:1.02)
    monkeypatch.setattr(mod.cv2,'imshow',lambda *a:None)
    monkeypatch.setattr(mod.cv2,'waitKey',lambda *a:ord('q'))
    monkeypatch.setattr(mod.cv2,'destroyAllWindows',lambda:None)
    monkeypatch.setattr(sys,'argv',['policy_camera','--rig',str(rig),'--output',str(output),'--preview'])
    mod.main()
    with np.load(output/'policy_front.npz',allow_pickle=False) as z:
        frame=json.loads(str(z['metadata']));assert frame['running'] and frame['track'] is None
    assert closed==[True] and json.loads((output/'summary.json').read_text())['frames']==1
