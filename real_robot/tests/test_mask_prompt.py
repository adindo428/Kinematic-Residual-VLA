import json
from types import SimpleNamespace
import numpy as np
import pytest
from real_robot import mask_prompt
from real_robot.evaluate import perception_ready


def test_empty_seed_can_be_redrawn_and_previous_profile_backed_up(monkeypatch,tmp_path):
    boxes=iter([(10,10,20,20),(12,12,25,25)]);seeds=[];saved=[]
    monkeypatch.setattr(mask_prompt.cv2,'selectROI',lambda *a,**k:next(boxes))
    monkeypatch.setattr(mask_prompt.cv2,'destroyWindow',lambda *a:None)
    monkeypatch.setattr(mask_prompt,'has_prompt',lambda root,role:root==tmp_path/'profile')
    monkeypatch.setattr(mask_prompt,'save_prompt',lambda src,dst,role:saved.append((src,dst,role)))
    def seed(role,image,stamp,box):
        seeds.append(box)
        return (None if len(seeds)==1 else np.ones((60,60),np.uint8)),{'seed':True}
    frame=dict(bgr=np.zeros((60,60,3),np.uint8),timing={'sensor_timestamp_s':1.})
    mask,_=mask_prompt.seed_live(SimpleNamespace(seed=seed),'front',frame,tmp_path,tmp_path/'profile',retry=True)
    assert mask is not None and len(seeds)==2
    assert saved==[(tmp_path/'profile',tmp_path/'previous_profile','front'),(tmp_path,tmp_path/'profile','front')]
    assert not (tmp_path/'MASK_PENDING').exists()
    assert json.loads((tmp_path/'prompt_status.json').read_text())['state']=='selected'


def test_cancelled_selection_remains_blocked(monkeypatch,tmp_path):
    monkeypatch.setattr(mask_prompt.cv2,'selectROI',lambda *a,**k:(0,0,0,0))
    monkeypatch.setattr(mask_prompt.cv2,'destroyWindow',lambda *a:None)
    with pytest.raises(ValueError,match='cancelled'):
        mask_prompt.seed_live(SimpleNamespace(),'front',{'bgr':np.zeros((10,10,3),np.uint8)},tmp_path,retry=True)
    assert (tmp_path/'MASK_PENDING').exists()


def test_evaluation_waits_for_valid_geometry_and_completed_selection(monkeypatch,tmp_path):
    import real_robot.evaluate as evaluate
    monkeypatch.setattr(evaluate.time,'monotonic',lambda:1.1)
    (tmp_path/'policy_front.npz').touch()
    row=dict(schema='front_side_observation_v1',running=True,rig_sha256='r',output_frame='right_base_estimate',stale=False,
        timing={'oldest_host_arrival_monotonic_s':1.},
        track=dict(accepted=True,ready=True,frame='right_base_estimate',position_m=[.3,0,.1]),
        velocity_report=dict(status='estimated',velocity_mps=[0,0,0]))
    (tmp_path/'observation.json').write_text(json.dumps(row))
    assert perception_ready(tmp_path,'r')
    (tmp_path/'MASK_PENDING').touch();assert not perception_ready(tmp_path,'r')
    (tmp_path/'MASK_PENDING').unlink();row['velocity_report']['status']='unavailable'
    (tmp_path/'observation.json').write_text(json.dumps(row));assert not perception_ready(tmp_path,'r')
