import json
import cv2
import numpy as np
import pytest
from real_robot.automatic_mask import propose_box
from real_robot import mask_prompt
from types import SimpleNamespace


def profile(tmp_path):
    root=tmp_path/'profile';folder=root/'prompt_front';folder.mkdir(parents=True)
    reference=np.full((120,160,3),40,np.uint8);mask=np.zeros((120,160),np.uint8)
    cv2.circle(reference,(60,60),12,(0,130,255),-1);cv2.circle(mask,(60,60),12,255,-1)
    cv2.imwrite(str(folder/'reference.png'),reference);cv2.imwrite(str(folder/'mask.png'),mask)
    (root/'front-prompt.json').write_text(json.dumps(dict(timing={'camera_serial':'camera'},box_xywh=[48,48,25,25])))
    return root


def test_automatic_box_tracks_reference_colour_at_a_new_location(tmp_path):
    root=profile(tmp_path);live=np.full((120,160,3),40,np.uint8);cv2.circle(live,(105,75),12,(0,130,255),-1)
    box,info=propose_box(live,root,'front','camera');x,y,w,h=box
    assert x<=93 and x+w>=117 and y<=63 and y+h>=87
    assert info['candidate_count']==1
    with pytest.raises(ValueError,match='another camera'):propose_box(live,root,'front','wrong')


def test_absent_or_ambiguous_objects_are_not_auto_selected(tmp_path):
    root=profile(tmp_path);live=np.full((120,160,3),40,np.uint8)
    with pytest.raises(ValueError,match='0 plausible'):propose_box(live,root,'front','camera')
    cv2.circle(live,(45,60),12,(0,130,255),-1);cv2.circle(live,(110,60),12,(0,130,255),-1)
    with pytest.raises(ValueError,match='2 plausible'):propose_box(live,root,'front','camera')


def test_auto_seed_uses_efficienttam_without_roi_window(monkeypatch,tmp_path):
    root=profile(tmp_path);out=tmp_path/'session';out.mkdir();live=np.full((120,160,3),40,np.uint8);cv2.circle(live,(100,60),12,(0,130,255),-1)
    monkeypatch.setattr(mask_prompt.cv2,'selectROI',lambda *a,**k:pytest.fail('unexpected manual selection'))
    model=SimpleNamespace(seed=lambda *args:(np.ones((120,160),np.uint8),{'seed':True}))
    _,info=mask_prompt.seed_live(model,'front',dict(bgr=live,timing=dict(camera_serial='camera',sensor_timestamp_s=1.)),out,root,retry=True,automatic=True)
    assert info['selection_mode']=='automatic' and not (out/'MASK_PENDING').exists()


def test_auto_ambiguity_falls_back_to_manual_selection(monkeypatch,tmp_path):
    root=profile(tmp_path);out=tmp_path/'session';out.mkdir();live=np.zeros((120,160,3),np.uint8)
    monkeypatch.setattr(mask_prompt.cv2,'selectROI',lambda *a,**k:(10,10,20,20));monkeypatch.setattr(mask_prompt.cv2,'destroyWindow',lambda *a:None)
    model=SimpleNamespace(seed=lambda *args:(np.ones((120,160),np.uint8),{'seed':True}))
    _,info=mask_prompt.seed_live(model,'front',dict(bgr=live,timing=dict(camera_serial='camera',sensor_timestamp_s=1.)),out,root,retry=True,automatic=True)
    assert info['selection_mode']=='manual_or_explicit_box' and '0 plausible' in info['automatic_proposal_failure']


def test_reference_matching_tolerates_a_small_lighting_hue_shift(tmp_path):
    root=profile(tmp_path);live=np.full((120,160,3),40,np.uint8)
    colour=cv2.cvtColor(np.array([[[20,255,205]]],np.uint8),cv2.COLOR_HSV2BGR)[0,0]
    cv2.circle(live,(100,75),12,tuple(map(int,colour)),-1)
    box,_=propose_box(live,root,'front','camera')
    assert box[0]<100<box[0]+box[2] and box[1]<75<box[1]+box[3]
