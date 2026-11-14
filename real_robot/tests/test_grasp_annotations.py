import cv2
import numpy as np
from real_robot.grasp_annotations import lift_intervals,visual_concurrence,review_decisions
from real_robot.grasp_phase import supports_visible_holding


def episode():
    rows=[];measured=[]
    for i in range(50):
        grip=1. if i<10 or i>=40 else -.7
        z=.01 if i<18 else min(.12,.01+(i-18)*.008)
        rows.append({'action':[0,0,0,0,0,0,grip],'observation.state':[0,0,z,0,0,0]})
        measured.append(grip+.1 if grip<0 else grip)
    return rows,np.array(measured)


def test_success_candidate_needs_measured_closure_and_lift_and_stops_at_release():
    rows,measured=episode();result=lift_intervals(rows,measured)
    assert len(result)==1
    candidate=result[0]
    assert candidate['start_frame']>candidate['close_frame']
    assert candidate['lift_m']>=.02
    assert candidate['end_frame']==40
    assert not lift_intervals(rows,np.ones_like(measured))
    assert not lift_intervals(rows,np.full_like(measured,np.nan))
    for row in rows:row['observation.state'][2]=.01
    assert not lift_intervals(rows,measured)


def test_live_holding_support_allows_object_obstructed_jaws_but_requires_lift_and_close_command():
    assert supports_visible_holding(.50,.40,.03)  # Typical orange/ball aperture.
    assert supports_visible_holding(.15,.05,.03)  # Thin marker.
    assert not supports_visible_holding(.50,1.,.03)
    assert not supports_visible_holding(.80,.40,.03)
    assert not supports_visible_holding(.50,.40,.01)
    assert not supports_visible_holding(float('nan'),.40,.03)


def test_reopening_trend_cuts_interval_before_full_open():
    rows,measured=episode()
    for i in range(33,40):rows[i]['action'][-1]=-.45
    assert lift_intervals(rows,measured)[0]['end_frame']==33


def test_single_lift_spike_does_not_confirm_a_grasp():
    rows,measured=episode()
    for row in rows:row['observation.state'][2]=.01
    rows[25]['observation.state'][2]=.10
    assert not lift_intervals(rows,measured)


def wrist(x=192):
    image=np.zeros((384,384,3),np.uint8)
    cv2.circle(image,(x,320),18,(0,150,255),-1)
    return image


def test_visual_attachment_requires_persistent_object_relative_to_wrist():
    assert visual_concurrence([wrist(190),wrist(194),wrist(192)],'orange')[0]
    assert not visual_concurrence([wrist(130),wrist(190),wrist(250)],'orange')[0]
    assert not visual_concurrence([wrist(),np.zeros((384,384,3),np.uint8),wrist()],'orange')[0]
    assert not visual_concurrence([wrist(),wrist(),wrist()],'marker')[0]


def test_visual_review_can_accept_a_conservative_rejection_without_rewriting_automatic_output(tmp_path):
    import json
    original={'0':{'held_intervals':[]}}
    (tmp_path/'annotations.json').write_text(json.dumps(original))
    (tmp_path/'review.json').write_text(json.dumps({'episode_reports':[{'episode_index':0,'outcome_source':'saved_success',
        'candidates':[{'start_frame':10,'end_frame':20,'contact_sheet':'sheets/page-000.jpg'}]}]}))
    decisions=[{'episode_index':0,'start_frame':10,'contact_sheet':'sheets/page-000.jpg','decision':'accept'}]
    path=tmp_path/'decisions.json';path.write_text(json.dumps(decisions))
    report=review_decisions(tmp_path,path)
    reviewed=json.loads((tmp_path/'annotations_reviewed.json').read_text())['0']
    assert reviewed['held_intervals']==[[10,20]] and reviewed['manual_reviewed_intervals']==[[10,20]]
    assert json.loads((tmp_path/'annotations.json').read_text())==original
    assert report['unresolved_episodes']==[]
