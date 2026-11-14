"""Stationary declarations stay distinct from missing tracking and held motion."""
import json
import os
import subprocess
import sys
import hashlib
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

import numpy as np
import pytest
import torch

from real_robot.static_object import SCHEMA,AUTOMATIC_PHASE_METHOD,lock_position,observe_motion,quality_valid,annotate_automatic_phases
from real_robot.dom_dataset import align_episode,export_sessions,json_write,jsonl_write,jsonl_read
from real_robot.dataset_quality import phases,repair_tracking,process
from real_robot.dynamic_capture import Episodes
from real_robot.tests.test_dom_collection import make_session


def observation(position=(.3,0,.05),stamp=100.):
    return dict(stale=False,host_published_monotonic_s=stamp,
                fusion=dict(stale=False,position_output_m=list(position)))


def static_session(root):
    manifest,episode=make_session(root)
    manifest['static_object_collection']={'schema':SCHEMA}
    episode['static_object']=lock_position(observation(),100.)
    episode['events']={'lift':101.,'operator_visible_held':101.,'release':102.}
    json_write(root/'manifest.json',manifest)
    json_write(root/'episodes/episode_000000.json',episode)
    obs=jsonl_read(root/'observations.jsonl')
    for row in obs:
        row.update(stale=True,fusion=None,velocity_report=None)
    jsonl_write(root/'observations.jsonl',obs)
    return manifest,episode


def automatic_session(root):
    manifest,episode=static_session(root)
    manifest['static_object_collection']['phase_labeling']=AUTOMATIC_PHASE_METHOD
    episode['events']={}
    robot=jsonl_read(root/'robot.jsonl')
    for i,row in enumerate(robot):
        # Measured TCP descends, then lifts while the measured gripper is closed.
        z=.15-.004*i if i<25 else min(.05+.003*(i-25),.11)
        row['T_base_tcp_candidate'][2][3]=z
        aperture=100. if i<10 else (0. if i<62 else min(100.,(i-62)*15.))
        row['follower_calibrated']=[0.]*8+[aperture]
        row['commanded_calibrated'][-1]=aperture
    jsonl_write(root/'robot.jsonl',robot)
    json_write(root/'manifest.json',manifest)
    json_write(root/'episodes/episode_000000.json',episode)
    return manifest,episode


def test_each_attempt_needs_its_own_fresh_lock(tmp_path):
    ledger=Episodes(tmp_path,'striped_ball','train')
    first=lock_position(observation(),100.)
    ledger.begin(100.,first);ledger.finish('failed',100.5)
    with pytest.raises(ValueError,match='per-episode'):
        ledger.begin(101.,first)
    second=lock_position(observation((.4,0,.05),101.),101.)
    ledger.begin(101.,second)
    assert ledger.current['static_object']['position_m'][0]==.4


@pytest.mark.parametrize('change',[{'stale':True},{'fusion':None},{'host_published_monotonic_s':99.},
                                  {'fusion':{'stale':False,'position_output_m':[float('nan'),0,0]}}])
def test_lock_refuses_missing_or_stale_position_without_requiring_velocity(change):
    row=observation();row.update(change)
    with pytest.raises(ValueError,match='fresh object position'):lock_position(row,100.)


def test_occlusion_does_not_renew_seed_and_visible_motion_invalidates_permanently():
    context=lock_position(observation(),100.)
    assert not observe_motion(context,{'stale':True},110.)
    assert context['measurement_monotonic_s']==100.
    assert observe_motion(context,observation((.33,0,.05),110.),110.)
    assert context['invalidated_monotonic_s']==110.
    assert not observe_motion(context,observation(stamp=111.),111.)
    assert context['position_m']==[.3,0,.05]


def test_export_and_quality_accept_declared_static_without_tracker_but_not_after_release(tmp_path):
    root=tmp_path/'recording';manifest,episode=static_session(root)
    rows,_,quality=align_episode(root,manifest,episode)
    labels,required=phases(rows,quality,episode['events'])
    repaired,flags=repair_tracking(rows,quality,required=required)
    for row,q,label in zip(repaired,flags,labels):
        t=q['anchor_monotonic_s']
        np.testing.assert_allclose(row['observation.environment_state'][:3],[.3,0,.05])
        assert row['observation.environment_state'][6:9]==[0.,0.,0.]
        assert not q['position_current'] and not q['velocity_estimated']
        if t<101.:
            assert quality_valid(q) and q['tracking_usable']
        elif t<102.:
            assert label=='held_confirmed' and q['tracking_usable']
            assert q['velocity_input_source']=='neutral_after_confirmed_lift'
        else:
            assert not q['tracking_usable']
    # A static position is an explicit declaration, not a missing-data fallback.
    del manifest['static_object_collection']
    with pytest.raises(ValueError,match='missing robot.*object'):
        align_episode(root,manifest,episode)


def test_contact_or_displacement_ends_assumption_before_held_confirmation(tmp_path):
    root=tmp_path/'recording';manifest,episode=static_session(root)
    episode['events']['contact']=100.5
    rows,_,quality=align_episode(root,manifest,episode)
    _,required=phases(rows,quality,episode['events'])
    _,quality=repair_tracking(rows,quality,required=required)
    assert not any(q['tracking_usable'] for q in quality if 100.5<=q['anchor_monotonic_s']<101.)
    del episode['events']['contact']
    episode['static_object']['invalidated_monotonic_s']=100.4
    rows,_,quality=align_episode(root,manifest,episode)
    assert not any(quality_valid(q) for q in quality if q['anchor_monotonic_s']>=100.4)


@pytest.mark.parametrize('automatic',[False,True])
def test_processed_export_loads_in_dom_and_family_training(tmp_path,monkeypatch,automatic):
    # Capture uses the pinned older LeRobot namespace; the isolated training
    # environment exposes the same statistics module under its newer name.
    try:
        import lerobot.common.datasets.compute_stats
    except ModuleNotFoundError:
        import lerobot.datasets.compute_stats as stats
        monkeypatch.setitem(sys.modules,'lerobot.common.datasets.compute_stats',stats)
    root=tmp_path/'recording'
    (automatic_session if automatic else static_session)(root)
    output=tmp_path/'dom';export_sessions([root],output)
    processed=tmp_path/'processed';process(output,processed,repair=True)
    source=jsonl_read(processed/'meta/sources.jsonl')[0]
    assert source['static_object']['schema']==SCHEMA
    code='''
import importlib,sys
from pathlib import Path
import lerobot
sys.path.insert(0,str(Path('third_party/DynamicVLA').resolve()))
try:
    import lerobot.datasets.utils
except ModuleNotFoundError:
    for alias,target in [('constants','lerobot.common.constants'),('datasets','lerobot.common.datasets')]:
        module=importlib.import_module(target);sys.modules['lerobot.'+alias]=module;setattr(lerobot,alias,module)
    for name in ['compute_stats','lerobot_dataset','utils']:
        sys.modules['lerobot.datasets.'+name]=importlib.import_module('lerobot.common.datasets.'+name)
from utils.datasets import LeRobotDataset
reader=LeRobotDataset('local/static-fixture',root=Path(sys.argv[1]),episodes=[0],split='train',
    pin_memory=False,delta_action=True,euler_delta_convention='wrapped_xyz_v1',
    required_features=['action','observation.state','observation.environment_state'],
    delta_timestamps={'observation':[-2,0],'action':list(range(20))})
item=reader[5]
assert item['dynamic_object.velocity_valid'].all()
assert item['dynamic_object.velocity_required'].all()
assert not item['dynamic_object.velocity_xyz'].any()
assert tuple(item['action'].shape)==(20,7)
'''
    result=subprocess.run([sys.executable,'-c',code,str(processed)],
        cwd=Path(__file__).resolve().parents[2],env=dict(os.environ,HF_HUB_OFFLINE='1'),
        capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workstation'))
    from family_data import RealFamilyDataset
    import pyarrow.parquet as pq
    table=pq.read_table(processed/'data/chunk-000/episode_000000.parquet')
    arrays={k:torch.tensor(np.array(table[k].to_pylist()),dtype=torch.float32)
            for k in ['observation.state','action','observation.environment_state']}
    q=jsonl_read(processed/'quality/episode_000000.jsonl')
    if automatic:
        held=[flag for flag in q if flag['manipulation_phase']=='held_success_inferred']
        assert held and not any(flag['phase_confirmed'] for flag in held)
        assert all(not flag['velocity_required'] and flag['tracking_usable'] for flag in held)
        release=max(flag['static_automatic_interval']['end_frame'] for flag in held)
        assert not any(flag['tracking_usable'] for flag in q[release:])
    dataset=RealFamilyDataset.__new__(RealFamilyDataset);dataset.anchors=[(0,5)];dataset.max_delay=0
    dataset.metadata={0:jsonl_read(processed/'meta/episodes.jsonl')[0]}
    decoder=SimpleNamespace(get_frame_at=lambda i:SimpleNamespace(data=torch.zeros(3,8,8,dtype=torch.uint8)))
    dataset.episode=lambda ep:(arrays,q,{'observation.images.opst_cam':decoder,'observation.images.wrist_cam':decoder})
    item=dataset[0]
    assert item['velocity_required'] and not item['velocity'].any()


def test_extrapolation_does_not_replace_static_labels(tmp_path):
    from real_robot.velocity_extrapolation import extrapolate_episode
    root=tmp_path/'recording';manifest,episode=static_session(root)
    rows,_,quality=align_episode(root,manifest,episode)
    changed,flags=extrapolate_episode(rows,quality)
    assert rows==changed and quality==flags


@pytest.mark.parametrize('visible',[True,False])
def test_static_keyboard_flow_requires_initial_position_but_no_lift_release_keys(tmp_path,visible):
    from real_robot.dynamic_capture import main
    root=tmp_path;rig=root/'rig.json';rig.write_text('{}')
    config=root/'config.yaml';config.write_text('test configuration')
    lease=root/'lease.json';obsfile=root/'observation.json'
    obs=observation(stamp=time.monotonic())
    obs.update(running=True,recording_lease=str(lease),camera_fps=30,output_frame='right_base_estimate',
               rig_sha256=hashlib.sha256(rig.read_bytes()).hexdigest())
    if not visible:obs.update(stale=True,fusion=None)
    obsfile.write_text(json.dumps(obs))
    bus=Mock(is_connected=True)
    control=Mock(enabled=False,error=None,pending=queue.Queue(),torque_enabled_here=False)
    control.toggle.side_effect=lambda:setattr(control,'enabled',True)
    wrist=Mock(enabled=threading.Event())
    wrist.snapshot.side_effect=lambda:(np.zeros((48,64,3),np.uint8),time.monotonic())
    keys=[32,13,ord('q')] if visible else [32,ord('q')]
    with patch('real_robot.tracking_config.load_config',return_value={'perception':{'rig':str(rig)}}), \
         patch('real_robot.grasp_capture.connect_buses',return_value=({'leader':bus,'follower':bus},{})), \
         patch('real_robot.grasp_capture.RobotIO',return_value=Mock(provenance={})), \
         patch('real_robot.collection_control.CollectionControl',return_value=control), \
         patch('real_robot.dynamic_capture.EpisodeWrist',return_value=wrist), \
         patch('real_robot.dynamic_capture.cv2.namedWindow'), \
         patch('real_robot.dynamic_capture.cv2.imshow'), \
         patch('real_robot.dynamic_capture.cv2.destroyAllWindows'), \
         patch('real_robot.dynamic_capture.cv2.waitKey',side_effect=keys):
        result=main(['--config',str(config),'--task','striped_ball','--split','train',
                     '--output',str(root/'capture'),'--observation',str(obsfile),
                     '--recording-lease',str(lease),'--static-object','--clutched','--execute',
                     '--default-success','--no-audio','--start-delay','0'])
    assert result==0
    saved=list((root/'capture/episodes').glob('*.json'))
    if visible:
        assert len(saved)==1
        episode=json.loads(saved[0].read_text())
        assert episode['outcome']=='success'
        assert episode['events']=={}
        assert episode['static_object']['schema']==SCHEMA
    else:
        assert not saved
        control.toggle.assert_not_called()


def automatic_trace():
    n=80
    z=np.r_[np.linspace(.15,.08,10),np.linspace(.08,.05,10),np.linspace(.05,.11,25),
            np.full(35,.11)]
    command=np.r_[np.ones(10),np.full(50,-.4),np.linspace(-.4,1,20)]
    measured=np.r_[np.ones(2),command[:-2]]
    rows=[{'observation.state':[0,0,float(z[i]),0,0,0],'action':[0]*6+[float(command[i])]} for i in range(n)]
    q=[dict(frame_index=i,anchor_monotonic_s=i/25,position_age_s=i/25,
            position_current=False,velocity_estimated=False,velocity_zero_filled=False,
            static_collection_schema=SCHEMA,static_assumption_valid=True,
            velocity_input_source='operator_static_assumption',measured_gripper_normalized=float(measured[i])) for i in range(n)]
    return rows,q


def test_automatic_held_interval_starts_at_lift_onset_and_stops_on_reopening():
    rows,q=automatic_trace()
    intervals=annotate_automatic_phases(rows,q,'success')
    assert len(intervals)==1
    item=intervals[0]
    assert 20<item['start_frame']<item['confirmed_frame']<45
    assert 60<item['end_frame']<70
    labels,required=phases(rows,q)
    a,b=item['start_frame'],item['end_frame']
    assert all(label=='held_success_inferred' for label in labels[a:b])
    assert not required[a:b].any() and required[b:].all()
    assert not any(quality_valid(flag) for flag in q[a:])
    assert not any(flag.get('phase_confirmed',False) for flag in q)


@pytest.mark.parametrize('failure',['failed_outcome','measured_jaws_open','no_lift','missing_gripper'])
def test_automatic_phase_inference_does_not_confirm_unsupported_pick(failure):
    rows,q=automatic_trace()
    if failure=='measured_jaws_open':
        for flag in q:flag['measured_gripper_normalized']=1.
    elif failure=='missing_gripper':
        for flag in q:flag.pop('measured_gripper_normalized')
    elif failure=='no_lift':
        for row in rows:row['observation.state'][2]=.05
    assert not annotate_automatic_phases(rows,q,'failed' if failure=='failed_outcome' else 'success')
    _,required=phases(rows,q)
    assert required.all()
