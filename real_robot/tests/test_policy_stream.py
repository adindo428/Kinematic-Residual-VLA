import numpy as np
import pytest
from real_robot.policy_stream import ActionStream, gripper_raw, live_object,action_clock_index,action_deadline_elapsed


def test_latency_discard_and_newest_overlap():
    q=ActionStream('a');old=np.zeros((20,7));new=np.ones((20,7))
    assert q.merge('a',0,old,4)==4
    assert q.pop(3) is None
    assert np.array_equal(q.pop(4),old[4])
    assert q.merge('a',3,new,6)==3
    assert np.array_equal(q.pop(6),new[3])
    assert q.pop(6) is None


def test_shared_clock_and_laas_preserve_action_alignment_under_timer_jitter():
    queue=ActionStream('trial');chunk=np.zeros((20,7));chunk[:,0]=np.arange(20)
    current=action_clock_index(.12-.0003)
    assert current==3
    assert queue.merge('trial',0,chunk,current)==3
    for index in range(3,20):
        jitter=.0004 if index%2 else -.0003
        result=queue.pop(action_clock_index(index*.04+jitter))
        assert result[0]==index
    assert queue.pop(action_clock_index(.8)) is None


def test_sampled_target_deadline_is_next_full_tick_not_rounding_boundary():
    sampled=action_clock_index(.1203)
    assert sampled==3
    # Processing can cross the half-period rounding boundary without sending
    # an already elapsed action. The real next target time is .160 seconds.
    assert action_clock_index(.147)==4
    assert not action_deadline_elapsed(sampled,.147)
    assert action_deadline_elapsed(sampled,.160)
    assert action_deadline_elapsed(sampled,.181)
    for index,elapsed in [(-1,0),(1,float('nan'))]:
        with pytest.raises(ValueError):action_deadline_elapsed(index,elapsed)


def test_optional_handoff_uses_time_aligned_overlap_and_short_rotation_path():
    from scipy.spatial.transform import Rotation
    queue=ActionStream('trial',handoff_steps=4)
    old=np.zeros((20,7));old[:,0]=np.arange(20)*.001
    old[:,5]=np.deg2rad(179)
    new=old[3:].copy();new[:,0]+=.04;new[:,5]=np.deg2rad(-179)
    queue.merge('trial',0,old,4)
    queue.pop(4)
    queue.merge('trial',3,new,6)
    # First handoff target keeps the old prediction for this execution time.
    np.testing.assert_allclose(queue.pop(6)[:3],old[6,:3])
    halfway=queue.pop(8)
    assert halfway[0]==pytest.approx(old[8,0]+.02)
    angle=Rotation.from_euler('xyz',halfway[3:6]).as_rotvec()
    assert np.linalg.norm(angle)==pytest.approx(np.pi)
    np.testing.assert_allclose(queue.pop(10),new[7])
    assert queue.pop(10) is None


def test_handoff_does_not_invent_missing_actions_or_accept_wrong_trials():
    queue=ActionStream('trial',handoff_steps=4)
    chunk=np.ones((20,7));chunk[:,3:6]=0
    queue.merge('wrong',0,chunk,4);assert queue.pop(4) is None
    queue.merge('trial',0,chunk,9)
    np.testing.assert_array_equal(queue.pop(9),chunk[9])
    assert queue.pop(20) is None
    for value in [-1,1.5,True]:
        with pytest.raises(ValueError):ActionStream('trial',handoff_steps=value)


def test_expired_chunks_and_episode_reset():
    q=ActionStream('new');chunk=np.ones((20,7))
    q.merge('old',0,chunk,5);assert q.pop(5) is None
    assert q.merge('new',0,chunk,30)==20
    assert q.pop(30) is None
    q.merge('new',31,chunk,32);assert q.pop(32) is not None
    q.merge('new',30,chunk*2,33);assert np.array_equal(q.pop(33),chunk[2])


def test_gripper_uses_export_convention():
    assert gripper_raw(-1,2475,3116)==2475
    assert gripper_raw(1,2475,3116)==3116
    assert gripper_raw(0,3116,2475) in [2795,2796]
    assert gripper_raw(1.0076,2475,3116)==3116
    assert gripper_raw(-1.0076,2475,3116)==2475
    assert gripper_raw(3,3116,2475)==2475
    for value in [np.nan,np.inf,-np.inf]:
        with pytest.raises(ValueError):gripper_raw(value,2475,3116)


def test_missing_velocity_is_not_stationary():
    record=dict(schema='front_side_observation_v1',running=True,rig_sha256='r',output_frame='right_base_estimate',
        stale=False,timing={'oldest_host_arrival_monotonic_s':1},
        track=dict(accepted=True,ready=True,frame='right_base_estimate',position_m=[.3,0,.1]),
        velocity_report=dict(status='unavailable',velocity_mps=[0,0,0]))
    with pytest.raises(ValueError):live_object(record,1.1,'r')
    record['velocity_report']['status']='estimated'
    assert np.array_equal(live_object(record,1.1,'r')[-3:],[0,0,0])
    record['track']['stationary_assumed']=True
    assert np.array_equal(live_object(record,1.1,'r')[-3:],[0,0,0])
    record['track']['measurement_age_s']=.5
    with pytest.raises(ValueError):live_object(record,1.1,'r')
    record['track']['measurement_age_s']=0.
    record['velocity_report']['status']='unavailable'
    with pytest.raises(ValueError):live_object(record,1.1,'r')
    with pytest.raises(ValueError):live_object(record,2,'r')
    with pytest.raises(ValueError):live_object(record,1.1,'wrong')


def test_controller_execute_starts_paused(monkeypatch,tmp_path):
    """Even --execute plus available model actions cannot send goals before SPACE."""
    from real_robot import policy_eval as mod
    from types import SimpleNamespace
    from real_robot.vx300s_kinematics import LOWER,UPPER
    calls=[]
    class Arm:
        def __init__(self):pass
        def check_health(self):pass
        def telemetry(self):return np.array([0.,-.2,.4,0.,1.,0.]),np.zeros(9),{9:2800}
        def goals(self):return np.array([0.,-.2,.4,0.,1.,0.])
        def read_register(self,*args):return 2800
        def joint_velocity_limits_deg_s(self):return [12]*6
        def goal(self,q):calls.append(q)
        def close(self):pass
    class Wrist:
        def __init__(self,*args,**kwargs):
            self.thread=SimpleNamespace(start=lambda:None)
            self.enabled=SimpleNamespace(clear=lambda:None,set=lambda:None)
        def snapshot(self):return np.zeros((480,640,3),np.uint8),mod.time.monotonic()
        def close(self):pass
    monkeypatch.setattr(mod,'TapeArm',Arm);monkeypatch.setattr(mod,'EpisodeWrist',Wrist)
    monkeypatch.setattr(mod,'GeometryGuard',lambda *args:SimpleNamespace(check=lambda *args:None))
    monkeypatch.setattr(mod.cv2,'imshow',lambda *args:None)
    monkeypatch.setattr(mod.cv2,'waitKey',lambda *args:ord('q'))
    monkeypatch.setattr(mod.cv2,'destroyAllWindows',lambda:None)
    exchange=tmp_path/'exchange';exchange.mkdir()
    mod.main(['--perception',str(tmp_path/'perception'),'--exchange',str(exchange),
              '--output',str(tmp_path/'controller'),'--execute'])
    assert calls==[]


def test_brief_tracking_gap_keeps_only_fresh_future_actions():
    queue=ActionStream('trial',max_source_age_steps=6)
    chunk=np.arange(140,dtype=float).reshape(20,7)
    queue.merge('trial',10,chunk,13)
    assert np.array_equal(queue.pop(13),chunk[3])
    # The controller issues no commands on invalid-tracking steps 14/15.
    # On recovery the still-valid future action is available without a reset.
    assert np.array_equal(queue.pop(16),chunk[6])
    assert queue.pop(17) is None  # observation is now too old despite a future suffix
    queue.merge('trial',16,chunk+1,19)
    assert np.array_equal(queue.pop(19),chunk[3]+1)


def test_last_checkpoint_selects_completed_epoch_and_preserves_best(tmp_path):
    import json
    from real_robot.evaluate import prepare_checkpoint
    (tmp_path/'launch_contract.json').write_text(json.dumps({'observation_history':[-2,0]}))
    for epoch,complete in [(3,True),(10,True),(11,False)]:
        directory=tmp_path/'krv100'/f'epoch-{epoch:03d}';directory.mkdir(parents=True)
        (directory/'provenance.json').write_text(json.dumps({'epoch':epoch,'completed_epoch':complete,'validation_loss':.2}))
        (directory/'config.json').write_text(json.dumps({'chunk_size':20,'use_checkpoint_normalization':False}))
        (directory/'model.safetensors').touch()
    (tmp_path/'krv100/best_checkpoint.json').write_text(json.dumps({'path':str(tmp_path/'krv100/epoch-003')}))
    last,record=prepare_checkpoint(tmp_path,'krv100','last')
    assert record['epoch']==10
    assert (last/'model.safetensors').resolve()==tmp_path/'krv100/epoch-010/model.safetensors'
    config=json.loads((last/'config.json').read_text())
    assert config['use_checkpoint_normalization'] and config['delta_timestamps']['observation']==[-2,0]
    best,record=prepare_checkpoint(tmp_path,'krv100','best')
    assert record['epoch']==3 and best!=last


def test_laas_keeps_valid_suffix_after_inference_delay_exceeds_six_steps():
    queue=ActionStream('trial');chunk=np.arange(140,dtype=float).reshape(20,7)
    assert queue.merge('trial',0,chunk,9)==9
    for index in range(9,20):np.testing.assert_array_equal(queue.pop(index),chunk[index])
    assert queue.pop(20) is None


def test_live_object_prediction_bridges_only_short_verified_depth_gaps():
    from real_robot.policy_stream import LiveObjectEstimator
    def record(stamp,accepted=True):
        return dict(schema='front_side_observation_v1',running=True,rig_sha256='r',output_frame='right_base_estimate',stale=False,
            timing={'oldest_host_arrival_monotonic_s':stamp},
            track=dict(accepted=accepted,ready=accepted,frame='right_base_estimate',position_m=[.3,0,.1],
                status='tracking' if accepted else 'coasting',measurement_age_s=0 if accepted else stamp-1.,
                covariance=np.diag([.005**2]*3+[.02**2]*3).tolist()),
            velocity_report=dict(status='estimated' if accepted else 'unavailable',velocity_mps=[.1,0,0]))
    estimator=LiveObjectEstimator(max_gap_s=.08)
    with pytest.raises(ValueError):estimator.read(record(1.04,False),1.06,'r')
    estimator.read(record(1.),1.02,'r')
    result=estimator.read(record(1.04,False),1.06,'r')
    np.testing.assert_allclose(result[:3],[.304,0,.1])
    assert estimator.diagnostics['source']=='constant_velocity_occlusion_prediction'
    # Predictions do not become new measurement anchors and cannot chain past 80ms.
    with pytest.raises(ValueError):estimator.read(record(1.12,False),1.14,'r')
    bad=record(1.);bad['track']['covariance']=np.eye(6).tolist()
    bad_estimator=LiveObjectEstimator();bad_estimator.read(bad,1.02,'r')
    with pytest.raises(ValueError):bad_estimator.read(record(1.04,False),1.06,'r')
    with pytest.raises(ValueError):estimator.read(record(1.04,False),1.3,'r')
    estimator.reset()
    with pytest.raises(ValueError):estimator.read(record(1.04,False),1.06,'r')


@pytest.mark.parametrize('object_inputs',['required','unused'])
def test_fixed_target_diagnostic_ignores_policy_pose_and_gripper(monkeypatch,tmp_path,object_inputs):
    import json,hashlib
    from contextlib import contextmanager
    from types import SimpleNamespace
    from real_robot import policy_eval as mod
    q=np.array([0.,-.2,.4,0.,1.,0.]);sent=[];gripper=[];clock=[1.]
    class Arm:
        def check_health(self):pass
        def telemetry(self):return q.copy(),np.zeros(9),{9:2800}
        def goals(self):return q.copy()
        def read_register(self,*args):return 2800
        def joint_velocity_limits_deg_s(self):return [60]*6
        def goal(self,value):sent.append(np.array(value))
        def close(self):pass
        packet=SimpleNamespace(write4ByteTxRx=lambda *a:gripper.append(a) or (0,0))
        dxl=SimpleNamespace(COMM_SUCCESS=0);port=None
    class Wrist:
        def __init__(self,*a,**k):
            self.thread=SimpleNamespace(start=lambda:None);self.enabled=SimpleNamespace(set=lambda:None,clear=lambda:None)
        def snapshot(self):return np.zeros((480,640,3),np.uint8),clock[0]
        def close(self):pass
    config=mod.load_config(mod.PROJECT/'real_robot/config_policy_eval.yaml',manipulation=True)
    digest=hashlib.sha256((mod.PROJECT/config['perception']['rig']).read_bytes()).hexdigest()
    @contextmanager
    def load(path,**kwargs):
        if path.name=='policy_front.npz':
            record=dict(schema='front_side_observation_v1',running=True,rig_sha256=digest,output_frame='right_base_estimate',stale=False,
                timing={'oldest_host_arrival_monotonic_s':clock[0]},
                track=dict(accepted=True,ready=True,frame='right_base_estimate',position_m=[.4,0,.05]),
                velocity_report=dict(status='estimated',velocity_mps=[0,0,0]))
            if object_inputs=='unused':record.update(track=None,velocity_report=dict(status='unavailable'))
            yield {'bgr':np.zeros((480,640,3),np.uint8),'metadata':json.dumps(record)}
        else:
            yield {'metadata':json.dumps({'token':'fixed','index':0,'inference_seconds':.1}),
                   'actions':np.tile([99.,99.,99.,0.,0.,0.,-1.],(20,1))}
    keys=iter([255]*13+[ord(' '),255,255,255,255,ord('q')])
    monkeypatch.setattr(mod,'TapeArm',Arm);monkeypatch.setattr(mod,'EpisodeWrist',Wrist)
    monkeypatch.setattr(mod,'GeometryGuard',lambda *a:SimpleNamespace(check=lambda *a:None))
    monkeypatch.setattr(mod,'time',SimpleNamespace(monotonic=lambda:clock[0],sleep=lambda seconds:clock.__setitem__(0,clock[0]+seconds)))
    monkeypatch.setattr(mod.uuid,'uuid4',lambda:'fixed');monkeypatch.setattr(mod.np,'load',load)
    monkeypatch.setattr(mod.cv2,'imshow',lambda *a:None);monkeypatch.setattr(mod.cv2,'waitKey',lambda *a:next(keys))
    monkeypatch.setattr(mod.cv2,'destroyAllWindows',lambda:None)
    monkeypatch.setattr(mod.cv2,'VideoWriter',lambda *a:SimpleNamespace(isOpened=lambda:True,write=lambda *a:None,release=lambda:None))
    perception=tmp_path/'perception';perception.mkdir()
    exchange=tmp_path/'exchange';exchange.mkdir()
    mod.main(['--perception',str(perception),'--exchange',str(exchange),'--output',str(tmp_path/'controller'),
              '--execute','--hold-target-check','--object-inputs',object_inputs])
    assert len(sent)>0 and gripper==[]
    for value in sent:np.testing.assert_allclose(value,q,atol=1e-10)
    episode=json.loads((tmp_path/'controller/trial-001.json').read_text())
    assert episode['excluded_from_task_success_rate'] and episode['evaluation_type']=='fixed_held_pose_diagnostic'
    if object_inputs=='unused':
        feedback=[json.loads(x) for x in (tmp_path/'controller/control.jsonl').read_text().splitlines() if json.loads(x)['event']=='feedback']
        assert any(x['active'] and x['tracking_reason'] is None and x['object_quality']['source']=='not_used_by_policy' for x in feedback)


def occlusion_record(stamp,*,accepted=True,velocity=(.1,0,0),position=(.3,0,.1),missing=False):
    return dict(schema='front_side_observation_v1',running=True,rig_sha256='r',output_frame='right_base_estimate',stale=False,
        timing={'oldest_host_arrival_monotonic_s':stamp},
        track=None if missing else dict(accepted=accepted,ready=accepted,frame='right_base_estimate',position_m=list(position),
            status='tracking' if accepted else 'coasting',measurement_age_s=0 if accepted else stamp-1.,
            covariance=np.diag([.005**2]*3+[.02**2]*3).tolist()),
        velocity_report=dict(status='estimated' if accepted else 'unavailable',velocity_mps=list(velocity)))


def test_occlusion_continues_after_tracker_expires_without_chaining_predictions():
    from real_robot.policy_stream import LiveObjectEstimator
    estimator=LiveObjectEstimator()
    estimator.read(occlusion_record(1.),1.02,'r')
    for stamp in [1.2,1.35,1.47]:
        prediction=estimator.read(occlusion_record(stamp,accepted=False,missing=True),stamp+.02,'r')
        np.testing.assert_allclose(prediction[:3],[.3+.1*(stamp-1),0,.1])
        np.testing.assert_allclose(prediction[6:],[.1,0,0])
        assert estimator.diagnostics['predicted'] and estimator.stamp==1.
    with pytest.raises(ValueError):estimator.read(occlusion_record(1.49,accepted=False,missing=True),1.51,'r')


def test_stationary_occlusion_uses_measured_zero_not_a_startup_assumption():
    from real_robot.policy_stream import LiveObjectEstimator
    estimator=LiveObjectEstimator()
    missing=occlusion_record(1.35,accepted=False,velocity=(0,0,0),missing=True)
    with pytest.raises(ValueError):estimator.read(missing,1.37,'r')
    expected=estimator.read(occlusion_record(1.,velocity=(0,0,0)),1.02,'r')
    np.testing.assert_array_equal(estimator.read(missing,1.37,'r'),expected)


@pytest.mark.parametrize('field,value',[('stale',True),('running',False),('rig_sha256','wrong')])
def test_camera_or_calibration_failure_clears_prediction_anchor(field,value):
    from real_robot.policy_stream import LiveObjectEstimator
    estimator=LiveObjectEstimator();estimator.read(occlusion_record(1.),1.02,'r')
    bad=occlusion_record(1.1,accepted=False,missing=True);bad[field]=value
    with pytest.raises(ValueError):estimator.read(bad,1.12,'r')
    assert estimator.previous is None
    with pytest.raises(ValueError):estimator.read(occlusion_record(1.2,accepted=False,missing=True),1.22,'r')


def test_prediction_expires_on_actual_age_travel_uncertainty_and_disabled_mode():
    from real_robot.policy_stream import LiveObjectEstimator
    fast=LiveObjectEstimator();fast.read(occlusion_record(1.,velocity=(.3,0,0)),1.02,'r')
    with pytest.raises(ValueError):fast.read(occlusion_record(1.2,accepted=False,missing=True),1.22,'r')
    uncertain=occlusion_record(1.);uncertain['track']['covariance']=np.diag([.005**2]*3+[.14**2]*3).tolist()
    estimator=LiveObjectEstimator();estimator.read(uncertain,1.02,'r')
    with pytest.raises(ValueError):estimator.read(occlusion_record(1.2,accepted=False,missing=True),1.22,'r')
    estimator=LiveObjectEstimator(max_gap_s=0);estimator.read(occlusion_record(1.),1.02,'r')
    with pytest.raises(ValueError):estimator.read(occlusion_record(1.04,accepted=False,missing=True),1.06,'r')


def test_repeated_frame_does_not_renew_anchor_and_reacquisition_requires_ready_measurement():
    from real_robot.policy_stream import LiveObjectEstimator
    estimator=LiveObjectEstimator();estimator.read(occlusion_record(1.),1.02,'r')
    estimator.read(occlusion_record(1.),1.19,'r');assert estimator.stamp==1.
    warming=occlusion_record(1.35);warming['track'].update(ready=False,status='warming_up')
    with pytest.raises(ValueError):estimator.read(warming,1.37,'r')
    estimator.read(occlusion_record(1.4,position=(.34,0,.1)),1.42,'r')
    predicted=estimator.read(occlusion_record(1.7,accepted=False,missing=True),1.72,'r')
    np.testing.assert_allclose(predicted[:3],[.37,0,.1]);assert estimator.stamp==1.4


@pytest.mark.parametrize('settings',[{'max_gap_s':-1},{'max_gap_s':6},{'max_gap_s':float('nan')},
                                    {'max_displacement_m':0},{'acceleration_std_mps2':-1}])
def test_invalid_occlusion_bounds_fail_before_hardware(settings):
    from real_robot.policy_stream import LiveObjectEstimator
    with pytest.raises(ValueError):LiveObjectEstimator(**settings)


def test_contact_stop_assumption_settles_position_and_bridges_two_seconds_without_renewal():
    from real_robot.policy_stream import LiveObjectEstimator
    estimator=LiveObjectEstimator(max_gap_s=2.,motion_model='contact_stop_assumption')
    estimator.read(occlusion_record(1.),1.02,'r')
    for stamp in [1.3,2.,2.95]:
        result=estimator.read(occlusion_record(stamp,accepted=False,missing=True),stamp+.02,'r')
        elapsed=stamp-1;decay=np.exp(-elapsed/.15)
        np.testing.assert_allclose(result[:3],[.3+.1*.15*(1-decay),0,.1])
        np.testing.assert_allclose(result[6:],[.1*decay,0,0])
        assert estimator.stamp==1 and estimator.diagnostics['contact_stop_is_assumption']
        assert estimator.diagnostics['position_std_m']<=.025
    assert result[0]<.316 and abs(result[6])<1e-6
    with pytest.raises(ValueError):estimator.read(occlusion_record(3.01,accepted=False,missing=True),3.03,'r')


def test_contact_stop_mode_does_not_change_visible_motion_or_ignore_reacquisition():
    from real_robot.policy_stream import LiveObjectEstimator
    estimator=LiveObjectEstimator(max_gap_s=2.,motion_model='contact_stop_assumption')
    measured=estimator.read(occlusion_record(1.),1.02,'r');np.testing.assert_allclose(measured[6:],[.1,0,0])
    estimator.read(occlusion_record(1.8,accepted=False,missing=True),1.82,'r')
    recovered=estimator.read(occlusion_record(2.,position=(.4,0,.1),velocity=(.2,0,0)),2.02,'r')
    np.testing.assert_allclose(recovered[:3],[.4,0,.1]);np.testing.assert_allclose(recovered[6:],[.2,0,0])
    assert estimator.diagnostics['source']=='accepted_measurement' and not estimator.diagnostics['predicted']
    assert estimator.settings['model']=='contact_stop_assumption'


def test_contact_stop_mode_keeps_freshness_travel_and_configuration_checks():
    from real_robot.policy_stream import LiveObjectEstimator
    estimator=LiveObjectEstimator(max_gap_s=2.,motion_model='contact_stop_assumption')
    estimator.read(occlusion_record(1.,velocity=(.6,0,0)),1.02,'r')
    with pytest.raises(ValueError):estimator.read(occlusion_record(1.5,accepted=False,missing=True),1.52,'r')
    stale=occlusion_record(1.7,accepted=False,missing=True);stale['stale']=True
    with pytest.raises(ValueError):estimator.read(stale,1.72,'r')
    assert estimator.previous is None
    with pytest.raises(ValueError):LiveObjectEstimator(motion_model='automatic_contact_detection')
