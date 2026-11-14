"""Return-to-start tests with no hardware, including a lagging plant."""
from types import SimpleNamespace
import numpy as np
import pytest
from real_robot.policy_reset import StartPoseGate,StartPoseReturn,check_joint_path
from real_robot.policy_ik import PolicyDifferentialIK
from real_robot.vx300s_kinematics import forward,jacobian

Q=np.array([0.,-.2,.4,0.,1.,0.]);OFFSET=[.16,0,0]
GEOMETRY=SimpleNamespace(check=lambda *a:None)


def test_start_gate_requires_measured_joints_gripper_and_continuous_settling():
    gate=StartPoseGate(Q,2800,OFFSET)
    for i in range(10):assert not gate.observe(Q,2800,i*.04)
    assert gate.observe(Q,2800,.5)
    # Wrong gripper and a transient excursion invalidate readiness immediately.
    assert not gate.observe(Q,2900,.54)
    for i in range(1,10):assert not gate.observe(Q,2800,.54+i*.04)
    assert gate.observe(Q,2800,1.04)
    assert not gate.observe(Q+.03,2800,1.08)


def test_return_is_checked_before_motion_and_reaches_the_same_fixed_target_twice():
    gate=StartPoseGate(Q,2800,OFFSET)
    for displacement in [.04,-.06]:
        actual=Q+displacement;retained=actual.copy();gate.invalidate()
        plan=StartPoseReturn(retained,gate,[60]*6,.15,45.,GEOMETRY,[.2,-.3,.02],[.58,.3,.35],0.)
        with pytest.raises(RuntimeError,match='checked'):plan.sample(0.)
        now=0.
        while not plan.advance_preflight(GEOMETRY,now):now+=.04
        ik=PolicyDifferentialIK(actual,retained,OFFSET,[60]*6,max_tcp_speed_mps=.15,max_angular_speed_deg_s=45.)
        for _ in range(500):
            now+=.04;candidate=plan.sample(now)
            ik.check_candidate(candidate,actual,.04);check_joint_path(GEOMETRY,ik.goal,candidate,actual)
            ik.accept(candidate);actual+=.65*(candidate-actual);gate.observe(actual,2800,now)
            if plan.finished(now):break
        else:raise AssertionError('Return did not settle')
        np.testing.assert_allclose(actual,Q,atol=np.deg2rad(1))
        np.testing.assert_allclose(candidate,Q,atol=1e-12)
        assert gate.ready


def test_preflight_rejects_blocked_geometry_without_a_command():
    gate=StartPoseGate(Q,2800,OFFSET)
    plan=StartPoseReturn(Q+.05,gate,[60]*6,.15,45.,GEOMETRY,[.2,-.3,.02],[.58,.3,.35],0.)
    def blocked(*args):raise RuntimeError('blocked path')
    with pytest.raises(RuntimeError,match='blocked'):plan.advance_preflight(SimpleNamespace(check=blocked),0.)
    assert not plan.prepared


def test_return_retimes_cartesian_speed_and_detects_failure_to_settle():
    gate=StartPoseGate(Q,2800,OFFSET)
    plan=StartPoseReturn(Q+.1,gate,[180]*6,.02,10.,GEOMETRY,[.2,-.3,.02],[.58,.3,.35],0.)
    now=0.
    while not plan.advance_preflight(GEOMETRY,now,max_samples=100):now+=.04
    for t in np.linspace(0,plan.duration,100):
        q,v,_=map(np.asarray,plan.trajectory.at_time(float(t/plan.scale)))
        twist=jacobian(q,OFFSET)@(v/plan.scale)
        assert np.linalg.norm(twist[:3])<=.02+1e-6
        assert np.linalg.norm(twist[3:])<=np.deg2rad(10)+1e-6
    with pytest.raises(RuntimeError,match='settle'):plan.sample(plan.deadline+.01)


def test_swept_geometry_checks_measured_following_offset():
    visited=[]
    check_joint_path(SimpleNamespace(check=lambda q,*a:visited.append(q.copy())),Q,Q+.01,Q-.02)
    assert any(np.allclose(q,Q-.02) for q in visited)


@pytest.mark.parametrize('holding_bias',[np.zeros(6),np.deg2rad([-.17578125,.791015625,1.318359375,0,-.087890625,-.087890625])])
@pytest.mark.parametrize('target_z',[None,.0123])
def test_controller_returns_and_verifies_the_same_start_before_trial_two(monkeypatch,tmp_path,holding_bias,target_z):
    import hashlib,json
    from contextlib import contextmanager
    from real_robot import policy_eval as mod
    q=Q.copy();goal=q-holding_bias;clock=[1.];writes=[]
    class Arm:
        def check_health(self):pass
        def telemetry(self):
            q[:]+=.65*(goal+holding_bias-q)
            return q.copy(),np.zeros(9),{9:2800}
        def goals(self):return goal.copy()
        def read_register(self,*args):return 2800
        def joint_velocity_limits_deg_s(self):return [60]*6
        def goal(self,value):goal[:]=value;writes.append(value.copy())
        def close(self):pass
        packet=SimpleNamespace(write4ByteTxRx=lambda *a:(0,0));dxl=SimpleNamespace(COMM_SUCCESS=0);port=None
    class Wrist:
        def __init__(self,*a,**k):
            self.thread=SimpleNamespace(start=lambda:None);self.enabled=SimpleNamespace(set=lambda:None,clear=lambda:None)
        def snapshot(self):return np.zeros((480,640,3),np.uint8),clock[0]
        def close(self):pass
    config=mod.load_config(mod.PROJECT/'real_robot/config_policy_eval.yaml',manipulation=True)
    digest=hashlib.sha256((mod.PROJECT/config['perception']['rig']).read_bytes()).hexdigest()
    original_load=np.load
    target=np.r_[mod.pose6(forward(Q+np.array([.06,-.06,.06,0,0,0]),OFFSET)),2*(2800-2475)/(3116-2475)-1]
    if target_z is not None:target[2]=target_z
    exchange=tmp_path/'exchange';exchange.mkdir();perception=tmp_path/'perception';perception.mkdir()
    @contextmanager
    def load(path,**kwargs):
        if path.name=='policy_front.npz':
            record=dict(schema='front_side_observation_v1',running=True,rig_sha256=digest,output_frame='right_base_estimate',stale=False,
                timing={'oldest_host_arrival_monotonic_s':clock[0]},
                track=dict(accepted=True,ready=True,frame='right_base_estimate',position_m=[.4,0,.05]),
                velocity_report=dict(status='estimated',velocity_mps=[0,0,0]))
            yield {'bgr':np.zeros((480,640,3),np.uint8),'metadata':json.dumps(record)}
        else:
            with original_load(exchange/'observation.npz',allow_pickle=False) as saved:meta=json.loads(str(saved['metadata']))
            yield dict(metadata=json.dumps(dict(meta,inference_seconds=.1)),actions=np.tile(target,(20,1)))
    output=tmp_path/'controller'
    def key(*args):
        status=json.loads((output/'status.json').read_text())
        return ord(' ') if status['status'].startswith('READY:') and status['trial']<2 else 255
    monkeypatch.setattr(mod,'TapeArm',Arm);monkeypatch.setattr(mod,'EpisodeWrist',Wrist)
    monkeypatch.setattr(mod,'GeometryGuard',lambda *a:GEOMETRY)
    monkeypatch.setattr(mod,'time',SimpleNamespace(monotonic=lambda:clock[0],sleep=lambda dt:clock.__setitem__(0,clock[0]+dt)))
    monkeypatch.setattr(mod.np,'load',load);monkeypatch.setattr(mod.cv2,'waitKey',key)
    monkeypatch.setattr(mod.cv2,'imshow',lambda *a:None);monkeypatch.setattr(mod.cv2,'destroyAllWindows',lambda:None)
    monkeypatch.setattr(mod.cv2,'VideoWriter',lambda *a:SimpleNamespace(isOpened=lambda:True,write=lambda *a:None,release=lambda:None))
    mod.main(['--perception',str(perception),'--exchange',str(exchange),'--output',str(output),
        '--execute','--trial-seconds','1','--max-trials','2','--seconds','30'])
    summary=json.loads((output/'summary.json').read_text())
    assert summary['trials']==2 and summary['completed_resets']==2 and summary['fixed_start_verified']
    assert np.max([np.linalg.norm(value-Q) for value in writes])>.02
    episodes=[json.loads((output/f'trial-{i:03d}.json').read_text()) for i in [1,2]]
    assert all(e['common_start_verified'] for e in episodes)
    assert all('workspace' not in e['reason'] for e in episodes)
    assert episodes[1]['completed_resets']==1
    np.testing.assert_allclose(episodes[0]['measured_start_q_rad'],episodes[1]['measured_start_q_rad'],atol=np.deg2rad(1))
    saved=json.loads((output/'start_pose.json').read_text())
    np.testing.assert_allclose(saved['command_joints_rad'],Q-holding_bias,atol=1e-12)
    np.testing.assert_allclose(saved['joints_rad'],Q,atol=1e-12)
    assert saved['tolerances']['tcp_m']==.010
    assert saved['tolerances']['joints_deg']==2.


def test_reset_can_use_joint_and_geometry_checks_without_an_ee_workspace_box():
    gate=StartPoseGate(Q,2800,OFFSET)
    plan=StartPoseReturn(Q,gate,[60]*6,.15,45.,GEOMETRY,None,None,0.)
    plan.check_pose(Q,GEOMETRY)


def test_measured_pose_is_not_substituted_for_holding_command():
    bias=np.deg2rad([-.17578125,.791015625,1.318359375,0,-.087890625,-.087890625])
    gate=StartPoseGate(Q,2800,OFFSET,command_joints=Q-bias,command_gripper=2802)
    # Old reset behavior: commanding measured Q leads to Q+bias under load.
    for i in range(20):gate.observe(Q+bias,2800,i*.04)
    assert not gate.ready and gate.diagnostics['position_error_m']>.005
    gate.invalidate()
    # Restoring the original command cancels the *existing* holding offset;
    # no accumulating error feedback or motor gain changes are needed.
    for i in range(20):gate.observe(gate.command_joints+bias,2800,i*.04)
    assert gate.ready
    assert gate.record()['command_gripper_raw']==2802


def test_one_tick_home_jitter_does_not_repeatedly_rearm_reset():
    # Real failing session: these two quiet readings straddled 5mm by 0.05mm.
    target=np.array([-.200951483213019,-.1718058482431918,.08743690490948154,
        .0966407896367954,1.423534171157875,-.0030679615757712823])
    near=target.copy();near[1]=-.16106798272799233
    far=near.copy();far[1]=-.1595340019401067;far[2]=.08897088569736719
    gate=StartPoseGate(target,3114,OFFSET)
    for i in range(40):
        gate.observe(near if i%2==0 else far,3113,i*.04)
        if i>=15:assert gate.ready
    assert .005<gate.diagnostics['position_error_m']<.006
    assert gate.diagnostics['filtered_position_error_m']<=.0055
    assert gate.diagnostics['filtered_position_threshold_m']==pytest.approx(.0055)
    assert gate.diagnostics['position_window_span_m']<.002


def test_filtered_position_does_not_hide_a_still_moving_arm():
    gate=StartPoseGate(Q,2800,OFFSET)
    for i in range(12):
        q=Q.copy();q[1]+=.0008*i
        gate.observe(q,2800,i*.04)
    assert not gate.ready
    assert gate.diagnostics['filtered_position_error_m']<.005
    assert gate.diagnostics['position_window_span_m']>.002
    assert 'TCP still moving >2mm/window' in gate.diagnostics['blocking_checks']


def test_home_hysteresis_requires_strict_initial_entry_and_drops_on_a_real_excursion():
    target=np.array([-.200951483213019,-.1718058482431918,.08743690490948154,
        .0966407896367954,1.423534171157875,-.0030679615757712823])
    near=target.copy();near[1]=-.16106798272799233
    far=near.copy();far[1]=-.1595340019401067;far[2]=.08897088569736719
    gate=StartPoseGate(target,3114,OFFSET)
    for i in range(20):gate.observe(far,3113,i*.04)
    assert not gate.ready  # 5.05mm cannot qualify for initial entry.
    for i in range(20,40):gate.observe(near,3113,i*.04)
    assert gate.ready
    for i in range(40,60):gate.observe(far,3113,i*.04)
    assert gate.ready  # A sub-tick boundary crossing cannot rearm the reset.
    excursion=target.copy();excursion[1]+=.03
    assert not gate.observe(excursion,3113,2.4)


def test_practical_start_tolerance_accepts_settled_recorded_offset_but_not_larger_motion():
    # Failed reset 20261110T004910Z: 8mm TCP / 1.14deg joint / 1.26deg rotation.
    target=np.array([-.200951483213019,-.1718058482431918,.08743690490948154,
        .0966407896367954,1.423534171157875,-.0030679615757712823])
    actual=np.array([-.19481556006147643,-.1518640980006785,.08897088569736719,
        .0966407896367954,1.423534171157875,-.0030679615757712823])
    relaxed=StartPoseGate(target,3114,OFFSET,position_tolerance_m=.010,joint_tolerance_deg=2.)
    strict=StartPoseGate(target,3114,OFFSET)
    for i in range(25):
        ready=relaxed.observe(actual,3113,i*.04)
        strict.observe(actual,3113,i*.04)
        if i<10:assert not ready
    assert relaxed.ready and not strict.ready
    assert .007<relaxed.diagnostics['position_error_m']<.009
    assert 1.1<relaxed.diagnostics['joint_error_deg']<1.2
    outside=actual.copy();outside[1]+=.04
    assert not relaxed.observe(outside,3113,1.0)


def test_larger_start_tolerance_still_requires_quiet_feedback_and_correct_gripper():
    gate=StartPoseGate(Q,2800,OFFSET,position_tolerance_m=.010,joint_tolerance_deg=2.)
    for i in range(12):
        q=Q.copy();q[1]+=.0008*i
        gate.observe(q,2800,i*.04)
    assert not gate.ready and 'TCP still moving >2mm/window' in gate.diagnostics['blocking_checks']
    for i in range(12,40):gate.observe(Q,2800,i*.04)
    assert gate.ready
    assert not gate.observe(Q,2900,1.6)


@pytest.mark.parametrize('position,joint',[(0,2),(.01,0),(-.01,2),(.01,float('nan'))])
def test_invalid_start_tolerances_are_rejected(position,joint):
    with pytest.raises(ValueError,match='finite and positive'):
        StartPoseGate(Q,2800,OFFSET,position_tolerance_m=position,joint_tolerance_deg=joint)
