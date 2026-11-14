"""No device access: changing targets, reference continuity and safety holds."""
import numpy as np
import pytest
from real_robot.policy_trajectory import JointTrajectoryGovernor
from real_robot.policy_trajectory import learned_governor_enabled
from real_robot.policy_ik import PolicyDifferentialIK
from real_robot.vx300s_kinematics import forward

Q=np.deg2rad([10,-20,30,15,65,-10])


def test_live_learned_execution_requires_acceleration_and_jerk_governor():
    assert learned_governor_enabled('hardware','auto',True)
    assert learned_governor_enabled('hardware','on',True)
    assert not learned_governor_enabled('hardware','off',False)
    with pytest.raises(ValueError,match='vibration'):
        learned_governor_enabled('hardware','off',True)
    for profile in ['hardware','smooth','auto','pilot','standard']:
        assert learned_governor_enabled(profile,'on',True)
        with pytest.raises(ValueError,match='vibration'):
            learned_governor_enabled(profile,'off',True)


def test_changing_targets_preserve_velocity_acceleration_and_jerk_bounds():
    governor=JointTrajectoryGovernor(Q,[60.]*6)
    rng=np.random.default_rng(4);last_v=np.zeros(6);last_a=np.zeros(6)
    for index in range(300):
        dt=.04 if index%5 else .047
        target=Q+rng.uniform(-.15,.15,6)
        q=governor.propose(target,dt);governor.accept(q)
        assert np.max(abs(governor.velocity))<=np.deg2rad(60)+1e-9
        assert np.max(abs(governor.acceleration))<=np.deg2rad(120)+1e-9
        assert np.max(abs(governor.acceleration-last_a))/dt<=np.deg2rad(600)+1e-8
        assert np.max(abs(governor.velocity-last_v))/dt<=np.deg2rad(120)+1e-8
        last_v=governor.velocity.copy();last_a=governor.acceleration.copy()


def test_reference_replans_from_accepted_state_and_hold_clears_momentum():
    g=JointTrajectoryGovernor(Q,[60.]*6);first=g.propose(Q+.1,.04)
    # An unsent proposal must not advance the trajectory.
    np.testing.assert_array_equal(g.propose(Q+.1,.04),first)
    with pytest.raises(ValueError):g.accept(first+.001)
    g.accept(first)
    for _ in range(5):g.accept(g.propose(Q+.1,.04))
    assert np.linalg.norm(g.velocity)>0
    stopped=g.position.copy();g.reset(stopped)
    np.testing.assert_allclose(g.propose(stopped,.04),stopped,atol=1e-12)
    np.testing.assert_array_equal(g.velocity,np.zeros(6))


def test_dls_and_governor_converge_with_lagging_plant_without_limit_violation():
    q=Q.copy();ik=PolicyDifferentialIK(q,q,[.16,0,0],[60.]*6,max_tcp_speed_mps=.15,max_angular_speed_deg_s=45.)
    g=JointTrajectoryGovernor(q,[60.]*6)
    target=forward(Q+np.array([.08,-.04,.06,.08,.04,-.08]),[.16,0,0])
    for _ in range(250):
        candidate=g.propose(ik.step(target,q,.04),.04)
        ik.check_candidate(candidate,q,.04);ik.accept(candidate);g.accept(candidate)
        q+=.65*(candidate-q)
    np.testing.assert_allclose(forward(q,[.16,0,0]),target,atol=1e-5)


def test_joint_and_following_guards_still_reject_governed_commands():
    ik=PolicyDifferentialIK(Q,Q,[.16,0,0],[60.]*6)
    with pytest.raises(RuntimeError,match='following'):ik.check_candidate(Q+.11,Q,.04)
    with pytest.raises(RuntimeError,match='velocity'):ik.check_candidate(Q+.05,Q,.04)
    with pytest.raises(RuntimeError,match='joint limits'):ik.check_candidate(np.ones(6)*100,Q,.04)


def test_bad_limits_and_intervals_are_rejected():
    for value in [0,-1,np.nan]:
        with pytest.raises(ValueError):JointTrajectoryGovernor(Q,[60.]*6,jerk_deg_s3=value)
    g=JointTrajectoryGovernor(Q,[60.]*6)
    for dt in [0,-.01,.2,np.nan]:
        with pytest.raises(ValueError):g.propose(Q,dt)


def test_bounded_dls_tracks_changing_xyz_with_rotation_noise_through_smooth_pipeline():
    from scipy.spatial.transform import Rotation
    from real_robot.policy_smoothing import PoseTargetSmoother
    q=Q.copy();origin=forward(q,[.16,0,0]);smoother=PoseTargetSmoother(.08);smoother.reset(origin)
    ik=PolicyDifferentialIK(q,q,[.16,0,0],[60]*6,max_tcp_speed_mps=.15,max_angular_speed_deg_s=45.)
    governor=JointTrajectoryGovernor(q,[60]*6);errors=[]
    for index in range(200):
        t=index*.04;target=origin.copy()
        target[:3,3]+=[.01*np.sin(t*2*np.pi*.4),.006*np.cos(t*2*np.pi*.3),-.008*np.sin(t*2*np.pi*.25)]
        target[:3,:3]=Rotation.from_euler('xyz',[np.sin(t*2*np.pi*.5),0,3*(-1)**index],degrees=True).as_matrix()@origin[:3,:3]
        command=governor.propose(ik.step(smoother.step(target,.04),q,.04),.04)
        ik.check_candidate(command,q,.04);ik.accept(command);governor.accept(command)
        q+=.65*(command-q);errors.append(np.linalg.norm(target[:3,3]-forward(q,[.16,0,0])[:3,3]))
    assert np.median(errors)<.006
