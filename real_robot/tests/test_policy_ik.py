"""Numerical kinematics and timed target tracking, without robot hardware."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real_robot.policy_ik import PolicyDifferentialIK,rotation_error
from real_robot.vx300s_kinematics import forward,jacobian,LOWER,UPPER

OFFSET=[.16,0,0]
Q=np.deg2rad([10,-20,30,15,65,-10])


def solver(q=Q,rates=None,**kwargs):
    defaults=dict(max_tcp_speed_mps=10.,max_angular_speed_deg_s=1000.)
    defaults.update(kwargs)
    return PolicyDifferentialIK(q,q,OFFSET,[1000]*6 if rates is None else rates,**defaults)


def test_tcp_jacobian_matches_finite_differences():
    rng=np.random.default_rng(42);h=1e-6
    for _ in range(30):
        q=rng.uniform(LOWER+.15,UPPER-.15);numeric=np.empty((6,6))
        for k in range(6):
            d=np.eye(6)[k]*h;a=forward(q+d,OFFSET);b=forward(q-d,OFFSET)
            numeric[:3,k]=(a[:3,3]-b[:3,3])/(2*h)
            r=a[:3,:3]@b[:3,:3].T
            numeric[3:,k]=np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])/(4*h)
        np.testing.assert_allclose(jacobian(q,OFFSET),numeric,atol=5e-9,rtol=1e-7)


def test_correction_is_from_measured_joints_and_does_not_wind_up():
    target=forward(Q+.005,OFFSET);controller=solver()
    first=controller.step(target,Q,.04)
    before=np.linalg.norm(target-forward(Q,OFFSET))
    assert np.linalg.norm(target-forward(first,OFFSET))<before*.1
    controller.accept(first)
    # A lagging plant must not turn repeated identical targets into growing goals.
    for _ in range(20):
        repeated=controller.step(target,Q,.04)
        np.testing.assert_allclose(repeated,first,atol=1e-12)
        controller.accept(repeated)


def test_small_pose_targets_are_not_retimed_by_a_fixed_gain():
    target=forward(Q+.003,OFFSET)
    a=solver().step(target,Q,.02);b=solver().step(target,Q,.04)
    np.testing.assert_allclose(a,b,atol=1e-12)
    controller=solver();controller.step(target,Q,.04)
    assert controller.diagnostics['command_scale']==1.


def test_pose_spacing_changes_requested_motion_automatically():
    near=forward(Q,OFFSET);far=near.copy();near[:3,3]+=[.0005,0,0];far[:3,3]+=[.001,0,0]
    a=solver();b=solver()
    qa=a.step(near,Q,.04);qb=b.step(far,Q,.04)
    np.testing.assert_allclose(qb-Q,2*(qa-Q),rtol=1e-6,atol=1e-12)
    assert b.diagnostics['requested_tcp_speed_mps']==pytest.approx(2*a.diagnostics['requested_tcp_speed_mps'])


def test_joint_and_cartesian_slew_bounds_survive_large_pose_error():
    rates=np.array([10,15,20,25,30,35]);controller=solver(rates=rates,max_tcp_speed_mps=.05,max_angular_speed_deg_s=15)
    target=forward(Q+np.array([.2,.1,-.1,.3,.1,-.3]),OFFSET)
    candidate=controller.step(target,Q,.04)
    assert np.all(np.abs(candidate-Q)<=np.deg2rad(rates)*.04+1e-10)
    old,new=forward(Q,OFFSET),forward(candidate,OFFSET)
    assert np.linalg.norm(new[:3,3]-old[:3,3])<=.05*.04+1e-10
    assert np.linalg.norm(rotation_error(new[:3,:3],old[:3,:3]))<=np.deg2rad(15)*.04+1e-10


def test_closed_loop_ideal_plant_converges_to_fixed_pose():
    q=Q.copy();controller=solver(rates=[60]*6,max_tcp_speed_mps=.3,max_angular_speed_deg_s=90)
    target=forward(Q+np.array([.08,-.04,.06,.08,.04,-.08]),OFFSET)
    for _ in range(100):
        q=controller.step(target,q,.04);controller.accept(q)
    np.testing.assert_allclose(forward(q,OFFSET),target,atol=1e-5)


def test_no_cartesian_velocity_caps_uses_hardware_joint_limits_without_old_ee_ceiling():
    rates=np.full(6,120.)
    controller=solver(rates=rates,max_tcp_speed_mps=None,max_angular_speed_deg_s=None)
    old=forward(Q,OFFSET);target=forward(Q+np.array([.04,-.04,.03,.04,.04,-.04]),OFFSET)
    candidate=controller.step(target,Q,.04);new=forward(candidate,OFFSET)
    assert np.linalg.norm(new[:3,3]-old[:3,3])>.15*.04
    assert np.all(np.abs(candidate-Q)<=np.deg2rad(rates)*.04+1e-9)
    assert controller.diagnostics['tcp_speed_ceiling_mps'] is None
    assert controller.diagnostics['angular_speed_ceiling_deg_s'] is None
    with pytest.raises(RuntimeError,match='velocity'):
        controller.check_candidate(Q+.09,Q,.04)


def test_euler_wrap_uses_shortest_physical_rotation():
    pose=forward(Q,OFFSET);target=pose.copy()
    target[:3,:3]=Rotation.from_rotvec([0,0,np.deg2rad(2)]).as_matrix()@pose[:3,:3]
    controller=solver();controller.step(target,Q,.04)
    assert controller.diagnostics['rotation_error_deg']==pytest.approx(2.)
    a=Rotation.from_euler('xyz',[.1,.3,2*np.pi-.01]).as_matrix()
    b=Rotation.from_euler('xyz',[.1,.3,-.01]).as_matrix()
    np.testing.assert_allclose(rotation_error(a,b),0.,atol=1e-9)


def test_singular_pose_is_finite_and_stall_bound_is_enforced():
    q=np.zeros(6);controller=solver(q=q,rates=[30]*6)
    target=forward(q,OFFSET);target[:3,3]+=[.2,.1,-.1]
    candidate=controller.step(target,q,.04)
    assert np.isfinite(candidate).all()
    controller.accept(q+.11)
    with pytest.raises(RuntimeError,match='following error'):controller.step(target,q,.04)


def test_slow_wrist_does_not_uniformly_suppress_xyz_motion():
    target=forward(Q,OFFSET);target[:3,3]+=[.003,-.002,-.004]
    target[:3,:3]=target[:3,:3]@Rotation.from_euler('x',60,degrees=True).as_matrix()
    rates=[60,60,60,60,60,1]
    old=solver(rates=rates,method='uniform_dls',max_tcp_speed_mps=.15,max_angular_speed_deg_s=45)
    new=solver(rates=rates,method='bounded_dls',max_tcp_speed_mps=.15,max_angular_speed_deg_s=45)
    a=old.step(target,Q,.04);b=new.step(target,Q,.04)
    initial=np.linalg.norm(target[:3,3]-forward(Q,OFFSET)[:3,3])
    progress_old=initial-np.linalg.norm(target[:3,3]-forward(a,OFFSET)[:3,3])
    progress_new=initial-np.linalg.norm(target[:3,3]-forward(b,OFFSET)[:3,3])
    assert progress_new>5*progress_old and progress_new>.003
    assert abs(b[-1]-Q[-1])<=np.deg2rad(1)*.04+1e-10
    assert new.diagnostics['orientation_weight_m']==.1


def test_bounded_dls_handles_joint_edges_and_noisy_rotation_targets():
    rng=np.random.default_rng(9)
    for _ in range(80):
        q=rng.uniform(LOWER+1e-4,UPPER-1e-4)
        target=forward(q,OFFSET);target[:3,3]+=rng.uniform(-.03,.03,3)
        target[:3,:3]=Rotation.from_rotvec(rng.uniform(-.7,.7,3)).as_matrix()@target[:3,:3]
        controller=solver(q=q,rates=[30]*6,max_tcp_speed_mps=.15,max_angular_speed_deg_s=45)
        candidate=controller.step(target,q,.04)
        controller.check_candidate(candidate,q,.04)
        assert np.isfinite(candidate).all() and np.all(candidate>=LOWER) and np.all(candidate<=UPPER)


def test_frozen_plant_cannot_accumulate_unbounded_command_lead():
    controller=solver(rates=[60]*6,max_tcp_speed_mps=.15,max_angular_speed_deg_s=45)
    target=forward(Q,OFFSET);target[:3,3]+=[.05,-.03,-.04]
    for _ in range(100):
        candidate=controller.step(target,Q,.04);controller.accept(candidate)
        assert np.max(abs(candidate-Q))<=np.deg2rad(6)+1e-9


def test_invalid_solver_configuration_is_rejected():
    with pytest.raises(ValueError):solver(method='unknown')


@pytest.mark.parametrize('source', ['measured', 'retained goal'])
@pytest.mark.parametrize('bound,offset', [(LOWER,-1e-5),(UPPER,1e-5)])
def test_startup_limit_rejection_identifies_joint_and_source(source,bound,offset):
    actual=Q.copy();retained=Q.copy()
    (actual if source=='measured' else retained)[1]=bound[1]+offset
    with pytest.raises(ValueError,match=source+r' shoulder \(motor 2\).*outside'):
        PolicyDifferentialIK(actual,retained,OFFSET,[60]*6)
    for value in [0,-1,np.nan]:
        with pytest.raises(ValueError):solver(orientation_weight_m=value)
