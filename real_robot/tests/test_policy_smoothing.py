import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real_robot.policy_smoothing import PoseTargetSmoother


def pose(x=0,angle=0):
    t=np.eye(4);t[0,3]=x;t[:3,:3]=Rotation.from_euler('z',angle,degrees=True).as_matrix();return t


def test_fixed_target_converges_and_hold_reset_has_no_residual_motion():
    s=PoseTargetSmoother(.08);s.reset(pose())
    for _ in range(60):out=s.step(pose(.01,10),.04)
    np.testing.assert_allclose(out,pose(.01,10),atol=1e-8)
    s.reset(pose(.005,5));out=s.step(pose(.005,5),.04)
    np.testing.assert_allclose(out,pose(.005,5),atol=1e-12)
    np.testing.assert_allclose(s.v,0,atol=1e-12)


def test_causal_blend_reduces_alternating_target_jerk():
    s=PoseTargetSmoother(.08);s.reset(pose())
    raw=.03*np.sin(np.arange(200)*.04*2*np.pi)+.003*(-1.)**np.arange(200)
    output=np.array([s.step(pose(x),.04)[0,3] for x in raw])
    raw_jerk=np.diff(raw,n=3)/.04**3;filtered_jerk=np.diff(output,n=3)/.04**3
    assert np.quantile(np.abs(filtered_jerk),.95)<.65*np.quantile(np.abs(raw_jerk),.95)
    assert np.quantile(np.abs(raw-output),.95)<.015


def test_rotation_crossing_pi_stays_on_short_continuous_branch():
    s=PoseTargetSmoother(.08);s.reset(pose())
    for _ in range(60):out=s.step(pose(angle=179),.04)
    previous=out.copy()
    out=s.step(pose(angle=-179),.04)
    distance=np.linalg.norm(Rotation.from_matrix(out[:3,:3]@previous[:3,:3].T).as_rotvec())
    assert distance<np.deg2rad(2)
    for _ in range(60):out=s.step(pose(angle=-179),.04)
    np.testing.assert_allclose(out,pose(angle=-179),atol=1e-8)
    assert s.x[5]>np.pi


def test_bad_timing_is_rejected():
    with pytest.raises(ValueError):PoseTargetSmoother(.01)
    s=PoseTargetSmoother(.08);s.reset(pose())
    for dt in [0,-.01,.2,np.nan]:
        with pytest.raises(ValueError):s.step(pose(.1),dt)


def test_lower_grasp_workspace_is_opt_in_to_policy_loading():
    from pathlib import Path
    from real_robot.tracking_config import load_config
    project=Path(__file__).resolve().parents[2]
    policy=project/'real_robot/config_policy_eval.yaml'
    with pytest.raises(ValueError,match='tracking envelope'):load_config(policy)
    settings=load_config(policy,manipulation=True)
    assert settings['servo']['workspace_min_m'][2]==.02
    assert load_config(project/'real_robot/config_collection.yaml')['servo']['workspace_min_m'][2]==.09
