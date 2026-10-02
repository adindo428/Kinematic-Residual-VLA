import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dom_tracking.config import load_config, validate_config
from dom_tracking.controller import Expert, ModeSelector
from dom_tracking.math3d import dom_euler, point_jacobian, quaternion, rotation, shift_point
from dom_tracking.trajectory import choose_trajectory, generate, predict_modes


def test_hysteresis_exact_boundaries_and_reversal():
    selector = ModeSelector()
    assert [selector.update(v) for v in [0.05, 0.051, 0.05, 0.03, 0.029, -0.05, -0.051, -0.03, -0.029]] == [0, -45, -45, -45, 0, 0, 45, 45, 0]
    assert selector.update(0.1) == -45
    assert selector.update(-0.1) == 45
    with pytest.raises(ValueError):
        selector.update(float("nan"))


def test_dom_angles_and_offset_roundtrip():
    q = quaternion(Rotation.from_euler("xyz", [-1.2, 0.3, -0.7]))
    e = dom_euler(q)
    assert e[0] >= 0 and e[2] >= 0
    np.testing.assert_allclose(Rotation.from_euler("xyz", e).as_matrix(), rotation(q).as_matrix(), atol=1e-6)
    p = np.array([0.4, 0.1, 0.2])
    p2 = shift_point(p, q, [0, 0, 0.1034], [0, 0, 0.107])
    np.testing.assert_allclose(shift_point(p2, q, [0, 0, 0.107], [0, 0, 0.1034]), p)


def test_shifted_jacobian_matches_rigid_point_velocity():
    rng = np.random.default_rng(5)
    j, dq, offset = rng.normal(size=(6, 7)), rng.normal(size=7), rng.normal(size=3)
    twist = j@dq
    shifted = point_jacobian(j, offset)@dq
    np.testing.assert_allclose(shifted[:3], twist[:3]+np.cross(twist[3:], offset))


def test_trajectory_continuous_bounds_and_reproducibility():
    cfg = load_config()
    t = cfg["trajectory"]
    for seed in range(10):
        path = generate(t, np.random.default_rng(seed), 10.2)
        p, v, a = path.sample(np.linspace(0, 10.2, 5001))
        assert np.all(np.abs(p-np.array(t["center"])) <= np.array(t["half_extents"])-t["workspace_margin"]+1e-9)
        assert np.max(np.linalg.norm(v, axis=1)) <= t["max_speed"]+1e-9
        assert np.max(np.linalg.norm(a, axis=1)) <= t["max_acceleration"]+1e-9
        h = 1e-5
        np.testing.assert_allclose((path.sample(1+h)[0]-path.sample(1-h)[0])/(2*h), path.sample(1)[1], atol=1e-9)
    cfg["collection"]["candidates"] = 4
    a, b = choose_trajectory(cfg, 2, [10, 20, 30]), choose_trajectory(cfg, 2, [10, 20, 30])
    assert a.serialize() == b.serialize()
    assert predict_modes(a, cfg).sum() == 250


def test_expert_bounds_and_negative_tilt():
    cfg = load_config()["controller"]
    state = dict(ee_position=np.zeros(3), ee_quaternion=np.array(cfg["down_quaternion"]),
                 ee_velocity=np.zeros(3), joint_position=np.zeros(7))
    j = np.zeros((6, 7)); j[:, :6] = np.eye(6)
    command = Expert(cfg, .005).command(state, dict(position=[10, 10, 10], velocity=[0, .1, 0]),
                                       j, np.array([[-1, 1]]*7), np.ones(7)*.1)
    assert command["mode"] == -45 and command["saturated"]
    assert np.max(np.abs(command["joint_velocity_target"])) <= .1
    assert np.linalg.norm(command["commanded_twist"][3:]) <= cfg["max_angular_speed"]+1e-9


def test_invalid_configuration():
    cfg = load_config()
    cfg["collection"]["physics_hz"] = 201
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_workspace_blocks_outward_velocity_and_allows_recovery():
    cfg = load_config()["controller"]
    state = dict(ee_position=np.array([.6, 0., .3]), ee_quaternion=np.array(cfg["down_quaternion"]),
                 ee_velocity=np.zeros(3), joint_position=np.zeros(7))
    j = np.zeros((6, 7)); j[:, :6] = np.eye(6)
    target = dict(position=[.6, 0., .3], velocity=[.1, 0, 0],
                  workspace_lower=[.3, -.2, .1], workspace_upper=[.6, .2, .5])
    command = Expert(cfg, .005).command(state, target, j, np.array([[-1, 1]]*7), np.ones(7))
    assert command["commanded_twist"][0] == 0
    assert command["saturated"]
    target["position"] = [.4, 0., .3]
    command = Expert(cfg, .005).command(state, target, j, np.array([[-1, 1]]*7), np.ones(7))
    assert -.3 <= command["commanded_twist"][0] < 0


@pytest.mark.parametrize("kind", ["stationary", "positive", "negative", "neutral", "switch"])
def test_named_smoke_trajectories_remain_bounded(kind):
    cfg = load_config()["trajectory"]
    cfg["kind"] = kind
    path = generate(cfg, np.random.default_rng(42), 2.2)
    p, v, a = path.sample(np.linspace(0, 2.2, 2001))
    assert np.all(np.abs(p-cfg["center"]) <= np.array(cfg["half_extents"])-cfg["workspace_margin"]+1e-9)
    assert np.max(np.linalg.norm(v, axis=1)) <= cfg["max_speed"]+1e-9
    assert np.max(np.linalg.norm(a, axis=1)) <= cfg["max_acceleration"]+1e-9
    if kind == "positive":
        assert np.all(v[:, 1] > 0)
    if kind == "negative":
        assert np.all(v[:, 1] < 0)
