import numpy as np
import pytest
import torch
from proactive_vla.phase_gate import CausalPhaseGate, PhaseGateConfig, gate_from_state, replay_prefix_states
from proactive_vla.residual_flow import residualize_actions, restore_residual_actions


@pytest.mark.parametrize("level", [0., .37, 1.])
def test_gate_covers_entire_centered_correction_and_only_translation(level):
    actions = torch.randn(2, 50, 32)
    velocity = torch.tensor([[1., 2., 3.], [-1., 1., .2]])
    mean = torch.full((50, 32), .7)
    kwargs = dict(drift_mean=mean, center_drift=True, action_dt=.04, drift_mode="trajectory")
    original, drift, _ = residualize_actions(actions, velocity, **kwargs)
    residual, gated, _ = residualize_actions(actions, velocity, phase_gate=torch.full((2, 50, 1), level), **kwargs)
    restored, again, _ = restore_residual_actions(residual, velocity, phase_gate=torch.full((2, 50, 1), level), **kwargs)
    torch.testing.assert_close(gated, drift*level)
    torch.testing.assert_close(again, gated)
    torch.testing.assert_close(restored, actions)
    torch.testing.assert_close(residual[..., 3:], actions[..., 3:])
    if level == 0:
        torch.testing.assert_close(residual, actions)


@pytest.mark.parametrize("gate", [torch.ones(2, 50), torch.full((2, 50, 1), 1.1), torch.full((2, 50, 1), float("nan"))])
def test_invalid_gate_is_rejected(gate):
    with pytest.raises(ValueError):
        residualize_actions(torch.zeros(2, 50, 7), torch.ones(2, 3), phase_gate=gate)


def attach():
    gate = CausalPhaseGate()
    for i in range(7):
        z = i*.004
        state = gate.observe(i*.04, [0, 0, z], [0, 0, .1+z], [0, 0, .1], -1)
    assert gate.confirmed_relative is not None
    return gate, state


def test_attachment_uses_closure_lift_and_past_stability_then_known_chunk_fade():
    gate, state = attach()
    chunk = gate_from_state(state, 50, .04)[:, 0]
    assert 0 < chunk[0] < 1
    assert np.all(np.diff(chunk) <= 0)
    assert np.all(chunk[5:] == 0)
    for i in range(7, 15):
        z = i*.004
        state = gate.observe(i*.04, [0, 0, z], [0, 0, .1+z], [0, 0, .1], -1)
    assert gate.phase == "carry"
    released = gate.observe(.60, [0, 0, .06], [0, 0, .16], [0, 0, .1], 1)
    assert np.all(gate_from_state(released, 50, .04) == 0)
    reset = gate.observe(.64, [0, 0, .06], [0, 0, .16], [0, 0, .1], 1, resetting=True)
    assert np.all(gate_from_state(reset, 50, .04) == 0)
    gate.reset()
    assert np.all(gate_from_state(gate.state(0), 50, .04) == 1)


@pytest.mark.parametrize("command,lift,velocity", [(1, .03, .1), (-1, .005, .1), (-1, .03, .5)])
def test_closure_alone_does_not_confirm(command, lift, velocity):
    gate = CausalPhaseGate()
    for i in range(7):
        z = i/6*lift
        state = gate.observe(i*.04, [0, 0, z], [0, 0, .1+z], [0, 0, velocity], command)
    assert gate.confirmed_relative is None
    assert np.all(gate_from_state(state, 50, .04) == 1)


def test_sustained_loss_while_closed_restores_smoothly():
    gate, _ = attach()
    for i in range(7, 20):
        state = gate.observe(i*.04, [0, 0, .03], [.1, 0, .13], [.2, 0, 0], -1)
    assert gate.confirmed_relative is None
    assert gate_from_state(state, 50, .04)[-1, 0] == 1
    # A single slipped observation cannot trigger the sustained-loss rule.
    other, _ = attach()
    state = other.observe(.28, [0, 0, .03], [.1, 0, .13], [.2, 0, 0], -1)
    assert other.confirmed_relative is not None


def test_prefix_replay_cannot_read_current_or_future_commands_and_matches_online():
    n = 20
    times = np.arange(n)*.04
    states = np.zeros((n, 6)); states[:, 2] = np.arange(n)*.004
    env = np.zeros((n, 9)); env[:, 2] = states[:, 2]+.1; env[:, 8] = .1
    actions = np.zeros((n, 7)); actions[:, -1] = -1
    compact, phases = replay_prefix_states(states, env, actions, times)
    changed = actions.copy(); changed[8:, -1] = 1
    alternate, _ = replay_prefix_states(states, env, changed, times)
    np.testing.assert_array_equal(compact[:9], alternate[:9])
    online = CausalPhaseGate()
    for i in range(n):
        state = online.observe(times[i], states[i, :3], env[i, :3], env[i, 6:9], None if i == 0 else actions[i-1, -1])
        np.testing.assert_array_equal(state, compact[i])
    assert np.all(gate_from_state(compact[0], 50, .04) == 1)
    with pytest.raises(ValueError):
        online.observe(0, states[0, :3], env[0, :3], env[0, 6:9], -1)


def test_stability_checks_every_past_velocity_not_only_latest():
    gate = CausalPhaseGate()
    for i in range(7):
        z = i*.004
        gate.observe(i*.04, [0, 0, z], [0, 0, .1+z], [0, 0, 1. if i == 4 else .1], -1)
    assert gate.confirmed_relative is None


def test_secure_rotating_grasp_remains_attached_in_ee_frame():
    from proactive_vla.phase_gate import rotation_xyz
    gate=CausalPhaseGate()
    for i in range(24):
        theta=i*.1;z=i*.004
        ee=np.array([0,0,z]);offset=rotation_xyz([0,0,theta])@np.array([.1,0,0])
        velocity=np.cross([0,0,2.5],offset)+[0,0,.1]
        state=gate.observe(i*.04,ee,ee+offset,velocity,-1,ee_rotation_xyz=[0,0,theta])
    assert gate.phase=='carry' and gate.confirmed_relative is not None
    assert np.all(gate_from_state(state,50,.04)==0)
