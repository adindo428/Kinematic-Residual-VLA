import torch

from proactive_vla.residual_flow import (
    remove_scheduled_residual_drift,
    residualize_actions,
    restore_residual_actions,
    scheduled_residual_drift,
    scheduled_residual_flow_target,
)


def test_zero_drift_keeps_actions_unchanged():
    actions = torch.randn(2, 4, 7)
    velocity = torch.zeros(2, 3)

    residual_actions, centered_drift, raw_drift = residualize_actions(
        actions,
        velocity,
        drift_mode="trajectory",
        action_dt=0.1,
    )
    restored_actions, restored_drift, _ = restore_residual_actions(
        residual_actions,
        velocity,
        drift_mode="trajectory",
        action_dt=0.1,
    )

    assert torch.allclose(centered_drift, torch.zeros_like(actions))
    assert torch.allclose(raw_drift, torch.zeros_like(actions))
    assert torch.allclose(restored_drift, torch.zeros_like(actions))
    assert torch.allclose(residual_actions, actions)
    assert torch.allclose(restored_actions, actions)


def test_positive_xyz_drift_restores_in_expected_direction():
    residual_actions = torch.zeros(1, 3, 7)
    velocity = torch.tensor([[1.0, 2.0, 3.0]])

    restored_actions, centered_drift, _ = restore_residual_actions(
        residual_actions,
        velocity,
        center_drift=False,
        drift_mode="trajectory",
        action_dt=0.1,
    )

    expected_xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3],
            [0.2, 0.4, 0.6],
        ]
    )
    assert torch.allclose(centered_drift[0, :, :3], expected_xyz)
    assert torch.allclose(restored_actions[0, :, :3], expected_xyz)
    assert torch.allclose(restored_actions[0, :, 3:], torch.zeros(3, 4))


def test_centering_subtracts_and_restores_drift_minus_mean():
    actions = torch.zeros(1, 3, 7)
    velocity = torch.tensor([[1.0, 0.0, 0.0]])
    drift_mean = torch.zeros(3, 7)
    drift_mean[:, 0] = torch.tensor([0.0, 0.05, 0.10])

    residual_actions, centered_drift, raw_drift = residualize_actions(
        actions,
        velocity,
        drift_mean=drift_mean,
        center_drift=True,
        drift_mode="trajectory",
        action_dt=0.1,
    )
    restored_actions, restored_drift, _ = restore_residual_actions(
        residual_actions,
        velocity,
        drift_mean=drift_mean,
        center_drift=True,
        drift_mode="trajectory",
        action_dt=0.1,
    )

    expected_raw_x = torch.tensor([0.0, 0.1, 0.2])
    expected_centered_x = torch.tensor([0.0, 0.05, 0.10])
    assert torch.allclose(raw_drift[0, :, 0], expected_raw_x)
    assert torch.allclose(centered_drift[0, :, 0], expected_centered_x)
    assert torch.allclose(restored_drift[0, :, 0], expected_centered_x)
    assert torch.allclose(residual_actions[0, :, 0], -expected_centered_x)
    assert torch.allclose(restored_actions, actions)


def test_linear_schedule_uses_source_to_target_time_orientation():
    velocity = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    time = torch.tensor([1.0, 0.0])  # DynamicVLA: 1=noise/source, 0=action/target.

    scheduled_drift, centered_drift, _ = scheduled_residual_drift(
        time,
        velocity,
        (2, 3, 7),
        lambda_type="linear",
        center_drift=False,
        drift_mode="trajectory",
        action_dt=0.1,
    )

    expected_x = torch.tensor([0.0, 0.1, 0.2])
    assert torch.allclose(centered_drift[0, :, 0], expected_x)
    assert torch.allclose(scheduled_drift[0, :, 0], expected_x)
    assert torch.allclose(scheduled_drift[1], torch.zeros_like(scheduled_drift[1]))


def test_scheduled_target_and_sampling_correction_are_inverse():
    actions = torch.zeros(1, 3, 7)
    noise = torch.ones_like(actions)
    velocity = torch.tensor([[1.0, 0.0, 0.0]])
    time = torch.tensor([1.0])

    target, scheduled_drift, _, _ = scheduled_residual_flow_target(
        noise,
        actions,
        time,
        velocity,
        lambda_type="cosine",
        center_drift=False,
        drift_mode="trajectory",
        action_dt=0.1,
    )
    corrected_velocity, removed_drift, _ = remove_scheduled_residual_drift(
        target,
        time,
        velocity,
        lambda_type="cosine",
        center_drift=False,
        drift_mode="trajectory",
        action_dt=0.1,
    )

    assert torch.allclose(removed_drift, scheduled_drift)
    assert torch.allclose(corrected_velocity, noise - actions)
