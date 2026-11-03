"""Residual action helpers shared by training, inference, and tests.

DynamicVLA integrates its action flow from noise at ``t=1`` to actions at
``t=0``.  Faithful Cartesian KRFM keeps that reverse-time convention and can
either change the action coordinate:

    residual_action = raw_action - (Phi(v_hat) - d_bar)

The neural action expert is trained and sampled in this residual coordinate. At
the end of sampling we restore the raw action by adding the same centered drift
back once. Rotation and gripper channels are intentionally left to the learned
expert; the analytic branch only fills XYZ action channels.

The scheduled vector-field variant mirrors ``residual_flow_2d.py``. It trains on
the raw-action interpolation but asks the expert to predict:

    target = noise - raw_action + lambda(1 - t) * (Phi(v_hat) - d_bar)

During reverse-time sampling we subtract the same scheduled drift from the
predicted denoising vector before the Euler step.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch


DEFAULT_COVARIANCE_STD = 0.03


def lambda_schedule(time: torch.Tensor, schedule_type: str = "constant") -> torch.Tensor:
    """Return lambda(t) for residual drift injection."""
    if not torch.is_tensor(time):
        time = torch.as_tensor(time, dtype=torch.float32)

    schedule = str(schedule_type or "constant").lower()
    if schedule == "constant":
        return torch.ones_like(time)
    if schedule == "linear":
        return 1.0 - time
    if schedule == "cosine":
        return torch.cos(math.pi * 0.5 * time)
    raise ValueError(f"Unknown residual lambda schedule: {schedule_type!r}")


def _as_action_std(
    action_std: Optional[torch.Tensor],
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if action_std is None:
        return None
    std = torch.as_tensor(action_std, device=device, dtype=dtype).flatten()
    if std.numel() < min(3, action_dim):
        return None
    return std


def _broadcast_velocity(
    velocity_xyz: torch.Tensor,
    batch_size: int,
    chunk_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    vel = torch.as_tensor(velocity_xyz, device=device, dtype=dtype)
    if vel.shape[-1] < 3:
        raise ValueError(f"Expected velocity last dimension >= 3, got {tuple(vel.shape)}")
    vel = vel[..., :3]

    if vel.ndim == 1:
        vel = vel.view(1, 1, 3).expand(batch_size, chunk_size, 3)
    elif vel.ndim == 2:
        if vel.shape[0] == batch_size:
            vel = vel[:, None, :].expand(batch_size, chunk_size, 3)
        elif vel.shape[0] == chunk_size:
            vel = vel[None, :, :].expand(batch_size, chunk_size, 3)
        else:
            raise ValueError(
                "Velocity shape must be (B,3), (T,3), or (B,T,3); "
                f"got {tuple(vel.shape)} for B={batch_size}, T={chunk_size}"
            )
    elif vel.ndim == 3:
        if vel.shape[0] != batch_size:
            raise ValueError(f"Velocity batch mismatch: {vel.shape[0]} != {batch_size}")
        if vel.shape[1] == 1:
            vel = vel.expand(batch_size, chunk_size, 3)
        elif vel.shape[1] != chunk_size:
            vel = vel[:, -1:, :].expand(batch_size, chunk_size, 3)
    else:
        raise ValueError(f"Unsupported velocity shape: {tuple(vel.shape)}")
    return vel


def _broadcast_xyz(
    values_xyz: Optional[torch.Tensor],
    batch_size: int,
    chunk_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if values_xyz is None:
        return None
    xyz = torch.as_tensor(values_xyz, device=device, dtype=dtype)
    if xyz.shape[-1] < 3:
        raise ValueError(f"Expected XYZ last dimension >= 3, got {tuple(xyz.shape)}")
    xyz = xyz[..., :3]

    if xyz.ndim == 1:
        xyz = xyz.view(1, 1, 3).expand(batch_size, chunk_size, 3)
    elif xyz.ndim == 2:
        if xyz.shape[0] == batch_size:
            xyz = xyz[:, None, :].expand(batch_size, chunk_size, 3)
        elif xyz.shape[0] == chunk_size:
            xyz = xyz[None, :, :].expand(batch_size, chunk_size, 3)
        else:
            raise ValueError(
                "XYZ shape must be (B,3), (T,3), or (B,T,3); "
                f"got {tuple(xyz.shape)} for B={batch_size}, T={chunk_size}"
            )
    elif xyz.ndim == 3:
        if xyz.shape[0] != batch_size:
            raise ValueError(f"XYZ batch mismatch: {xyz.shape[0]} != {batch_size}")
        if xyz.shape[1] == 1:
            xyz = xyz.expand(batch_size, chunk_size, 3)
        elif xyz.shape[1] != chunk_size:
            xyz = xyz[:, -1:, :].expand(batch_size, chunk_size, 3)
    else:
        raise ValueError(f"Unsupported XYZ shape: {tuple(xyz.shape)}")
    return xyz


def _broadcast_horizon(
    horizon_s: Optional[torch.Tensor | float],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    default_horizon_s: float,
) -> torch.Tensor:
    if horizon_s is None:
        horizon = torch.full((batch_size,), float(default_horizon_s), device=device, dtype=dtype)
    else:
        horizon = torch.as_tensor(horizon_s, device=device, dtype=dtype)
        if horizon.ndim == 0:
            horizon = horizon.expand(batch_size)
        elif horizon.ndim == 1:
            if horizon.shape[0] == 1:
                horizon = horizon.expand(batch_size)
            elif horizon.shape[0] != batch_size:
                horizon = horizon[-1:].expand(batch_size)
        else:
            horizon = horizon.reshape(horizon.shape[0], -1)[:, -1]
            if horizon.shape[0] != batch_size:
                horizon = horizon[-1:].expand(batch_size)
    return torch.clamp(horizon, min=1e-6)


def _action_step_times(
    chunk_size: int,
    device: torch.device,
    dtype: torch.dtype,
    action_dt: float,
) -> torch.Tensor:
    return torch.arange(chunk_size, device=device, dtype=dtype) * float(action_dt)


def project_velocity_to_action_drift(
    velocity_xyz: Optional[torch.Tensor],
    action_shape: Tuple[int, int, int] | torch.Size,
    *,
    position_xyz: Optional[torch.Tensor] = None,
    lookahead_position_xyz: Optional[torch.Tensor] = None,
    lookahead_horizon_s: Optional[torch.Tensor | float] = None,
    action_std: Optional[torch.Tensor] = None,
    drift_scale: float = 1.0,
    drift_mode: str = "trajectory",
    action_dt: float = 1.0 / 25.0,
    lookahead_horizon_default_s: float = 1.0,
    axis_scale_xyz: Tuple[float, float, float] | torch.Tensor = (1.0, 1.0, 1.0),
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """Map object linear velocity into DynamicVLA action channels.

    ``drift_mode="velocity"`` preserves the original MVP behavior and injects raw
    velocity into every action step. ``drift_mode="trajectory"`` integrates
    velocity over the action-chunk time grid, producing per-step object
    displacement. ``drift_mode="lookahead"`` uses a filtered future object
    position and interpolates from current position to that future belief.
    Rotation and gripper channels are left at zero so the action expert remains
    responsible for dexterity.
    """
    batch_size, chunk_size, action_dim = tuple(action_shape)
    if device is None:
        for candidate in (velocity_xyz, position_xyz, lookahead_position_xyz, action_std):
            if torch.is_tensor(candidate):
                device = candidate.device
                break
        else:
            device = torch.device("cpu")
    drift = torch.zeros(batch_size, chunk_size, action_dim, device=device, dtype=dtype)
    if action_dim == 0:
        return drift

    n_xyz = min(3, action_dim)
    mode = str(drift_mode or "trajectory").lower()
    if mode == "velocity":
        if velocity_xyz is None:
            return None
        vel = _broadcast_velocity(velocity_xyz, batch_size, chunk_size, device, dtype)
        xyz_drift = vel[..., :n_xyz]
    elif mode == "trajectory":
        if velocity_xyz is None:
            return None
        vel = _broadcast_velocity(velocity_xyz, batch_size, chunk_size, device, dtype)
        step_times = _action_step_times(chunk_size, device, dtype, action_dt)
        xyz_drift = vel[..., :n_xyz] * step_times[None, :, None]
    elif mode == "lookahead":
        current_pos = _broadcast_xyz(position_xyz, batch_size, chunk_size, device, dtype)
        future_pos = _broadcast_xyz(
            lookahead_position_xyz, batch_size, chunk_size, device, dtype
        )
        step_times = _action_step_times(chunk_size, device, dtype, action_dt)
        if current_pos is not None and future_pos is not None:
            horizon = _broadcast_horizon(
                lookahead_horizon_s,
                batch_size,
                device,
                dtype,
                lookahead_horizon_default_s,
            )
            frac = torch.clamp(step_times[None, :] / horizon[:, None], 0.0, 1.0)
            xyz_drift = (future_pos[:, :1, :n_xyz] - current_pos[:, :1, :n_xyz]) * frac[:, :, None]
        else:
            if velocity_xyz is None:
                return None
            vel = _broadcast_velocity(velocity_xyz, batch_size, chunk_size, device, dtype)
            xyz_drift = vel[..., :n_xyz] * step_times[None, :, None]
    else:
        raise ValueError(
            f"Unknown residual drift mode: {drift_mode!r}. "
            "Expected 'trajectory', 'velocity', or 'lookahead'."
        )

    axis_scale = torch.as_tensor(axis_scale_xyz, device=device, dtype=dtype).flatten()
    if axis_scale.numel() < n_xyz:
        raise ValueError(
            f"axis_scale_xyz must provide at least {n_xyz} values, got {axis_scale.numel()}."
        )
    drift[..., :n_xyz] = xyz_drift * axis_scale[:n_xyz] * float(drift_scale)

    std = _as_action_std(action_std, action_dim, device, dtype)
    if std is not None:
        drift[..., :n_xyz] = drift[..., :n_xyz] / torch.clamp(std[:n_xyz], min=1e-6)

    return drift


def _broadcast_drift_mean(
    drift_mean: Optional[torch.Tensor | list | tuple],
    action_shape: Tuple[int, int, int] | torch.Size,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Broadcast a stored global drift mean to ``(B, H, D)``."""
    batch_size, chunk_size, action_dim = tuple(action_shape)
    if drift_mean is None:
        return torch.zeros(batch_size, chunk_size, action_dim, device=device, dtype=dtype)

    mean = torch.as_tensor(drift_mean, device=device, dtype=dtype)
    if mean.ndim == 1:
        if mean.numel() != action_dim:
            raise ValueError(
                f"1D drift mean must have action_dim={action_dim} values, got {mean.numel()}."
            )
        mean = mean.view(1, 1, action_dim).expand(batch_size, chunk_size, action_dim)
    elif mean.ndim == 2:
        if mean.shape != (chunk_size, action_dim):
            raise ValueError(
                "2D drift mean must have shape (chunk_size, action_dim), "
                f"got {tuple(mean.shape)} expected {(chunk_size, action_dim)}."
            )
        mean = mean.view(1, chunk_size, action_dim).expand(batch_size, chunk_size, action_dim)
    elif mean.ndim == 3:
        if mean.shape[1:] != (chunk_size, action_dim):
            raise ValueError(
                "3D drift mean must have trailing shape (chunk_size, action_dim), "
                f"got {tuple(mean.shape)} expected (*, {chunk_size}, {action_dim})."
            )
        if mean.shape[0] == 1:
            mean = mean.expand(batch_size, chunk_size, action_dim)
        elif mean.shape[0] != batch_size:
            raise ValueError(f"Drift mean batch mismatch: {mean.shape[0]} != {batch_size}")
    else:
        raise ValueError(f"Unsupported drift mean shape: {tuple(mean.shape)}")
    return mean


def centered_action_drift(
    velocity_xyz: Optional[torch.Tensor],
    action_shape: Tuple[int, int, int] | torch.Size,
    *,
    position_xyz: Optional[torch.Tensor] = None,
    lookahead_position_xyz: Optional[torch.Tensor] = None,
    lookahead_horizon_s: Optional[torch.Tensor | float] = None,
    action_std: Optional[torch.Tensor] = None,
    drift_mean: Optional[torch.Tensor | list | tuple] = None,
    center_drift: bool = True,
    drift_scale: float = 1.0,
    drift_mode: str = "trajectory",
    action_dt: float = 1.0 / 25.0,
    lookahead_horizon_default_s: float = 1.0,
    axis_scale_xyz: Tuple[float, float, float] | torch.Tensor = (1.0, 1.0, 1.0),
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(drift - d_bar, drift)`` in normalized action coordinates."""
    drift = project_velocity_to_action_drift(
        velocity_xyz,
        action_shape,
        position_xyz=position_xyz,
        lookahead_position_xyz=lookahead_position_xyz,
        lookahead_horizon_s=lookahead_horizon_s,
        action_std=action_std,
        drift_scale=drift_scale,
        drift_mode=drift_mode,
        action_dt=action_dt,
        lookahead_horizon_default_s=lookahead_horizon_default_s,
        axis_scale_xyz=axis_scale_xyz,
        device=device,
        dtype=dtype,
    )
    if drift is None:
        batch_size, chunk_size, action_dim = tuple(action_shape)
        if device is None:
            device = torch.device("cpu")
        drift = torch.zeros(batch_size, chunk_size, action_dim, device=device, dtype=dtype)

    if center_drift:
        mean = _broadcast_drift_mean(
            drift_mean,
            drift.shape,
            device=drift.device,
            dtype=drift.dtype,
        )
        return drift - mean, drift
    return drift, drift


def residualize_actions(
    actions: torch.Tensor,
    velocity_xyz: Optional[torch.Tensor],
    *,
    position_xyz: Optional[torch.Tensor] = None,
    lookahead_position_xyz: Optional[torch.Tensor] = None,
    lookahead_horizon_s: Optional[torch.Tensor | float] = None,
    action_std: Optional[torch.Tensor] = None,
    drift_mean: Optional[torch.Tensor | list | tuple] = None,
    center_drift: bool = True,
    drift_scale: float = 1.0,
    drift_mode: str = "trajectory",
    action_dt: float = 1.0 / 25.0,
    lookahead_horizon_default_s: float = 1.0,
    axis_scale_xyz: Tuple[float, float, float] | torch.Tensor = (1.0, 1.0, 1.0),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Subtract the centered analytic drift from normalized raw actions."""
    centered_drift, raw_drift = centered_action_drift(
        velocity_xyz,
        actions.shape,
        position_xyz=position_xyz,
        lookahead_position_xyz=lookahead_position_xyz,
        lookahead_horizon_s=lookahead_horizon_s,
        action_std=action_std,
        drift_mean=drift_mean,
        center_drift=center_drift,
        drift_scale=drift_scale,
        drift_mode=drift_mode,
        action_dt=action_dt,
        lookahead_horizon_default_s=lookahead_horizon_default_s,
        axis_scale_xyz=axis_scale_xyz,
        device=actions.device,
        dtype=actions.dtype,
    )
    return actions - centered_drift, centered_drift, raw_drift


def restore_residual_actions(
    residual_actions: torch.Tensor,
    velocity_xyz: Optional[torch.Tensor],
    *,
    position_xyz: Optional[torch.Tensor] = None,
    lookahead_position_xyz: Optional[torch.Tensor] = None,
    lookahead_horizon_s: Optional[torch.Tensor | float] = None,
    action_std: Optional[torch.Tensor] = None,
    drift_mean: Optional[torch.Tensor | list | tuple] = None,
    center_drift: bool = True,
    drift_scale: float = 1.0,
    drift_mode: str = "trajectory",
    action_dt: float = 1.0 / 25.0,
    lookahead_horizon_default_s: float = 1.0,
    axis_scale_xyz: Tuple[float, float, float] | torch.Tensor = (1.0, 1.0, 1.0),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add the centered analytic drift back to sampled residual actions."""
    centered_drift, raw_drift = centered_action_drift(
        velocity_xyz,
        residual_actions.shape,
        position_xyz=position_xyz,
        lookahead_position_xyz=lookahead_position_xyz,
        lookahead_horizon_s=lookahead_horizon_s,
        action_std=action_std,
        drift_mean=drift_mean,
        center_drift=center_drift,
        drift_scale=drift_scale,
        drift_mode=drift_mode,
        action_dt=action_dt,
        lookahead_horizon_default_s=lookahead_horizon_default_s,
        axis_scale_xyz=axis_scale_xyz,
        device=residual_actions.device,
        dtype=residual_actions.dtype,
    )
    return residual_actions + centered_drift, centered_drift, raw_drift


def _broadcast_lambda(
    time: torch.Tensor | float,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    lambda_type: str,
) -> torch.Tensor:
    # DynamicVLA uses reverse time: t=1 is source/noise and t=0 is target/action.
    # The residual_flow_2d schedule uses s=0 at source and s=1 at target.
    schedule_time = 1.0 - torch.as_tensor(time, device=device, dtype=dtype)
    lam = lambda_schedule(schedule_time, lambda_type)
    lam = torch.as_tensor(lam, device=device, dtype=dtype)
    if lam.ndim == 0:
        lam = lam.expand(batch_size)
    else:
        lam = lam.reshape(lam.shape[0], -1)[:, -1]
        if lam.shape[0] == 1:
            lam = lam.expand(batch_size)
        elif lam.shape[0] != batch_size:
            raise ValueError(
                f"Lambda schedule batch mismatch: {lam.shape[0]} != {batch_size}"
            )
    return lam.view(batch_size, 1, 1)


def scheduled_residual_drift(
    time: torch.Tensor | float,
    velocity_xyz: Optional[torch.Tensor],
    action_shape: Tuple[int, int, int] | torch.Size,
    *,
    lambda_type: str = "constant",
    position_xyz: Optional[torch.Tensor] = None,
    lookahead_position_xyz: Optional[torch.Tensor] = None,
    lookahead_horizon_s: Optional[torch.Tensor | float] = None,
    action_std: Optional[torch.Tensor] = None,
    drift_mean: Optional[torch.Tensor | list | tuple] = None,
    center_drift: bool = True,
    drift_scale: float = 1.0,
    drift_mode: str = "trajectory",
    action_dt: float = 1.0 / 25.0,
    lookahead_horizon_default_s: float = 1.0,
    axis_scale_xyz: Tuple[float, float, float] | torch.Tensor = (1.0, 1.0, 1.0),
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(lambda(1 - t) * centered_drift, centered_drift, raw_drift)``."""
    batch_size, _, _ = tuple(action_shape)
    if device is None:
        for candidate in (
            velocity_xyz,
            position_xyz,
            lookahead_position_xyz,
            action_std,
            drift_mean if torch.is_tensor(drift_mean) else None,
        ):
            if torch.is_tensor(candidate):
                device = candidate.device
                break
        else:
            device = torch.device("cpu")

    centered_drift, raw_drift = centered_action_drift(
        velocity_xyz,
        action_shape,
        position_xyz=position_xyz,
        lookahead_position_xyz=lookahead_position_xyz,
        lookahead_horizon_s=lookahead_horizon_s,
        action_std=action_std,
        drift_mean=drift_mean,
        center_drift=center_drift,
        drift_scale=drift_scale,
        drift_mode=drift_mode,
        action_dt=action_dt,
        lookahead_horizon_default_s=lookahead_horizon_default_s,
        axis_scale_xyz=axis_scale_xyz,
        device=device,
        dtype=dtype,
    )
    lam = _broadcast_lambda(
        time,
        batch_size,
        centered_drift.device,
        centered_drift.dtype,
        lambda_type,
    )
    return lam * centered_drift, centered_drift, raw_drift


def scheduled_residual_flow_target(
    noise: torch.Tensor,
    actions: torch.Tensor,
    time: torch.Tensor,
    velocity_xyz: Optional[torch.Tensor],
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the reverse-time residual vector-field target.

    DynamicVLA predicts ``d x_t / d t`` for increasing ``t`` where
    ``x_t = t * noise + (1 - t) * action``. Since sampling integrates backward
    from ``t=1`` to ``t=0``, the scheduled object-motion contribution is removed
    from the denoising vector during sampling.
    """
    lambda_type = kwargs.pop("lambda_type", "constant")
    scheduled_drift, centered_drift, raw_drift = scheduled_residual_drift(
        time,
        velocity_xyz,
        actions.shape,
        device=actions.device,
        dtype=actions.dtype,
        lambda_type=lambda_type,
        **kwargs,
    )
    target = noise - actions + scheduled_drift
    return target, scheduled_drift, centered_drift, raw_drift


def remove_scheduled_residual_drift(
    predicted_velocity: torch.Tensor,
    time: torch.Tensor | float,
    velocity_xyz: Optional[torch.Tensor],
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Subtract scheduled drift from a reverse-time denoising vector."""
    lambda_type = kwargs.pop("lambda_type", "constant")
    scheduled_drift, centered_drift, _ = scheduled_residual_drift(
        time,
        velocity_xyz,
        predicted_velocity.shape,
        device=predicted_velocity.device,
        dtype=predicted_velocity.dtype,
        lambda_type=lambda_type,
        **kwargs,
    )
    return predicted_velocity - scheduled_drift, scheduled_drift, centered_drift


def residual_flow_target(
    noise: torch.Tensor,
    actions: torch.Tensor,
    time: torch.Tensor,
    velocity_xyz: Optional[torch.Tensor],
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Legacy compatibility wrapper for residual-vector-field experiments.

    Returns the scheduled target and scheduled drift using the same sign
    convention as the main DynamicVLA residual vector-field implementation.
    """
    target, scheduled_drift, _, _ = scheduled_residual_flow_target(
        noise,
        actions,
        time,
        velocity_xyz,
        **kwargs,
    )
    return target, scheduled_drift


def apply_residual_drift(
    predicted_velocity: torch.Tensor,
    time: torch.Tensor,
    velocity_xyz: Optional[torch.Tensor],
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Legacy compatibility wrapper for old vector-field drift injection."""
    corrected_velocity, scheduled_drift, _ = remove_scheduled_residual_drift(
        predicted_velocity,
        time,
        velocity_xyz,
        **kwargs,
    )
    return corrected_velocity, scheduled_drift


def extract_dom_dynamic_fields(
    environment_state: torch.Tensor,
    *,
    covariance_std: float = DEFAULT_COVARIANCE_STD,
    lookahead_horizon_s: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Extract object pose, velocity, and default covariance from DOM env state."""
    env = torch.as_tensor(environment_state)
    if env.shape[-1] < 9:
        raise ValueError(f"DOM environment_state must have at least 9 dims, got {tuple(env.shape)}")
    position = env[..., :3]
    velocity = env[..., 6:9]
    lookahead_horizon = torch.full(
        env.shape[:-1],
        float(lookahead_horizon_s),
        dtype=env.dtype,
        device=env.device,
    )
    lookahead_position = position + velocity * lookahead_horizon[..., None]

    eye = torch.eye(3, dtype=env.dtype, device=env.device) * float(covariance_std) ** 2
    covariance = eye.expand(*env.shape[:-1], 3, 3).clone()
    return {
        "dynamic_object.position_xyz": position,
        "dynamic_object.velocity_xyz": velocity,
        "dynamic_object.covariance_xyz": covariance,
        "dynamic_object.lookahead_position_xyz": lookahead_position,
        "dynamic_object.lookahead_horizon_s": lookahead_horizon,
        "dynamic_object.lookahead_covariance_xyz": covariance.clone(),
    }
