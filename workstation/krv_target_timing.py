"""Inference-only target-lead intervention in physical Cartesian action units.

The checkpoint already restores its h*dt drift. Add only the missing lead:
gate[h] * lead_s * (v - training_mean_v). This is equivalent to changing
the centered reconstruction horizon to lead_s+h*dt; it does not retrain the
expert in those new residual coordinates or add inference latency twice.
"""
import math


def lead_correction(policy, batch, actions, lead_s, gate):
    import torch
    lead_s = float(lead_s)
    if not math.isfinite(lead_s) or lead_s < 0:
        raise ValueError('Target lead must be finite and nonnegative')
    correction = torch.zeros_like(actions)
    if lead_s == 0:
        return correction
    cfg = policy.config
    if not (cfg.enable_residual_flow and cfg.residual_flow_formulation == 'coordinate'
            and cfg.residual_drift_mode == 'trajectory' and cfg.residual_normalize_drift):
        raise ValueError('Target-lead diagnostic requires normalized trajectory-coordinate KR')
    if not math.isfinite(cfg.residual_action_dt) or cfg.residual_action_dt <= 0:
        raise ValueError('Invalid physical action period')
    velocity = policy.prepare_residual_velocity(batch, actions.device, actions.dtype)
    if velocity is None:
        raise ValueError('Missing current velocity')
    axis = torch.as_tensor(cfg.residual_axis_scale_xyz, device=actions.device, dtype=actions.dtype)
    physical = velocity * axis * float(cfg.residual_drift_scale)
    if cfg.residual_center_drift:
        # Recover the exact training drift slope from serialized centering,
        # rather than fitting statistics from evaluation episodes.
        mean = torch.as_tensor(cfg.residual_drift_mean, device=actions.device, dtype=actions.dtype)
        if mean.shape != (cfg.chunk_size, cfg.max_action_dim) or cfg.chunk_size < 2:
            raise ValueError('Expected saved horizon-dependent training drift mean')
        std = policy.get_action_normalization_std(actions.device, actions.dtype)
        if std is None or not torch.isfinite(std[:3]).all() or (std[:3] <= 0).any():
            raise ValueError('Missing checkpoint action normalization')
        slope = (mean[1, :3] - mean[0, :3]) / cfg.residual_action_dt
        grid = torch.arange(cfg.chunk_size, device=actions.device, dtype=actions.dtype)
        torch.testing.assert_close(mean[:, :3], mean[:1, :3] + grid[:, None] * cfg.residual_action_dt * slope,
                                   rtol=1e-5, atol=1e-6)
        physical = physical - slope * std[:3]
    correction[..., :3] = physical[:, None, :] * lead_s
    if gate is not None:
        if gate.shape != (*actions.shape[:2], 1) or not torch.isfinite(gate).all() or (gate < 0).any() or (gate > 1).any():
            raise ValueError('Invalid grasp gate')
        correction = correction * gate.to(device=actions.device, dtype=actions.dtype)
    if not torch.isfinite(correction).all():
        raise ValueError('Nonfinite target-lead correction')
    return correction
