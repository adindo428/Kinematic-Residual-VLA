"""Model-based hover preview. No serial, ROS, LeRobot, or actuator imports."""
from dataclasses import dataclass
import numpy as np


def _vector(value):
    a = np.asarray(value, float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError('Expected finite XYZ')
    return a


def _limit(value, maximum):
    norm = np.linalg.norm(value)
    return value * min(1., maximum / max(norm, 1e-12))


def hover_preview_target(track, height_m=0.05, horizon_s=0.10):
    """Visual target only, including provisional base frames; never a robot action.

    Height is along base +Z, measured from the visible tape surface proxy.
    A prediction during occlusion is deliberately not exposed as a target.
    """
    if (not np.isfinite([height_m, horizon_s]).all()
            or not 0 < height_m <= .30 or not 0 <= horizon_s <= .25):
        raise ValueError('Preview height must be in (0, .30] m and horizon in [0, .25] s')
    result = {'mode': 'preview_only', 'motion_sent': False, 'target_m': None,
              'height_m': height_m, 'prediction_horizon_s': horizon_s,
              'position_semantics': 'visible_surface_proxy_plus_base_z'}
    if track is None or not track.ready or not track.accepted or track.measurement_age_s > .12:
        return {**result, 'reason': 'missing_or_unready_track'}
    if track.frame not in ('robot_base', 'right_base_estimate', 'left_base_estimate'):
        return {**result, 'reason': 'base_frame_required'}
    projection = np.c_[np.eye(3), horizon_s * np.eye(3)]
    covariance = projection @ track.covariance @ projection.T
    if (not np.isfinite(covariance).all()
            or np.sqrt(max(0., np.linalg.eigvalsh(covariance).max())) > .025):
        return {**result, 'reason': 'prediction_uncertain'}
    target = track.position_m + horizon_s * track.velocity_mps + [0, 0, height_m]
    if not np.isfinite(target).all():
        return {**result, 'reason': 'nonfinite_target'}
    return {**result, 'reason': None, 'target_m': target.tolist()}


@dataclass
class HoverLimits:
    workspace_min_m: list
    workspace_max_m: list
    hover_height_m: float = 0.05
    prediction_horizon_s: float = 0.10
    max_observation_age_s: float = 0.12
    max_speed_mps: float = 0.03
    max_acceleration_mps2: float = 0.10
    kp_per_s: float = 2.0
    max_prediction_std_m: float = 0.025
    max_step_s: float = 0.10

    def __post_init__(self):
        lo, hi = _vector(self.workspace_min_m), _vector(self.workspace_max_m)
        if np.any(lo >= hi):
            raise ValueError('Workspace lower bound must be below upper bound')
        values = [v for k, v in vars(self).items() if not k.startswith('workspace')]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('All controller thresholds must be finite and positive')


class ShadowHoverController:
    """Candidates only: a real adapter still needs IK, collision and watchdog checks."""
    def __init__(self, limits):
        self.limits = limits
        self.previous_velocity = np.zeros(3)

    def step(self, track, ee_position_m, now_s, dt_s):
        cfg = self.limits
        ee = _vector(ee_position_m)
        result = {'mode': 'shadow_only', 'motion_sent': False,
                  'candidate_velocity_mps': None, 'candidate_target_m': None}
        def hold(reason):
            self.previous_velocity[:] = 0
            return {**result, 'status': 'hold', 'reason': reason}
        if not np.isfinite([now_s, dt_s]).all() or not 0 < dt_s <= cfg.max_step_s:
            return hold('invalid_control_timing')
        if track is None or track.frame != 'robot_base' or not track.ready:
            return hold('unready_or_uncalibrated_track')
        age = now_s - track.timestamp_s
        if age < 0 or age + track.measurement_age_s > cfg.max_observation_age_s:
            return hold('stale_or_future_observation')
        lo, hi = _vector(cfg.workspace_min_m), _vector(cfg.workspace_max_m)
        if np.any(ee < lo) or np.any(ee > hi):
            return hold('end_effector_outside_workspace')
        horizon = age + cfg.prediction_horizon_s
        target = track.position_m + horizon * track.velocity_mps
        target = target + [0, 0, cfg.hover_height_m]
        projection = np.c_[np.eye(3), horizon * np.eye(3)]
        covariance = projection @ track.covariance @ projection.T
        if np.sqrt(np.linalg.eigvalsh(covariance).max()) > cfg.max_prediction_std_m:
            return hold('prediction_uncertain')
        if not np.isfinite(target).all() or np.any(target < lo) or np.any(target > hi):
            return hold('predicted_target_outside_workspace')
        velocity = _limit(track.velocity_mps + cfg.kp_per_s * (target - ee), cfg.max_speed_mps)
        velocity = self.previous_velocity + _limit(velocity - self.previous_velocity,
                                                   cfg.max_acceleration_mps2 * dt_s)
        next_position = ee + dt_s * velocity
        if np.any(next_position < lo) or np.any(next_position > hi):
            return hold('candidate_step_outside_workspace')
        self.previous_velocity = velocity
        return {**result, 'status': 'candidate', 'reason': None,
                'candidate_velocity_mps': velocity.tolist(),
                'candidate_target_m': target.tolist(),
                'candidate_next_position_m': next_position.tolist()}


def damped_joint_velocity(jacobian, twist, damping, max_joint_speed):
    """Reference DLS solver only; supply a calibrated, embodiment-specific Jacobian.

    The caller chooses feasible task rows (SO101 cannot realize arbitrary 6D
    twists). This alone is NOT a collision-aware or executable robot adapter.
    """
    j, v = np.asarray(jacobian, float), np.asarray(twist, float)
    if (j.ndim != 2 or v.shape != (j.shape[0],) or j.shape[0] == 0
            or j.shape[1] == 0 or not np.isfinite(j).all() or not np.isfinite(v).all()
            or not np.isfinite([damping, max_joint_speed]).all()
            or damping <= 0 or max_joint_speed <= 0):
        raise ValueError('Invalid Jacobian, twist or bounds')
    qdot = j.T @ np.linalg.solve(j @ j.T + damping ** 2 * np.eye(j.shape[0]), v)
    scale = min(1., max_joint_speed / max(np.max(np.abs(qdot)), 1e-12))
    return qdot * scale
