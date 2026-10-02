from dataclasses import dataclass

import numpy as np

from .math3d import angle_error, limit_norm, mode_quaternion, rotation, shift_point


@dataclass
class ModeSelector:
    enter: float = 0.05
    exit: float = 0.03
    mode: int = 0

    def __post_init__(self):
        if not 0 <= self.exit < self.enter or self.mode not in (-45, 0, 45):
            raise ValueError("Invalid hysteresis settings")

    def update(self, vy):
        if not np.isfinite(vy):
            raise ValueError("Nonfinite lateral velocity")
        if vy > self.enter:
            self.mode = -45
        elif vy < -self.enter:
            self.mode = 45
        elif self.mode == -45 and vy < self.exit:
            self.mode = 0
        elif self.mode == 45 and vy > -self.exit:
            self.mode = 0
        return self.mode


class Expert:
    def __init__(self, cfg, dt):
        self.cfg, self.dt = cfg, dt
        self.selector = ModeSelector(cfg["enter_speed"], cfg["exit_speed"])

    def command(self, state, target, jacobian, joint_limits, velocity_limits):
        """State, twist and Jacobian use base axes; target is the hover anchor."""
        k = self.cfg
        for value in (jacobian, joint_limits, velocity_limits, target["position"], target["velocity"]):
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite controller input")
        mode = self.selector.update(target["velocity"][1])
        desired_q = mode_quaternion(mode, k["down_quaternion"])
        e = np.asarray(target["position"]) - state["ee_position"]
        raw_v = np.asarray(target["velocity"]) + k["kp"]*e + k["kd"]*(np.asarray(target["velocity"])-state["ee_velocity"])
        v = limit_norm(raw_v, k["max_linear_speed"])
        if "workspace_lower" in target:
            # Prevent the requested next position leaving the tracking workspace.
            # During warm-up outside the box, allow bounded inward recovery only.
            lower_v = np.minimum((np.asarray(target["workspace_lower"])-state["ee_position"])/self.dt, 0)
            upper_v = np.maximum((np.asarray(target["workspace_upper"])-state["ee_position"])/self.dt, 0)
            v = np.clip(v, lower_v, upper_v)
        raw_omega = k["orientation_kp"]*angle_error(desired_q, state["ee_quaternion"])
        omega = limit_norm(raw_omega, k["max_angular_speed"])
        # Control at the action tool point while tracking the observation tool point.
        delta = rotation(state["ee_quaternion"]).apply(np.array(k["action_offset"])-k["observation_offset"])
        twist = np.r_[v + np.cross(omega, delta), omega]
        j = np.asarray(jacobian)
        dq = j.T @ np.linalg.solve(j @ j.T + k["damping"]**2*np.eye(6), twist)
        q = np.asarray(state["joint_position"])
        lower = np.maximum(-np.asarray(velocity_limits), (joint_limits[:, 0]+k["joint_margin"]-q)/self.dt)
        upper = np.minimum(velocity_limits, (joint_limits[:, 1]-k["joint_margin"]-q)/self.dt)
        if np.any(lower > upper):
            raise ValueError("Robot outside recoverable joint limits")
        applied = np.clip(dq, lower, upper)
        if not np.isfinite(applied).all():
            raise ValueError("Nonfinite IK solution")
        action_point = shift_point(state["ee_position"], state["ee_quaternion"], k["observation_offset"], k["action_offset"])
        return dict(mode=mode, desired_quaternion=desired_q.tolist(), position_error=e.tolist(),
                    requested_linear_velocity=raw_v.tolist(), commanded_twist=twist.tolist(),
                    requested_joint_velocity=dq.tolist(), joint_velocity_target=applied.tolist(),
                    joint_position_target=(q+self.dt*applied).tolist(),
                    immediate_action_position=(action_point+self.dt*twist[:3]).tolist(),
                    saturated=bool(not np.allclose(dq, applied) or not np.allclose(raw_v, v) or not np.allclose(raw_omega, omega)),
                    orientation_error=float(np.linalg.norm(angle_error(desired_q, state["ee_quaternion"]))))
