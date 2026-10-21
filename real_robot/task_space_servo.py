"""Shared Cartesian feedback/feedforward -> differential IK -> joint goals.

No device access. A single instance retains joint goals and velocity across task
phases. Integrating measured-pose corrections into the retained goals preserves
holding offsets without a separate, slowly settling nominal-pose integrator.
"""
import cv2
import numpy as np

from .geometry import validate_transform
from .vx300s_kinematics import forward, jacobian, LOWER, UPPER


def bounded(vector, magnitude):
    return vector*min(1., magnitude/max(float(np.linalg.norm(vector)), 1e-12))


class PoseVelocityServo:
    def __init__(self, actual, retained, tcp_offset, *, kp=4., ki=0., kd=0.,
                 integral_limit=.02, rotation_kp=4., max_following_error_deg=6.,
                 hardware_joint_rates_deg_s=None):
        actual = np.asarray(actual, float)
        self.goal = np.asarray(retained, float).copy()
        self.offset = np.asarray(tcp_offset, float)
        forward(actual, self.offset)
        forward(self.goal, self.offset)
        if np.any(self.goal < LOWER) or np.any(self.goal > UPPER):
            raise ValueError('Retained joint goal outside model limits')
        gains = [kp, ki, kd, integral_limit, rotation_kp, max_following_error_deg]
        if not np.isfinite(gains).all() or min(gains) < 0 or min(kp, rotation_kp, max_following_error_deg) <= 0:
            raise ValueError('Invalid Cartesian servo gains')
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_limit, self.rotation_kp = integral_limit, rotation_kp
        self.max_following_error = np.deg2rad(max_following_error_deg)
        self.hardware_rates = None
        if hardware_joint_rates_deg_s is not None:
            rates = np.asarray(hardware_joint_rates_deg_s, float)
            if rates.shape != (6,) or not np.isfinite(rates).all() or np.any(rates <= 0):
                raise ValueError('Expected six positive motor velocity/profile limits')
            self.hardware_rates = np.deg2rad(rates)
        self.velocity = np.zeros(3)
        self.measured_velocity = np.zeros(3)
        self.integral = np.zeros(3)
        self.previous_position = None
        self.diagnostics = {}

    def hold(self):
        self.velocity[:] = 0
        self.integral[:] = 0
        self.measured_velocity[:] = 0
        self.previous_position = None
        return self.goal.copy()

    def step(self, target_pose, actual, dt, *, tcp_rate_m_s, joint_rate_deg_s,
             angular_rate_deg_s, acceleration_m_s2=None, target_velocity_m_s=None,
             orientation_weight=1.):
        target = validate_transform(target_pose)
        actual = np.asarray(actual, float)
        if actual.shape != (6,) or not np.isfinite(actual).all() or not 0 < dt <= .15:
            raise ValueError('Invalid joint feedback or control interval')
        scalars = [tcp_rate_m_s, angular_rate_deg_s, orientation_weight]
        if not np.isfinite(scalars).all() or min(scalars) <= 0:
            raise ValueError('Cartesian trajectory rates/weight must be positive')
        if acceleration_m_s2 is not None and (not np.isfinite(acceleration_m_s2) or acceleration_m_s2 <= 0):
            raise ValueError('Trajectory acceleration must be positive or None')
        rates = np.asarray(joint_rate_deg_s, float)
        if rates.ndim == 0:
            rates = np.full(6, float(rates))
        if rates.shape != (6,) or not np.isfinite(rates).all() or np.any(rates <= 0):
            raise ValueError('Expected positive joint trajectory rates')
        rates = np.deg2rad(rates)
        if self.hardware_rates is not None:
            rates = np.minimum(rates, self.hardware_rates)
        feedforward = np.zeros(3) if target_velocity_m_s is None else np.asarray(target_velocity_m_s, float)
        if feedforward.shape != (3,) or not np.isfinite(feedforward).all():
            raise ValueError('Expected finite target velocity XYZ')
        following_error = self.goal-actual
        if np.max(np.abs(following_error)) > self.max_following_error:
            raise RuntimeError('Joint following error exceeds configured stall bound')
        pose = forward(actual, self.offset)
        position = pose[:3, 3]
        if self.previous_position is not None:
            observed = (position-self.previous_position)/dt
            self.measured_velocity += dt/(.12+dt)*(observed-self.measured_velocity)
        self.previous_position = position.copy()
        error = target[:3, 3]-position
        self.integral = np.clip(self.integral+error*dt, -self.integral_limit, self.integral_limit)
        desired_velocity = bounded(feedforward+self.kp*error+self.ki*self.integral
                                   -self.kd*self.measured_velocity, tcp_rate_m_s)
        if acceleration_m_s2 is None:
            self.velocity = desired_velocity
        else:
            self.velocity += bounded(desired_velocity-self.velocity, acceleration_m_s2*dt)
            self.velocity = bounded(self.velocity, tcp_rate_m_s)
        rotation_error = cv2.Rodrigues(target[:3, :3]@pose[:3, :3].T)[0].ravel()
        angular_velocity = bounded(self.rotation_kp*rotation_error, np.deg2rad(angular_rate_deg_s))
        j = jacobian(actual, self.offset)
        weighted = j.copy()
        weighted[3:] *= .1*orientation_weight
        desired = np.r_[self.velocity, .1*orientation_weight*angular_velocity]*dt
        singular = np.linalg.svd(weighted, compute_uv=False)
        damping = .01+max(0., .025-singular[-1])
        delta = weighted.T@np.linalg.solve(weighted@weighted.T+damping*damping*np.eye(6), desired)
        delta *= min(1., float(np.min(rates*dt/np.maximum(np.abs(delta), 1e-12))))
        delta = np.clip(self.goal+delta, LOWER, UPPER)-self.goal
        old_position = forward(self.goal, self.offset)[:3, 3]
        for _ in range(5):
            distance = np.linalg.norm(forward(self.goal+delta, self.offset)[:3, 3]-old_position)
            if distance <= tcp_rate_m_s*dt+1e-10:
                break
            delta *= tcp_rate_m_s*dt/distance
        candidate = self.goal+delta
        self.diagnostics = {
            'backend': 'shared_pose_velocity_servo', 'position_error_m': float(np.linalg.norm(error)),
            'rotation_error_deg': float(np.rad2deg(np.linalg.norm(rotation_error))),
            'minimum_singular_value': float(singular[-1]), 'damping': float(damping),
            'following_error_rad': following_error.tolist(),
            'target_velocity_m_s': feedforward.tolist(), 'requested_velocity_m_s': self.velocity.tolist(),
            'measured_velocity_m_s': self.measured_velocity.tolist(),
            'joint_trajectory_rates_deg_s': np.rad2deg(rates).tolist()}
        return candidate

    def accept(self, candidate):
        candidate = np.asarray(candidate, float)
        if candidate.shape != (6,) or not np.isfinite(candidate).all() or np.any(candidate < LOWER) or np.any(candidate > UPPER):
            raise ValueError('Invalid accepted joint goal')
        self.goal = candidate.copy()
