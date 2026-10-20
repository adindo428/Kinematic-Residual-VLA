"""Simulator-independent pose execution and metrics for live tracking evaluation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def limit_norm(value, limit):
    value = np.asarray(value, dtype=float)
    return value * min(1., limit / max(np.linalg.norm(value), 1e-12))


class PosePlan:
    """Future measured observation-tool poses, executed with their original time labels.

    Anchor at the measured pose at replan time; first model pose is +0.2 s.
    Linear XYZ interpolation and shortest-path SLERP avoid Euler branch artifacts.
    Replanning can replace the remainder of this plan after 20 camera frames.
    """
    def __init__(self, start_time, current_pose, chunk, label_horizon=.2, action_dt=.04):
        chunk = np.asarray(chunk, dtype=float)
        current_pose = np.asarray(current_pose, dtype=float)
        if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) < 2:
            raise ValueError('Expected a finite [chunk,7] absolute pose/gripper chunk')
        if not np.isfinite(chunk).all() or not np.isfinite(current_pose).all():
            raise ValueError('Nonfinite policy pose')
        if label_horizon <= 0 or action_dt <= 0:
            raise ValueError('Positive physical time intervals required')
        self.times = np.r_[start_time, start_time + label_horizon + np.arange(len(chunk))*action_dt]
        self.xyz = np.vstack([current_pose[:3], chunk[:, :3]])
        self.rotations = Rotation.from_euler('xyz', np.vstack([current_pose[3:6], chunk[:, 3:6]]))
        self.slerp = Slerp(self.times, self.rotations)

    def sample(self, time):
        t = float(np.clip(time, self.times[0], self.times[-1]))
        i = min(max(np.searchsorted(self.times, t, side='right')-1, 0), len(self.times)-2)
        dt = self.times[i+1]-self.times[i]
        v = (self.xyz[i+1]-self.xyz[i])/dt
        xyz = self.xyz[i] + (t-self.times[i])*v
        omega = (self.rotations[i+1]*self.rotations[i].inv()).as_rotvec()/dt
        if time > self.times[-1]:
            v = np.zeros(3); omega = np.zeros(3)
        return xyz, self.slerp(t), v, omega


class TrackingTermination(ValueError):
    """A policy-driven physical limit ends this rollout, not the benchmark."""


def pose_command(state, desired, jacobian, joint_limits, velocity_limits, cfg, dt):
    """Damped IK servo; only learned pose-plan feedback/feedforward, no object velocity.

    State and target are at the observation tool. Jacobian is at the action tool.
    Gripper is held open by IsaacBackend.apply, matching this task's data.
    """
    xyz, desired_r, desired_v, desired_omega = desired
    current_r = Rotation.from_quat(np.asarray(state['ee_quaternion'])[[1,2,3,0]])
    error = xyz - state['ee_position']
    raw_v = desired_v + cfg['kp']*error + cfg['kd']*(desired_v-state['ee_velocity'])
    raw_omega = desired_omega + cfg['orientation_kp']*(desired_r*current_r.inv()).as_rotvec()
    v = limit_norm(raw_v, cfg['max_linear_speed'])
    omega = limit_norm(raw_omega, cfg['max_angular_speed'])
    delta = current_r.apply(np.asarray(cfg['action_offset']) - cfg['observation_offset'])
    twist = np.r_[v+np.cross(omega, delta), omega]
    j = np.asarray(jacobian, dtype=float)
    q = np.asarray(state['joint_position'])
    if not all(np.isfinite(x).all() for x in (j, twist, q)):
        raise ValueError('Nonfinite servo input')
    requested = j.T @ np.linalg.solve(j @ j.T + cfg['damping']**2*np.eye(6), twist)
    lower = np.maximum(-np.asarray(velocity_limits), (joint_limits[:,0]+cfg['joint_margin']-q)/dt)
    upper = np.minimum(velocity_limits, (joint_limits[:,1]-cfg['joint_margin']-q)/dt)
    if np.any(lower > upper):
        raise TrackingTermination('Robot outside recoverable joint limits')
    applied = np.clip(requested, lower, upper)
    if not np.isfinite(applied).all():
        raise ValueError('Nonfinite IK command')
    return {'joint_position_target': (q+dt*applied).tolist(),
            'joint_velocity_target': applied.tolist(),
            'saturated': bool(not np.allclose(applied,requested) or not np.allclose(v,raw_v) or not np.allclose(omega,raw_omega)),
            'servo_position_error_m': float(np.linalg.norm(error))}


def summarize_rollout(rows, quality, train_speed_max, completed):
    if not rows:
        return {'completed': False, 'success': False, 'samples': 0}
    errors = np.array([r['hover_error_m'] for r in rows])
    angles = np.array([r['orientation_error_deg'] for r in rows])
    settled = np.array([r['settled'] for r in rows], dtype=bool)
    speed = np.array([r['object_speed_mps'] for r in rows])
    max_settled = float(angles[settled].max()) if settled.any() else None
    result = {'completed': bool(completed), 'samples': len(rows),
        'hover_rmse_m': float(np.sqrt(np.mean(errors**2))), 'hover_p95_m': float(np.quantile(errors,.95)),
        'hover_max_m': float(errors.max()), 'position_within_2cm_fraction': float(np.mean(errors<=.02)),
        'orientation_rmse_deg': float(np.sqrt(np.mean(angles**2))),
        'settled_orientation_max_deg': max_settled,
        'settled_pose_within_tolerance_fraction': float(np.mean((errors[settled]<=quality['rmse']) & (angles[settled]<=quality['orientation_degrees']))) if settled.any() else None,
        'max_contact_force_n': float(max(r['contact_force_n'] for r in rows)),
        'joint_limit_violation': any(r['joint_limit_violation'] for r in rows),
        'workspace_violation': any(r['workspace_violation'] for r in rows),
        'saturation_fraction': float(np.mean([r['saturated'] for r in rows])),
        'speed_mps': dict(zip(('min','median','p95','max'),map(float,np.quantile(speed,[0,.5,.95,1])))),
        'fraction_above_train_max_speed': float(np.mean(speed>train_speed_max))}
    result['success'] = bool(completed and result['hover_rmse_m']<=quality['rmse'] and result['hover_max_m']<=quality['max_error']
        and max_settled is not None and max_settled<=quality['orientation_degrees']
        and result['max_contact_force_n']<=quality['contact_force'] and not result['joint_limit_violation'] and not result['workspace_violation'])
    return result
