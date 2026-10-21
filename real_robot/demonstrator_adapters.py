"""State/target adapters for the shared automatic demonstrator; no devices opened."""
import json
from pathlib import Path

import numpy as np

from .automatic_demonstrator import GraspPrior, ObjectState, Observation
from .geometry import validate_transform
from .grasp_prior import fresh_observation


def template_reference(template):
    """Keep the collection heading as a prior, not a claimed measured rotation."""
    if 'object_reference_pose' in template:
        return validate_transform(template['object_reference_pose']).copy()
    source = Path(template['source_episode'])
    episode = json.loads(source.read_text())
    return validate_transform(episode['T_base_tape_reference']).copy()


def prior_from_template(template):
    reference = template_reference(template)
    inverse = np.linalg.inv(reference)
    return GraspPrior(inverse@np.asarray(template['stages']['pregrasp']['T_base_tcp']),
                      inverse@np.asarray(template['stages']['before_close']['T_base_tcp']),
                      semantics='fused_visible_surface_proxy')


def release_from_registration(template, box):
    release = validate_transform(template['stages']['release']['T_base_tcp']).copy()
    release[:2, 3] = box['release_xy_m']
    release[2, 3] = max(release[2, 3], box['rim_z_base_m']+.065)
    return release


class EstimatedObjectAdapter:
    """Same teacher-state interface as a simulator, with explicit proxy origin."""
    def __init__(self, template):
        self.rig_sha = template['rig_sha256']
        self.reference = template_reference(template)

    def from_record(self, record, now_s):
        point = fresh_observation(record, self.rig_sha, now_s)
        pose = self.reference.copy()
        pose[:3, 3] = point
        track = record.get('track') or {}
        velocity = np.asarray(track.get('velocity_mps', [0., 0., 0.]), float)
        if track.get('frame') != 'right_base_estimate' or not track.get('ready'):
            velocity = np.zeros(3)
        return ObjectState(pose, velocity,
            record['timing']['oldest_host_arrival_monotonic_s'],
            semantics='fused_visible_surface_proxy')

    def teacher_metadata(self, state):
        return {'pose': state.pose.tolist(), 'velocity_m_s': state.velocity_m_s.tolist(),
                'stamp_s': state.stamp_s, 'frame': state.frame, 'origin': state.semantics,
                'orientation_source': 'collection heading prior; refined by wrist alignment',
                'policy_input': False}


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, float)


def pose_from_wxyz(position, quaternion):
    q = np.asarray(quaternion, float)
    p = np.asarray(position, float)
    if q.shape != (4,) or p.shape != (3,) or not np.isfinite([*p, *q]).all() or np.linalg.norm(q) < 1e-9:
        raise ValueError('Expected finite XYZ and nonzero wxyz quaternion')
    w, x, y, z = q/np.linalg.norm(q)
    result = np.eye(4)
    result[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    result[:3, 3] = p
    return result


def action_wxyz(action):
    """Isaac's absolute pose action [xyz, quaternion(wxyz), gripper(+1/-1)]."""
    from scipy.spatial.transform import Rotation
    xyzw = Rotation.from_matrix(action.tcp_pose[:3, :3]).as_quat()
    # HOLD and CLOSE both request the existing binary closed state in Isaac.
    return np.r_[action.tcp_pose[:3, 3], xyzw[[3, 0, 1, 2]], 1. if action.gripper == 'OPEN' else -1.]


class IsaacDemonstratorAdapter:
    """Bridge DynamicVLA get_curr_state() to the identical NumPy task core.

    DynamicVLA's state dictionary is already in the robot's base frame. The
    caller supplies simulator grasp/holding/placement feedback explicitly.
    No Isaac imports are needed here; tensor conversion is duck-typed.
    """
    def __init__(self, demonstrator, *, env_index=0, T_object_prior_origin=None):
        self.demonstrator = demonstrator
        self.env_index = int(env_index)
        if demonstrator.prior.semantics != 'object_center' and T_object_prior_origin is None:
            raise ValueError('A proxy-origin prior needs an explicit object-to-prior-origin transform in simulation')
        self.origin = validate_transform(np.eye(4) if T_object_prior_origin is None else T_object_prior_origin).copy()

    def _item(self, value):
        return _array(value)[self.env_index]

    def step(self, curr_state, now_s, *, gripper_open, grasp_contact, holding, placed):
        ee, obj = curr_state['end_effector'], curr_state['object']
        obj_pose = pose_from_wxyz(self._item(obj['pos']), self._item(obj['quat']))@self.origin
        velocity = self._item(obj['velocity'])
        if np.linalg.norm(self.origin[:3, 3]) > 1e-9:
            # Offset origins also move under rotation. Never silently omit it.
            if 'angular_velocity' not in obj:
                raise ValueError('Offset simulation origins require angular_velocity')
            center = self._item(obj['pos'])
            velocity = velocity+np.cross(self._item(obj['angular_velocity']), obj_pose[:3, 3]-center)
        state = ObjectState(obj_pose, velocity, now_s, semantics=self.demonstrator.prior.semantics)
        approach, grasp = self.demonstrator.grasp_targets(state)
        obs = Observation(now_s,
            pose_from_wxyz(self._item(ee['pos']), self._item(ee['quat'])), state,
            gripper_open=bool(gripper_open), grasp_contact=bool(grasp_contact),
            holding=bool(holding), placed=bool(placed), confirmation_stamp=now_s,
            alignment_complete=True, alignment_target=approach, aligned_grasp=grasp)
        action = self.demonstrator.step(obs)
        return {'action': action_wxyz(action), 'sm_state': action.stage.value,
                'teacher_action': action}
