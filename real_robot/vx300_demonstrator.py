"""VX300 task-space servo and wrist/gripper adapter for the shared demonstrator.

No devices are opened here. The executor owns the existing serial connection.
"""
import cv2
import numpy as np

from .automatic_demonstrator import AutomaticDemonstrator, DemonstratorConfig, Observation, Stage
from .demonstrator_adapters import EstimatedObjectAdapter, prior_from_template, release_from_registration
from .geometry import validate_transform
from .pick_geometry import inside_polygon, pose_error, rx
from .vx300s_kinematics import forward
from .wrist_alignment import corrections, line_error


def manipulation_motion(template, values=None, scale=1.5):
    """Resolve phase motion targets; values are rates, not measured-speed stops."""
    defaults = {
        'precision': {'joint_rate_deg_s':template['limits']['joint_rate_deg_s'],
                      'tcp_rate_m_s':template['limits']['tcp_rate_m_s'], 'angular_rate_deg_s':15.},
        'travel': {'joint_rate_deg_s':12., 'tcp_rate_m_s':.080, 'angular_rate_deg_s':30.},
        'release': {'joint_rate_deg_s':8., 'tcp_rate_m_s':.040, 'angular_rate_deg_s':20.},
        'acceleration_m_s2':.30, 'position_kp':4., 'rotation_kp':4.,
        'release_height_tolerance_m':.010, 'release_lateral_speed_m_s':.015,
        'carry_orientation':True}
    values = {} if values is None else values
    if not isinstance(values,dict) or set(values)-set(defaults):
        raise ValueError('Unknown manipulation setting')
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Speed scale must be finite and positive')
    result = {}
    for name,default in defaults.items():
        value = values.get(name,default)
        if isinstance(default,dict):
            if not isinstance(value,dict) or set(value)-set(default):
                raise ValueError('Unknown manipulation motion profile setting: '+name)
            profile = {**default,**value}
            if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<=0 for v in profile.values()):
                raise ValueError('Manipulation trajectory rates must be positive')
            result[name] = {k:v*scale for k,v in profile.items()}
        elif isinstance(default,bool):
            if not isinstance(value,bool):raise ValueError(name+' must be boolean')
            result[name] = value
        else:
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:
                raise ValueError(name+' must be positive')
            result[name] = value
    result['alignment_wrist_roll_rate_deg_s'] = template['limits'].get('alignment_wrist_roll_rate_deg_s',6.)*scale
    result['gripper_rate_raw_s'] = 250.*scale
    return result


class WristGraspAligner:
    """Tape-specific image alignment; the task core only sees pose targets."""
    def __init__(self, template):
        self.template = template
        self.phase = 'WAIT'
        self.target = self.grasp = None
        self.origin = self.initial_approach = self.initial_grasp = None
        self.probe_axis = self.probe_q8 = None
        self.roll_gain = None
        self.started = self.stable_since = self.last_stamp = None

    def _settled(self, condition, now, duration=.7):
        if not condition:
            self.stable_since = None
            return False
        if self.stable_since is None:
            self.stable_since = now
        return now-self.stable_since >= duration

    def update(self, now, q, wrist, approach, grasp):
        pose = forward(q, self.template['tcp_offset_m'])
        feature = wrist.get('feature')
        stamp = wrist['timing']['sensor_timestamp_s']
        fresh = stamp != self.last_stamp
        self.last_stamp = stamp
        reliable = feature is not None and feature.get('axis_reliable') and feature.get('depth_m') is not None
        if self.started is None:
            self.started = now
            self.target = approach.copy()
        if now-self.started > 90:
            raise RuntimeError('Wrist alignment timeout')
        if self.phase == 'DONE':
            return
        if self.phase == 'WAIT':
            if not reliable:
                if now-self.started > 8:
                    raise RuntimeError('No reliable wrist tape axis/depth at pregrasp')
                return
            self.origin = (pose.copy(), float(q[5]))
            self.initial_approach, self.initial_grasp = approach.copy(), grasp.copy()
            self.probe_axis, self.probe_q8 = feature['axis_rad'], float(q[5])
            self.target = pose.copy(); self.target[:3, :3] = pose[:3, :3]@rx(np.deg2rad(3))
            self.phase = 'PROBE'
            self.stable_since = None
            return
        distance, angle = pose_error(pose, self.target)
        arrived = distance < .002 and angle < np.deg2rad(.6)
        if self.phase == 'PROBE':
            if self._settled(arrived and reliable, now):
                dq = q[5]-self.probe_q8
                gain = line_error(feature['axis_rad'], self.probe_axis)/dq if abs(dq) > np.deg2rad(1.5) else 0.
                if not -1.8 <= gain <= -.35:
                    raise RuntimeError(f'Wrist roll/image response not verified: gain {gain:.3f}')
                self.roll_gain = float(gain)
                self.phase = 'ALIGN'; self.stable_since = None
            return
        if not reliable:
            raise RuntimeError('Wrist tape axis/depth lost during alignment')
        reference = self.template['wrist_reference']
        pixel_error = np.linalg.norm(np.asarray(feature['center_px'])-reference['center_px'])
        axis_error = line_error(reference['axis_rad'], feature['axis_rad'])
        aligned = arrived and pixel_error < 7 and abs(axis_error) < np.deg2rad(3)
        if aligned:
            # Hold one target while confirming alignment; chasing image noise
            # would reset the settling interval on every new frame.
            if self._settled(True, now):
                delta = pose[:3, 3]-self.initial_approach[:3, 3]
                roll = q[5]-self.origin[1]
                self.grasp = self.initial_grasp.copy()
                self.grasp[:3, 3] += delta
                self.grasp[:3, :3] = self.grasp[:3, :3]@rx(roll)
                self.target = pose.copy(); self.phase = 'DONE'
            return
        self.stable_since = None
        if fresh and arrived:
            camera = forward(q)@np.asarray(self.template['T_gripper_wrist_camera'])
            delta, _ = corrections(feature, reference, wrist['timing']['intrinsics'], camera)
            desired = pose.copy(); desired[:3, 3] += delta
            desired[:3, :3] = pose[:3, :3]@rx(float(np.clip(.6*axis_error/self.roll_gain,
                                                              -np.deg2rad(3), np.deg2rad(3))))
            if np.linalg.norm(desired[:2, 3]-self.origin[0][:2, 3]) > .035:
                raise RuntimeError('Wrist centering exceeded 35mm correction budget')
            if abs(q[5]-self.origin[1]) > np.deg2rad(65):
                raise RuntimeError('Wrist alignment exceeded 65-degree correction budget')
            self.target = desired


class TapeDemonstrator:
    """Real estimated-state/feedback adapter around AutomaticDemonstrator."""
    def __init__(self, template, box, object_point, actual, now, motion=None):
        self.template, self.box = template, box
        self.motion = manipulation_motion(template) if motion is None else motion
        shift = np.asarray(object_point)-template['object_reference_m']
        if np.linalg.norm(shift[:2]) > .15 or abs(shift[2]) > .025:
            raise ValueError('Tape outside the demonstrated neighborhood (150mm XY / 25mm Z)')
        self.object_adapter = EstimatedObjectAdapter(template)
        config_values = dict(template.get('automatic_demonstrator', {}).get('config', {}))
        config_values.update(carry_orientation=self.motion['carry_orientation'],
            release_height_tolerance_m=self.motion['release_height_tolerance_m'])
        config = DemonstratorConfig(**config_values)
        if not config.stationary:
            raise ValueError('The VX300 entry point currently supports stationary tape only')
        self.core = AutomaticDemonstrator(prior_from_template(template),
            release_from_registration(template, box), forward(actual, template['tcp_offset_m']),
            rim_z_m=box['rim_z_base_m'], config=config)
        self.aligner = WristGraspAligner(template)
        self.open_raw = template['gripper']['open_raw']
        self.closed_raw = template['gripper']['empty_closed_raw']
        self.gripper_target = self.open_raw
        self.closed_feature = self.hold_goal = None
        self.teacher_state = None
        self.object_feedback_reason = None

    @property
    def stage(self): return self.core.stage.value

    @property
    def target_pose(self): return self.core.action.tcp_pose

    @property
    def done(self): return self.core.action.done

    @property
    def roll_gain(self): return self.aligner.roll_gain

    @property
    def needs_object(self): return self.core.requires_object

    def flags(self): return self.core.action.geometry_flags()

    def motion_profile(self, tcp_pose):
        if self.core.stage in (Stage.APPROACH_ABOVE_OBJECT, Stage.APPROACH_ABOVE_TARGET,
                               Stage.RETREAT, Stage.RETURN_HOME):
            return 'travel'
        if self.core.stage == Stage.LIFT_OBJECT and tcp_pose[2,3] > self.core.grasp[2,3]+.05:
            return 'travel'
        if self.core.stage in (Stage.APPROACH_TARGET, Stage.PLACE_OBJECT, Stage.VERIFY_PLACE):
            return 'release'
        return 'precision'

    def _placed(self, q, feature, wrist):
        if feature is None or feature.get('depth_m') is None or self.closed_feature is None:
            return False
        intr = wrist['timing']['intrinsics']; z = feature['depth_m']; u, v = feature['center_px']
        # Same provisional eye-in-hand model as the recorded grasp review.
        point = np.array([(u-intr['ppx'])*z/intr['fx'], (v-intr['ppy'])*z/intr['fy'], z, 1.])
        point = forward(q)@np.asarray(self.template['T_gripper_wrist_camera'])@point
        return bool(inside_polygon(point[:2], self.box['interior_polygon_xy_m'])
            and -.015 <= point[2] <= self.box['rim_z_base_m']+.04
            and z > self.closed_feature['depth_m']+.025)

    def update(self, now, q, raw_gripper, current, wrist, *, object_record=None, tcp_velocity_m_s=None):
        obj = None
        self.object_feedback_reason = None
        if object_record is not None:
            try:
                obj = self.object_adapter.from_record(object_record, now)
            except ValueError as exc:
                # A frame may expire between the executor's check and this
                # conversion. Optional external feedback must not stop a stage
                # already driven by the wrist and committed grasp reference.
                if self.core.requires_object:
                    raise
                self.object_feedback_reason = str(exc)
        self.teacher_state = None if obj is None else self.object_adapter.teacher_metadata(obj)
        feature = wrist.get('feature')
        contact = bool(abs(current) >= .18 and self.closed_raw+45 < raw_gripper < self.open_raw-45
                       and feature is not None and feature.get('depth_m') is not None)
        old_stage = self.core.stage
        if old_stage == Stage.ALIGN_GRASP:
            if obj is None:
                depth = None if feature is None else feature.get('depth_m')
                if (feature is None or not feature.get('axis_reliable') or depth is None
                        or not np.isfinite(depth) or depth <= 0):
                    raise RuntimeError('Reliable wrist tape axis/depth required during fixed-camera occlusion')
            self.aligner.update(now, q, wrist, *self.core.alignment_targets(obj))
        if old_stage == Stage.APPROACH_OBJECT and feature is None:
            raise RuntimeError('Tape lost during descent')
        if old_stage == Stage.GRASP_OBJECT and contact:
            self.closed_feature = dict(feature)
        held = bool(self.closed_feature is not None and feature is not None
            and feature.get('depth_m') is not None and abs(current) > .12 and raw_gripper > self.closed_raw+45
            and abs(feature['depth_m']-self.closed_feature['depth_m']) < .035
            and np.linalg.norm(np.asarray(feature['center_px'])-self.closed_feature['center_px']) < 65)
        tcp_pose = forward(q, self.template['tcp_offset_m'])
        velocity = None if tcp_velocity_m_s is None else np.asarray(tcp_velocity_m_s,float)
        drop_ready = bool(inside_polygon(tcp_pose[:2,3], self.box['interior_polygon_xy_m'], .035)
            and abs(tcp_pose[2,3]-self.core.release[2,3]) < self.motion['release_height_tolerance_m']
            and velocity is not None and velocity.shape==(3,) and np.isfinite(velocity).all()
            and np.linalg.norm(velocity[:2]) < self.motion['release_lateral_speed_m_s'])
        observation = Observation(now, tcp_pose, obj,
            gripper_open=bool(raw_gripper >= self.open_raw-12 and abs(current) < .12),
            grasp_contact=contact, holding=held, placed=self._placed(q, feature, wrist),
            confirmation_stamp=wrist['timing']['sensor_timestamp_s'],
            alignment_complete=self.aligner.phase == 'DONE', alignment_target=self.aligner.target,
            aligned_grasp=self.aligner.grasp,
            release_ready=drop_ready,
            pregrasp_ready=bool(feature is not None and feature.get('axis_reliable')
                               and feature.get('depth_m') is not None))
        action = self.core.step(observation)
        if old_stage == Stage.GRASP_OBJECT and action.stage == Stage.LIFT_OBJECT:
            self.hold_goal = max(self.closed_raw, raw_gripper-32)
        if action.gripper == 'OPEN':
            self.gripper_target = self.open_raw
        elif action.gripper == 'CLOSE':
            self.gripper_target = max(self.closed_raw, raw_gripper-32)
        else:
            if self.hold_goal is None: raise RuntimeError('No contact preload available for holding')
            self.gripper_target = self.hold_goal
        return action
