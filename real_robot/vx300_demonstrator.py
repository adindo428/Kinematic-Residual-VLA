"""VX300 task-space servo and wrist/gripper adapter for the shared demonstrator.

No devices are opened here. The executor owns the existing serial connection.
"""
import cv2
import numpy as np

from .automatic_demonstrator import AutomaticDemonstrator, DemonstratorConfig, Observation, Stage
from .demonstrator_adapters import EstimatedObjectAdapter, prior_from_template, release_from_registration
from .geometry import validate_transform
from .pick_geometry import inside_polygon, pose_error, rx
from .vx300s_kinematics import forward, jacobian, LOWER, UPPER
from .wrist_alignment import corrections, line_error


class VX300DifferentialIK:
    """Damped least-squares TCP tracking with retained gravity compensation.

    Smooth trajectory rates shape commands; measured speed is diagnostic only.
    There is no iterative whole-pose optimizer or whole-target joint interpolation.
    """
    def __init__(self, actual, retained, tcp_offset, *, joint_rate_deg_s=3., tcp_rate_m_s=.020,
                 angular_rate_deg_s=15.):
        self.nominal = np.asarray(actual, float).copy()
        self.goal = np.asarray(retained, float).copy()
        self.offset = np.asarray(tcp_offset, float)
        forward(self.nominal, self.offset)
        if self.goal.shape != (6,) or not np.isfinite(self.goal).all():
            raise ValueError('Expected six retained joint goals')
        self.compensation = self.goal-self.nominal
        if np.max(np.abs(self.compensation)) > np.deg2rad(3):
            raise ValueError('Existing holding compensation exceeds three degrees')
        if (not np.isfinite([joint_rate_deg_s, tcp_rate_m_s, angular_rate_deg_s]).all()
                or min(joint_rate_deg_s, tcp_rate_m_s, angular_rate_deg_s) <= 0):
            raise ValueError('Command trajectory rates must be finite and positive')
        self.joint_rate = np.deg2rad(joint_rate_deg_s)
        self.tcp_rate = tcp_rate_m_s
        self.angular_rate = np.deg2rad(angular_rate_deg_s)
        self.diagnostics = {}

    def step(self, target_pose, actual, dt, *, wrist_roll_rate_deg_s=None):
        target = validate_transform(target_pose)
        actual = np.asarray(actual, float)
        if actual.shape != (6,) or not np.isfinite(actual).all() or not 0 < dt <= .15:
            raise ValueError('Invalid joint feedback or control interval')
        joint_rates = np.full(6, self.joint_rate)
        if wrist_roll_rate_deg_s is not None:
            if not np.isfinite(wrist_roll_rate_deg_s) or wrist_roll_rate_deg_s <= 0:
                raise ValueError('Wrist roll trajectory rate must be finite and positive')
            joint_rates[-1] = np.deg2rad(wrist_roll_rate_deg_s)
        if np.max(np.abs(self.goal-actual)) > np.deg2rad(6):
            raise RuntimeError('Following error exceeds six degrees')
        pose = forward(self.nominal, self.offset)
        translation = target[:3, 3]-pose[:3, 3]
        rotation = cv2.Rodrigues(target[:3, :3]@pose[:3, :3].T)[0].ravel()
        translation *= min(1., self.tcp_rate*dt/max(np.linalg.norm(translation), 1e-12))
        rotation *= min(1., self.angular_rate*dt/max(np.linalg.norm(rotation), 1e-12))
        j = jacobian(self.nominal, self.offset)
        # Express angular error in equivalent meters for DLS conditioning.
        weighted = j.copy(); weighted[3:] *= .1
        error = np.r_[translation, .1*rotation]
        singular_values = np.linalg.svd(weighted, compute_uv=False)
        damping = .01+max(0., .025-singular_values[-1])
        delta = weighted.T@np.linalg.solve(weighted@weighted.T+damping*damping*np.eye(6), error)
        delta *= min(1., float(np.min(joint_rates*dt/np.maximum(np.abs(delta), 1e-12))))
        candidate = np.clip(self.nominal+delta, LOWER+1e-6, UPPER-1e-6)
        distance = np.linalg.norm(forward(candidate, self.offset)[:3, 3]-pose[:3, 3])
        if distance > self.tcp_rate*dt:
            candidate = self.nominal+(candidate-self.nominal)*(self.tcp_rate*dt/distance)
        self.nominal = candidate
        self.compensation = np.clip(self.compensation+.4*(self.nominal-actual)*dt,
                                    -np.deg2rad(3), np.deg2rad(3))
        requested = self.nominal+self.compensation
        self.goal += np.clip(requested-self.goal, -joint_rates*dt, joint_rates*dt)
        if np.any(self.goal < LOWER) or np.any(self.goal > UPPER):
            raise RuntimeError('Differential IK goal outside joint limits')
        self.diagnostics = {'damping': float(damping), 'minimum_singular_value': float(singular_values[-1]),
            'position_error_m': float(np.linalg.norm(target[:3, 3]-pose[:3, 3])),
            'actual_position_error_m': float(np.linalg.norm(target[:3, 3]-forward(actual,self.offset)[:3,3])),
            'nominal_q_rad': self.nominal.tolist(), 'holding_compensation_rad': self.compensation.tolist(),
            'following_error_rad': (self.goal-actual).tolist(),
            'joint_trajectory_rates_deg_s': np.rad2deg(joint_rates).tolist()}
        return self.goal.copy()


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
    def __init__(self, template, box, object_point, actual, now):
        self.template, self.box = template, box
        shift = np.asarray(object_point)-template['object_reference_m']
        if np.linalg.norm(shift[:2]) > .15 or abs(shift[2]) > .025:
            raise ValueError('Tape outside the demonstrated neighborhood (150mm XY / 25mm Z)')
        self.object_adapter = EstimatedObjectAdapter(template)
        config = DemonstratorConfig(**template.get('automatic_demonstrator', {}).get('config', {}))
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

    def update(self, now, q, raw_gripper, current, wrist, *, object_record=None):
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
        observation = Observation(now, tcp_pose, obj,
            gripper_open=bool(raw_gripper >= self.open_raw-12 and abs(current) < .12),
            grasp_contact=contact, holding=held, placed=self._placed(q, feature, wrist),
            confirmation_stamp=wrist['timing']['sensor_timestamp_s'],
            alignment_complete=self.aligner.phase == 'DONE', alignment_target=self.aligner.target,
            aligned_grasp=self.aligner.grasp,
            release_ready=inside_polygon(tcp_pose[:2,3], self.box['interior_polygon_xy_m'], .035),
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
