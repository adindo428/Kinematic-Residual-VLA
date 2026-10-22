"""VX300 task-space servo and wrist/gripper adapter for the shared demonstrator.

No devices are opened here. The executor owns the existing serial connection.
"""
import numpy as np

from .automatic_demonstrator import AutomaticDemonstrator, DemonstratorConfig, Observation, Stage
from .demonstrator_adapters import EstimatedObjectAdapter, prior_from_template, release_from_registration
from .geometry import interpolate_pose
from .front_placement import FrontPlacementVerifier
from .pick_geometry import inside_polygon, rx
from .vx300s_kinematics import forward
from .wrist_alignment import corrections, line_error
from .wrist_target import WristTargetEstimator, camera_from_tcp, feature_point, project


def manipulation_motion(template, values=None, scale=1.5):
    """Resolve phase motion targets; values are rates, not measured-speed stops."""
    defaults = {
        'precision': {'joint_rate_deg_s':template['limits']['joint_rate_deg_s'],
                      'tcp_rate_m_s':template['limits']['tcp_rate_m_s'], 'angular_rate_deg_s':15.},
        'travel': {'joint_rate_deg_s':12., 'tcp_rate_m_s':.080, 'angular_rate_deg_s':30.},
        'release': {'joint_rate_deg_s':8., 'tcp_rate_m_s':.040, 'angular_rate_deg_s':20.},
        'acceleration_m_s2':.30, 'position_kp':4., 'rotation_kp':4.,
        'release_height_tolerance_m':.010, 'release_lateral_speed_m_s':.015,
        'carry_orientation':True, 'approach_rotation_scale':1.0}
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
    """Continuous image feedback during approach, descent and closure.

    The installed hand-eye transform supplies the roll response; no physical
    probe or arrival/settle handshake is required before each correction.
    """
    def __init__(self, template):
        self.template = template
        self.phase = 'WAIT'
        self.target = self.grasp = None
        self.roll_gain = -float(template['T_gripper_wrist_camera'][0][2])
        self.roll = 0.
        self.translation = np.zeros(3)
        self.last_stamp = None
        self.complete = False
        self.established = False
        self.axis_error = self.pixel_error = None

    def update(self, now, q, wrist, approach, grasp, *, fraction=0., capture_camera=None,
               velocity=None, prediction_s=0., roll_rate_deg_s=9.):
        feature = wrist.get('feature') or {}
        stamp = wrist['timing']['host_arrival_monotonic_s']
        intr = wrist['timing']['intrinsics']
        reliable = feature.get('axis_reliable') and feature.get('depth_m') is not None
        # Near closure the tape may be clipped by the fingers. Keep the last
        # aligned orientation while fresh centroid/depth still localizes it.
        self.complete = bool(self.established and feature.get('depth_m') is not None)
        nominal = interpolate_pose(approach,grasp,fraction)
        planned = nominal.copy()
        planned[:3,3] += self.translation
        planned[:3,:3] = planned[:3,:3]@rx(self.roll)
        if reliable and capture_camera is not None:
            # Project the demonstrated surface/axis into the camera expected at
            # this descent depth. A fixed pregrasp pixel is wrong near contact.
            taught_approach = np.asarray(self.template['stages']['pregrasp']['T_base_tcp'])
            taught_grasp = np.asarray(self.template['stages']['before_close']['T_base_tcp'])
            reference = self.template['wrist_reference']
            taught_camera = camera_from_tcp(taught_approach,self.template)
            surface = taught_camera[:3,:3]@feature_point(reference,intr)+taught_camera[:3,3]
            axis = np.array([np.cos(reference['axis_rad'])/intr['fx'],
                             np.sin(reference['axis_rad'])/intr['fy'],0.])
            expected = project(surface,taught_camera[:3,:3]@axis,
                camera_from_tcp(interpolate_pose(taught_approach,taught_grasp,fraction),self.template),intr)
            observed = capture_camera[:3,:3]@feature_point(feature,intr)+capture_camera[:3,3]
            if velocity is not None:
                observed += np.asarray(velocity)*(max(0.,now-stamp)+prediction_s)
            direction = capture_camera[:3,:3]@np.array([np.cos(feature['axis_rad'])/intr['fx'],
                                                        np.sin(feature['axis_rad'])/intr['fy'],0.])
            camera = camera_from_tcp(planned,self.template)
            projected = project(observed,direction,camera,intr)
            if expected is not None and projected is not None:
                self.axis_error = line_error(expected['axis_rad'],projected['axis_rad'])
                self.pixel_error = float(np.linalg.norm(np.asarray(projected['center_px'])-expected['center_px']))
                self.complete = abs(self.axis_error) < np.deg2rad(4) and self.pixel_error < 10
                self.established = self.established or self.complete
                if stamp != self.last_stamp:
                    elapsed = 1/15 if self.last_stamp is None else float(np.clip(stamp-self.last_stamp,0.,.15))
                    # Trajectory shaping limits each update, not the total
                    # correction or target displacement. Joint/table geometry
                    # still applies to the resulting commands.
                    shifted = planned.copy()
                    shifted[:3,:3] = shifted[:3,:3]@rx(.001)
                    response = project(observed,direction,camera_from_tcp(shifted,self.template),intr)
                    if response is not None:
                        self.roll_gain = line_error(response['axis_rad'],projected['axis_rad'])/.001
                    step = .5*self.axis_error*self.roll_gain/(self.roll_gain**2+.01)
                    rate = np.deg2rad(roll_rate_deg_s)*elapsed
                    self.roll += float(np.clip(step,-rate,rate))
                    delta,_ = corrections(projected,expected,intr,camera,
                                          max_translation=.04*elapsed)
                    self.translation += delta
                self.phase = 'ALIGNED' if self.complete else 'TRACK'
            else:
                self.phase = 'WAIT'
        else:
            self.phase = 'WAIT'
        if reliable:
            self.last_stamp = stamp
        self.target,self.grasp = approach.copy(),grasp.copy()
        for pose in (self.target,self.grasp):
            pose[:3,3] += self.translation
            pose[:3,:3] = pose[:3,:3]@rx(self.roll)
        return self.complete


class TapeDemonstrator:
    """Real estimated-state/feedback adapter around AutomaticDemonstrator."""
    def __init__(self, template, box, object_point, actual, now, motion=None):
        self.template, self.box = template, box
        self.motion = manipulation_motion(template) if motion is None else motion
        self.object_adapter = EstimatedObjectAdapter(template)
        config_values = dict(template.get('automatic_demonstrator', {}).get('config', {}))
        config_values.update(stationary=False,carry_orientation=self.motion['carry_orientation'],
            object_max_age_s=.25,
            release_height_tolerance_m=self.motion['release_height_tolerance_m'],
            transport_speed_m_s=self.motion['travel']['tcp_rate_m_s'],
            contact_speed_m_s=self.motion['precision']['tcp_rate_m_s'])
        config = DemonstratorConfig(**config_values)
        self.core = AutomaticDemonstrator(prior_from_template(template),
            release_from_registration(template, box), forward(actual, template['tcp_offset_m']),
            rim_z_m=box['rim_z_base_m'], config=config)
        self.aligner = WristGraspAligner(template)
        self.object_estimator = WristTargetEstimator(template,self.object_adapter.reference)
        self.object_state = None
        self.observation_ready = False
        self.open_raw = template['gripper']['open_raw']
        self.closed_raw = template['gripper']['empty_closed_raw']
        self.gripper_target = self.open_raw
        self.closed_feature = self.hold_goal = None
        self.teacher_state = None
        self.object_feedback_reason = None
        self.placement_verifier = FrontPlacementVerifier(template,box)
        self.release_observed_at = None
        self.placement_verified = False
        self.placement_evidence = {'source':'front_efficienttam_rgbd','confirmed':False,
                                   'reason':'waiting for release'}

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

    @property
    def needs_wrist_camera(self):
        return self.core.stage not in (Stage.PLACE_OBJECT,Stage.VERIFY_PLACE,Stage.RETREAT,
                                       Stage.RETURN_HOME,Stage.COMPLETE)

    @property
    def needs_wrist_holding_feedback(self):
        return self.core.stage in (Stage.LIFT_OBJECT,Stage.VERIFY_LIFT,
            Stage.APPROACH_ABOVE_TARGET,Stage.APPROACH_TARGET)

    @property
    def approach_rotation_scale(self):
        return (self.motion['approach_rotation_scale'] if self.core.stage in
            (Stage.APPROACH_ABOVE_OBJECT,Stage.ALIGN_GRASP,Stage.APPROACH_OBJECT) else 1.)

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

    def update(self, now, q, raw_gripper, current, wrist, *, object_record=None,
               front_record=None, tcp_velocity_m_s=None, wrist_capture_q=None):
        obj = None
        self.object_feedback_reason = None
        if object_record is not None:
            try:
                obj = self.object_adapter.from_record(object_record, now)
            except ValueError as exc:
                # A frame may expire between the executor's check and this
                # conversion. Optional external feedback must not stop a stage
                # already driven by the wrist and committed grasp reference.
                self.object_feedback_reason = str(exc)
        wrist = wrist or {'timing':{},'feature':None}
        obj = self.object_estimator.update(obj,wrist,wrist_capture_q,now) if self.core.requires_object else obj
        self.object_state = obj
        feature = wrist.get('feature')
        old_stage = self.core.stage
        self.observation_ready = not self.core.requires_object or obj is not None
        if not self.observation_ready:
            self.core.last_now = now
            self.core.stable_since = None
            return None
        self.teacher_state = None if obj is None else {
            **self.object_adapter.teacher_metadata(obj),'source':self.object_estimator.source}
        contact = bool(abs(current) >= .18 and self.closed_raw+45 < raw_gripper < self.open_raw-45
                       and feature is not None and feature.get('depth_m') is not None)
        if old_stage in (*self.core.PRECOMMIT, Stage.GRASP_OBJECT) and wrist.get('timing',{}).get('intrinsics'):
            fraction = self.core.descent
            horizon = self.core.config.prediction_horizon_s*(1-fraction)
            approach,grasp = self.core.alignment_targets(obj,now)
            self.aligner.update(now,q,wrist,approach,grasp,fraction=fraction,
                capture_camera=self.object_estimator.capture_pose,
                velocity=obj.velocity_m_s,prediction_s=horizon,
                roll_rate_deg_s=self.motion['alignment_wrist_roll_rate_deg_s']*self.approach_rotation_scale)
        elif old_stage in (*self.core.PRECOMMIT, Stage.GRASP_OBJECT):
            self.aligner.complete = False
            self.aligner.target,self.aligner.grasp = self.core.alignment_targets(obj)
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
        gripper_open = bool(raw_gripper >= self.open_raw-12 and abs(current) < .12)
        if old_stage == Stage.PLACE_OBJECT and gripper_open and self.release_observed_at is None:
            self.release_observed_at = now
        placed,front_stamp,self.placement_evidence = self.placement_verifier.evaluate(
            front_record,now,self.release_observed_at)
        observation = Observation(now, tcp_pose, obj,
            gripper_open=gripper_open,
            grasp_contact=contact, holding=held, placed=placed and gripper_open,
            confirmation_stamp=wrist['timing'].get('sensor_timestamp_s'),
            placement_confirmation_stamp=front_stamp,
            alignment_complete=self.aligner.complete, alignment_target=self.aligner.target,
            aligned_grasp=self.aligner.grasp,
            release_ready=drop_ready,
            tcp_velocity_m_s=np.zeros(3) if velocity is None else velocity,
            pregrasp_ready=bool(feature is not None and feature.get('depth_m') is not None
                and (self.aligner.established or (self.aligner.axis_error is not None
                    and abs(self.aligner.axis_error)<np.deg2rad(12)
                    and self.aligner.pixel_error is not None and self.aligner.pixel_error<25))))
        action = self.core.step(observation)
        if old_stage == Stage.VERIFY_PLACE and action.stage == Stage.RETREAT:
            self.placement_verified = True
        self.placement_evidence.update(
            verified=self.placement_verified,
            verified_frames=(self.core.config.verification_frames if self.placement_verified
                else self.core.confirmations if old_stage==Stage.VERIFY_PLACE else 0),
            required_frames=self.core.config.verification_frames)
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
