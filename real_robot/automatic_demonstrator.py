"""Simulator-independent automatic pick/place demonstrator.

The state sequence follows DynamicVLA's PlaceStateMachine; see DYNAMICVLA_NOTICE.
All transforms map into ``robot_base``, use meters, and have column vectors.
This module has no robot, camera, Isaac, Torch, or learned-policy dependencies.
Estimated/ground-truth object states are teacher inputs, not policy features.
"""
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .geometry import validate_transform, interpolate_pose


class Stage(str, Enum):
    INIT = 'INIT'
    APPROACH_ABOVE_OBJECT = 'APPROACH_ABOVE_OBJECT'
    ALIGN_GRASP = 'ALIGN_GRASP'
    APPROACH_OBJECT = 'APPROACH_OBJECT'
    GRASP_OBJECT = 'GRASP_OBJECT'
    LIFT_OBJECT = 'LIFT_OBJECT'
    VERIFY_LIFT = 'VERIFY_LIFT'
    APPROACH_ABOVE_TARGET = 'APPROACH_ABOVE_TARGET'
    APPROACH_TARGET = 'APPROACH_TARGET'
    PLACE_OBJECT = 'PLACE_OBJECT'
    VERIFY_PLACE = 'VERIFY_PLACE'
    RETREAT = 'RETREAT'
    RETURN_HOME = 'RETURN_HOME'
    COMPLETE = 'COMPLETE'


@dataclass(frozen=True)
class ObjectState:
    pose: np.ndarray
    velocity_m_s: np.ndarray
    stamp_s: float
    frame: str = 'robot_base'
    semantics: str = 'object_center'

    def __post_init__(self):
        object.__setattr__(self, 'pose', validate_transform(self.pose).copy())
        velocity = np.asarray(self.velocity_m_s, float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all() or not np.isfinite(self.stamp_s):
            raise ValueError('Object state requires finite velocity XYZ and timestamp')
        if self.frame != 'robot_base':
            raise ValueError('Convert object state to robot_base before the demonstrator')
        object.__setattr__(self, 'velocity_m_s', velocity.copy())


@dataclass(frozen=True)
class GraspPrior:
    T_object_approach: np.ndarray
    T_object_grasp: np.ndarray
    semantics: str = 'object_center'

    def __post_init__(self):
        for name in ('T_object_approach', 'T_object_grasp'):
            object.__setattr__(self, name, validate_transform(getattr(self, name)).copy())


@dataclass(frozen=True)
class DemonstratorConfig:
    stationary: bool = False
    prediction_horizon_s: float = .23
    max_prediction_m: float = .08
    object_max_age_s: float = .20
    stationary_tolerance_m: float = .015
    position_tolerance_m: float = .003
    rotation_tolerance_rad: float = np.deg2rad(2)
    pregrasp_position_tolerance_m: float = .005
    pregrasp_rotation_tolerance_rad: float = np.deg2rad(3)
    lift_m: float = .14
    transport_z_m: float = .25
    above_target_m: float = .06
    retreat_m: float = .06
    verification_frames: int = 5
    stage_timeout_s: float = 100.
    attempt_timeout_s: float = 360.
    return_home: bool = True
    carry_orientation: bool = True
    waypoint_position_tolerance_m: float = .012
    release_settle_s: float = .20
    release_height_tolerance_m: float = .010
    transport_speed_m_s: float = .12
    contact_speed_m_s: float = .03
    path_lookahead_m: float = .025
    grasp_relative_speed_m_s: float = .04

    def __post_init__(self):
        positive = [self.max_prediction_m, self.object_max_age_s, self.stationary_tolerance_m,
                    self.position_tolerance_m, self.rotation_tolerance_rad,
                    self.pregrasp_position_tolerance_m, self.pregrasp_rotation_tolerance_rad, self.lift_m,
                    self.transport_z_m, self.above_target_m, self.retreat_m,
                    self.stage_timeout_s, self.attempt_timeout_s,
                    self.waypoint_position_tolerance_m, self.release_settle_s,
                    self.release_height_tolerance_m, self.transport_speed_m_s,
                    self.contact_speed_m_s, self.path_lookahead_m, self.grasp_relative_speed_m_s]
        if not np.isfinite(positive).all() or min(positive) <= 0:
            raise ValueError('Demonstrator distances, tolerances and timeouts must be positive')
        if not np.isfinite(self.prediction_horizon_s) or self.prediction_horizon_s < 0:
            raise ValueError('Prediction horizon must be nonnegative')
        if self.verification_frames < 1:
            raise ValueError('At least one independent verification frame is required')


@dataclass(frozen=True)
class Observation:
    now_s: float
    tcp_pose: np.ndarray
    object: ObjectState | None
    gripper_open: bool
    grasp_contact: bool = False
    holding: bool = False
    placed: bool = False
    confirmation_stamp: float | None = None
    alignment_complete: bool = False
    alignment_target: np.ndarray | None = None
    aligned_grasp: np.ndarray | None = None
    release_ready: bool = True
    pregrasp_ready: bool = True
    tcp_velocity_m_s: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass(frozen=True)
class Action:
    stage: Stage
    tcp_pose: np.ndarray
    gripper: str  # OPEN, CLOSE, HOLD; adapter owns physical gripper units.
    contact: bool = False
    carrying: bool = False
    over_box: bool = False
    done: bool = False
    metadata: dict = field(default_factory=dict)
    target_velocity_m_s: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def geometry_flags(self):
        return dict(contact=self.contact, carrying=self.carrying, over_box=self.over_box)


def pose_reached(actual, target, config, *, pregrasp=False, tolerances=None):
    translation = np.linalg.norm(actual[:3, 3]-target[:3, 3])
    cosine = np.clip((np.trace(target[:3, :3].T@actual[:3, :3])-1)/2, -1., 1.)
    position_tolerance = config.pregrasp_position_tolerance_m if pregrasp else config.position_tolerance_m
    rotation_tolerance = config.pregrasp_rotation_tolerance_rad if pregrasp else config.rotation_tolerance_rad
    if tolerances is not None:
        position_tolerance, rotation_tolerance = tolerances
    return translation < position_tolerance and np.arccos(cosine) < rotation_tolerance


class AutomaticDemonstrator:
    """One attempt, shared between simulator and estimated-state adapters.

    Wrist alignment overlaps approach/descent; pursuit continues during closure.
    Lift, transport and verification share a continuous curved trajectory. This
    core never automatically opens, retries, or resets after a failure.
    """
    PRECOMMIT = frozenset((Stage.INIT, Stage.APPROACH_ABOVE_OBJECT,
                          Stage.ALIGN_GRASP, Stage.APPROACH_OBJECT))

    def __init__(self, prior, release_pose, home_pose, *, rim_z_m, config=None):
        self.prior = prior
        self.config = config or DemonstratorConfig()
        self.release = validate_transform(release_pose).copy()
        self.home = validate_transform(home_pose).copy()
        self.rim_z_m = float(rim_z_m)
        if not np.isfinite(self.rim_z_m) or self.release[2, 3] < self.rim_z_m+.065-1e-9:
            raise ValueError('Release TCP must be at least 65mm above the measured rim')
        self.above_target = self.release.copy()
        self.above_target[2, 3] += self.config.above_target_m
        self.stage = Stage.INIT
        self.started = self.stage_started = self.last_now = None
        self.stable_since = None
        self.last_confirmation = None
        self.confirmations = 0
        self.anchor = self.aligned_at = self.aligned_grasp = None
        self.alignment_reference = None
        self.alignment_reference_stamp_s = None
        self.grasp = self.lift = self.retreat = None
        self.carry_rotation = self.carry_heading = None
        self.carry_path = self.carry_distances = None
        self.path_progress_m = 0.
        self.descent = 0.
        self.last_dt = 0.
        self.holding_confirmed = False
        self.path_velocity = np.zeros(3)
        self.action = Action(self.stage, self.home.copy(), 'OPEN')

    def grasp_targets(self, obj, now_s=None):
        if obj.semantics != self.prior.semantics:
            raise ValueError('Object-state origin does not match the grasp-prior origin')
        predicted = obj.pose.copy()
        now = self.last_now if now_s is None else now_s
        age = 0. if now is None else max(0., now-obj.stamp_s)
        # Shrink interception lookahead during descent; at contact track the
        # present object rather than closing on a point ahead of it.
        horizon = self.config.prediction_horizon_s
        if self.stage in (Stage.APPROACH_OBJECT, Stage.GRASP_OBJECT):
            horizon *= 1-self.descent
        lead = np.zeros(3) if self.config.stationary else obj.velocity_m_s*(age+horizon)
        predicted[:3, 3] += lead
        return predicted@self.prior.T_object_approach, predicted@self.prior.T_object_grasp

    def _carry_pose(self, pose):
        result = pose.copy()
        if self.config.carry_orientation and self.carry_rotation is not None:
            # Keep the grasp's orientation relative to the arm's radial heading.
            # The overhead drop has no taught world-yaw/roll requirement.
            angle = np.arctan2(result[1,3],result[0,3])-self.carry_heading
            c, s = np.cos(angle), np.sin(angle)
            result[:3,:3] = np.array([[c,-s,0],[s,c,0],[0,0,1]])@self.carry_rotation
        return result

    def alignment_targets(self, obj, now_s=None):
        # The stationary reference was established from a fresh object estimate
        # at pregrasp. It is a fixed pose reference, never a refreshed observation.
        if self.config.stationary and self.alignment_reference is not None:
            return tuple(pose.copy() for pose in self.alignment_reference)
        if obj is None:
            raise RuntimeError('No object estimate or established alignment reference')
        return self.grasp_targets(obj,now_s)

    @property
    def requires_object(self):
        # After stationary pregrasp, wrist observations refine the established
        # reference through alignment and descent despite external occlusion.
        if self.stage == Stage.ALIGN_GRASP:
            return not self.config.stationary or self.alignment_reference is None
        return self.stage in (Stage.INIT, Stage.APPROACH_ABOVE_OBJECT) or (
            self.stage in (Stage.APPROACH_OBJECT, Stage.GRASP_OBJECT) and not self.config.stationary)

    def _start_carry(self, tcp):
        self.grasp = tcp.copy()
        self.carry_rotation = tcp[:3,:3].copy()
        self.carry_heading = float(np.arctan2(tcp[1,3],tcp[0,3]))
        self.release = self._carry_pose(self.release)
        height = max(tcp[2,3]+self.config.lift_m, self.config.transport_z_m,
                     self.release[2,3]+self.config.above_target_m)
        p0, p3 = tcp[:3,3].copy(), self.release[:3,3].copy()
        p1, p2 = p0.copy(), p3.copy()
        # Bezier control points are above the actual arc. Set their height so
        # the midpoint reaches the requested transport height.
        p1[2] = p2[2] = (height-.125*(p0[2]+p3[2]))/.75
        u = np.linspace(0.,1.,201)[:,None]
        self.carry_path = (1-u)**3*p0+3*(1-u)**2*u*p1+3*(1-u)*u**2*p2+u**3*p3
        self.carry_distances = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(self.carry_path,axis=0),axis=1))]
        self.path_progress_m = 0.

    def _carry_target(self, tcp):
        # Measured path progress, rather than a clock, drives the lookahead.
        # The first tangent is vertical; lateral travel grows continuously.
        distances = self.carry_distances
        eligible = np.flatnonzero(distances >= self.path_progress_m-1e-9)
        i = eligible[np.argmin(np.linalg.norm(self.carry_path[eligible]-tcp[:3,3],axis=1))]
        self.path_progress_m = max(self.path_progress_m,float(distances[i]))
        ahead = min(distances[-1], self.path_progress_m+self.config.path_lookahead_m)
        point = np.array([np.interp(ahead,distances,self.carry_path[:,axis]) for axis in range(3)])
        j = min(len(distances)-1,max(1,int(np.searchsorted(distances,ahead))))
        tangent = self.carry_path[j]-self.carry_path[j-1]
        speed = (self.config.contact_speed_m_s if tcp[2,3] < self.grasp[2,3]+.05
                 else self.config.transport_speed_m_s)
        speed = min(speed,4*max(0.,distances[-1]-ahead))
        self.path_velocity = tangent/max(np.linalg.norm(tangent),1e-12)*speed
        result = self.release.copy()
        result[:3,3] = point
        return self._carry_pose(result)

    def _settled(self, condition, now, duration=.4):
        if not condition:
            self.stable_since = None
            return False
        if self.stable_since is None:
            self.stable_since = now
        return now-self.stable_since >= duration

    def _advance(self, stage, now):
        self.stage, self.stage_started = stage, now
        self.stable_since = None
        self.confirmations = 0
        # Do not count the same image again after a state transition.

    def _confirmed(self, condition, obs):
        if obs.confirmation_stamp is not None and obs.confirmation_stamp != self.last_confirmation:
            self.last_confirmation = obs.confirmation_stamp
            self.confirmations = self.confirmations+1 if condition else 0
        return self.confirmations >= self.config.verification_frames

    def _target(self, obs):
        stage = self.stage
        if stage == Stage.INIT:
            return self.home.copy()
        if stage == Stage.APPROACH_ABOVE_OBJECT:
            return (self.alignment_targets(obs.object)[0] if obs.alignment_target is None
                    else validate_transform(obs.alignment_target).copy())
        if stage == Stage.ALIGN_GRASP:
            return (self.alignment_targets(obs.object)[0] if obs.alignment_target is None
                    else validate_transform(obs.alignment_target).copy())
        if stage == Stage.APPROACH_OBJECT:
            approach, grasp = self.alignment_targets(obs.object)
            if obs.alignment_target is not None:
                approach = validate_transform(obs.alignment_target).copy()
            if obs.aligned_grasp is not None:
                grasp = validate_transform(obs.aligned_grasp).copy()
            return interpolate_pose(approach,grasp,self.descent)
        if stage == Stage.GRASP_OBJECT:
            return (validate_transform(obs.aligned_grasp).copy() if obs.aligned_grasp is not None
                    else self.alignment_targets(obs.object)[1])
        if stage in (Stage.LIFT_OBJECT, Stage.VERIFY_LIFT, Stage.APPROACH_ABOVE_TARGET, Stage.APPROACH_TARGET):
            return self._carry_target(obs.tcp_pose)
        if stage in (Stage.PLACE_OBJECT, Stage.VERIFY_PLACE):
            return self.release.copy()
        if stage == Stage.RETREAT:
            return self.retreat.copy()
        return self.home.copy()

    def step(self, obs, *, progress=True):
        now = float(obs.now_s)
        tcp = validate_transform(obs.tcp_pose)
        if not np.isfinite(now) or (self.last_now is not None and now <= self.last_now):
            raise ValueError('Demonstrator time must advance monotonically')
        self.last_dt = 0. if self.last_now is None else now-self.last_now
        self.last_now = now
        if self.started is None:
            self.started = self.stage_started = now
        if self.requires_object:
            if obs.object is None or not 0 <= now-obs.object.stamp_s <= self.config.object_max_age_s:
                raise RuntimeError('Fresh object estimate required before grasp commitment')
            if self.anchor is None:
                self.anchor = obs.object.pose[:3, 3].copy()
        if self.stage == Stage.APPROACH_OBJECT and progress:
            # Progress the descending reference while pursuing XY and aligning
            # the wrist. Large XY tracking error pauses vertical progress only.
            approach, grasp = self.alignment_targets(obs.object)
            lateral = np.linalg.norm(tcp[:2,3]-(obs.aligned_grasp if obs.aligned_grasp is not None else grasp)[:2,3])
            if obs.pregrasp_ready and lateral < .02:
                span = max(.01,abs(approach[2,3]-grasp[2,3]))
                self.descent = min(1.,self.descent+self.last_dt*self.config.contact_speed_m_s/span)

        target = self._target(obs)
        stage = self.stage
        if progress:
            if stage == Stage.INIT:
                self._advance(Stage.APPROACH_ABOVE_OBJECT, now)
            elif stage in (Stage.APPROACH_ABOVE_OBJECT, Stage.ALIGN_GRASP) and (
                    np.linalg.norm(tcp[:3,3]-target[:3,3]) < .015 and obs.pregrasp_ready and obs.gripper_open):
                if self.config.stationary:
                    self.alignment_reference = tuple(pose.copy() for pose in self.grasp_targets(obs.object))
                    self.alignment_reference_stamp_s = float(obs.object.stamp_s)
                self._advance(Stage.APPROACH_OBJECT, now)
            elif stage == Stage.APPROACH_OBJECT and self._settled(
                    self.descent >= 1 and obs.alignment_complete and
                    pose_reached(tcp,target,self.config,tolerances=(.008,np.deg2rad(5))) and
                    np.linalg.norm(np.asarray(obs.tcp_velocity_m_s)-(np.zeros(3) if self.config.stationary
                        else obs.object.velocity_m_s)) < self.config.grasp_relative_speed_m_s, now, .06):
                self.grasp = target.copy()
                self._advance(Stage.GRASP_OBJECT, now)
            elif stage == Stage.GRASP_OBJECT:
                if self._settled(obs.grasp_contact, now, .08):
                    self._start_carry(tcp)
                    self._advance(Stage.LIFT_OBJECT, now)
            elif stage in (Stage.LIFT_OBJECT, Stage.VERIFY_LIFT, Stage.APPROACH_ABOVE_TARGET, Stage.APPROACH_TARGET):
                self.holding_confirmed = self.holding_confirmed or self._confirmed(obs.holding, obs)
                if not obs.holding:
                    # Retain the grasp while feedback recovers; never open on
                    # an unverified carry or turn verification into a waypoint.
                    self.path_velocity[:] = 0
                if stage in (Stage.LIFT_OBJECT, Stage.VERIFY_LIFT) and tcp[2,3] > self.grasp[2,3]+.05:
                    self._advance(Stage.APPROACH_ABOVE_TARGET, now)
                if stage == Stage.APPROACH_ABOVE_TARGET and self.carry_distances[-1]-self.path_progress_m < .06:
                    self._advance(Stage.APPROACH_TARGET, now)
                if stage == Stage.APPROACH_TARGET and self._settled(
                        obs.release_ready and self.holding_confirmed and obs.holding and
                        np.linalg.norm(tcp[:2,3]-self.release[:2,3]) < self.config.waypoint_position_tolerance_m and
                        abs(tcp[2,3]-self.release[2,3]) < self.config.release_height_tolerance_m,
                        now, self.config.release_settle_s):
                    self._advance(Stage.PLACE_OBJECT, now)
            elif stage == Stage.PLACE_OBJECT and self._settled(obs.gripper_open, now):
                self._advance(Stage.VERIFY_PLACE, now)
            elif stage == Stage.VERIFY_PLACE:
                if self._confirmed(obs.placed, obs) and now-self.stage_started > .5:
                    self.retreat = self.release.copy()
                    self.retreat[2, 3] += self.config.retreat_m
                    self._advance(Stage.RETREAT, now)
            elif stage == Stage.RETREAT and self._settled(
                    np.linalg.norm(tcp[:3,3]-target[:3,3]) < self.config.waypoint_position_tolerance_m, now, .1):
                self._advance(Stage.RETURN_HOME if self.config.return_home else Stage.COMPLETE, now)
            elif stage == Stage.RETURN_HOME and self._settled(
                    pose_reached(tcp, target, self.config, tolerances=(.008,np.deg2rad(5))), now, .2):
                self._advance(Stage.COMPLETE, now)

        stage = self.stage
        gripper = ('CLOSE' if stage == Stage.GRASP_OBJECT else 'HOLD' if stage in
            (Stage.LIFT_OBJECT, Stage.VERIFY_LIFT, Stage.APPROACH_ABOVE_TARGET, Stage.APPROACH_TARGET) else 'OPEN')
        result = self._target(obs)
        carrying_motion = stage in (Stage.LIFT_OBJECT,Stage.VERIFY_LIFT,Stage.APPROACH_ABOVE_TARGET,Stage.APPROACH_TARGET)
        if carrying_motion and not obs.holding:
            result = tcp.copy()
            self.path_velocity[:] = 0
        self.action = Action(stage, result, gripper,
            contact=stage in (Stage.APPROACH_OBJECT, Stage.GRASP_OBJECT, Stage.LIFT_OBJECT),
            carrying=gripper == 'HOLD',
            # Containment is a release condition; approach must be allowed to
            # converge, and retreat no longer carries the released object.
            over_box=stage in (Stage.PLACE_OBJECT, Stage.VERIFY_PLACE),
            done=stage == Stage.COMPLETE,
            metadata={'stationary': self.config.stationary, 'teacher_states_only': True,
                      'alignment_reference_stamp_s': self.alignment_reference_stamp_s,
                      'wrist_only_alignment': stage in self.PRECOMMIT and obs.object is None,
                      'descent_fraction': self.descent, 'path_progress_m': self.path_progress_m,
                      'holding_confirmed': self.holding_confirmed},
            target_velocity_m_s=(self.path_velocity.copy() if carrying_motion else
                obs.object.velocity_m_s.copy() if not self.config.stationary
                and stage in (Stage.APPROACH_ABOVE_OBJECT,Stage.ALIGN_GRASP,Stage.APPROACH_OBJECT,Stage.GRASP_OBJECT)
                and obs.object is not None else np.zeros(3)))
        return self.action
