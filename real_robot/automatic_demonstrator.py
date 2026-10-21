"""Simulator-independent automatic pick/place demonstrator.

The state sequence follows DynamicVLA's PlaceStateMachine; see DYNAMICVLA_NOTICE.
All transforms map into ``robot_base``, use meters, and have column vectors.
This module has no robot, camera, Isaac, Torch, or learned-policy dependencies.
Estimated/ground-truth object states are teacher inputs, not policy features.
"""
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .geometry import validate_transform


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
    stationary: bool = True
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

    def __post_init__(self):
        positive = [self.max_prediction_m, self.object_max_age_s, self.stationary_tolerance_m,
                    self.position_tolerance_m, self.rotation_tolerance_rad,
                    self.pregrasp_position_tolerance_m, self.pregrasp_rotation_tolerance_rad, self.lift_m,
                    self.transport_z_m, self.above_target_m, self.retreat_m,
                    self.stage_timeout_s, self.attempt_timeout_s]
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

    def geometry_flags(self):
        return dict(contact=self.contact, carrying=self.carrying, over_box=self.over_box)


def pose_reached(actual, target, config, *, pregrasp=False):
    translation = np.linalg.norm(actual[:3, 3]-target[:3, 3])
    cosine = np.clip((np.trace(target[:3, :3].T@actual[:3, :3])-1)/2, -1., 1.)
    position_tolerance = config.pregrasp_position_tolerance_m if pregrasp else config.position_tolerance_m
    rotation_tolerance = config.pregrasp_rotation_tolerance_rad if pregrasp else config.rotation_tolerance_rad
    return translation < position_tolerance and np.arccos(cosine) < rotation_tolerance


class AutomaticDemonstrator:
    """One attempt, shared between simulator and estimated-state adapters.

    Position targets track the object before closure, and grasp orientation is
    latched after alignment. Failed grasp/placement verification raises; this
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
        self.action = Action(self.stage, self.home.copy(), 'OPEN')

    def grasp_targets(self, obj):
        if obj.semantics != self.prior.semantics:
            raise ValueError('Object-state origin does not match the grasp-prior origin')
        predicted = obj.pose.copy()
        lead = np.zeros(3) if self.config.stationary else obj.velocity_m_s*self.config.prediction_horizon_s
        length = np.linalg.norm(lead)
        if length > self.config.max_prediction_m:
            lead *= self.config.max_prediction_m/length
        predicted[:3, 3] += lead
        return predicted@self.prior.T_object_approach, predicted@self.prior.T_object_grasp

    def alignment_targets(self, obj):
        # The stationary reference was established from a fresh object estimate
        # at pregrasp. It is a fixed pose reference, never a refreshed observation.
        if self.config.stationary and self.alignment_reference is not None:
            return tuple(pose.copy() for pose in self.alignment_reference)
        if obj is None:
            raise RuntimeError('No object estimate or established alignment reference')
        return self.grasp_targets(obj)

    @property
    def requires_object(self):
        # After stationary pregrasp, wrist observations refine the established
        # reference through alignment and descent despite external occlusion.
        if self.stage == Stage.ALIGN_GRASP:
            return not self.config.stationary or self.alignment_reference is None
        return self.stage in (Stage.INIT, Stage.APPROACH_ABOVE_OBJECT) or (
            self.stage == Stage.APPROACH_OBJECT and not self.config.stationary)

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
            return self.grasp_targets(obs.object)[0]
        if stage == Stage.ALIGN_GRASP:
            return (self.alignment_targets(obs.object)[0] if obs.alignment_target is None
                    else validate_transform(obs.alignment_target).copy())
        if stage == Stage.APPROACH_OBJECT:
            target = self.aligned_grasp.copy()
            if not self.config.stationary:
                target[:3, 3] += self.grasp_targets(obs.object)[1][:3, 3]-self.aligned_at
            return target
        if stage == Stage.GRASP_OBJECT:
            return self.grasp.copy()
        if stage in (Stage.LIFT_OBJECT, Stage.VERIFY_LIFT):
            return self.lift.copy()
        if stage == Stage.APPROACH_ABOVE_TARGET:
            return self.above_target.copy()
        if stage in (Stage.APPROACH_TARGET, Stage.PLACE_OBJECT, Stage.VERIFY_PLACE):
            return self.release.copy()
        if stage == Stage.RETREAT:
            return self.retreat.copy()
        return self.home.copy()

    def step(self, obs, *, progress=True):
        now = float(obs.now_s)
        tcp = validate_transform(obs.tcp_pose)
        if not np.isfinite(now) or (self.last_now is not None and now <= self.last_now):
            raise ValueError('Demonstrator time must advance monotonically')
        self.last_now = now
        if self.started is None:
            self.started = self.stage_started = now
        if now-self.started > self.config.attempt_timeout_s or now-self.stage_started > self.config.stage_timeout_s:
            raise RuntimeError('Automatic demonstrator stage/attempt timeout')
        if self.requires_object:
            if obs.object is None or not 0 <= now-obs.object.stamp_s <= self.config.object_max_age_s:
                raise RuntimeError('Fresh object estimate required before grasp commitment')
            if self.anchor is None:
                self.anchor = obs.object.pose[:3, 3].copy()
            if self.config.stationary and np.linalg.norm(obs.object.pose[:3, 3]-self.anchor) > self.config.stationary_tolerance_m:
                raise RuntimeError('Tape moved during the stationary attempt')
        elif (self.stage in (Stage.ALIGN_GRASP, Stage.APPROACH_OBJECT)
              and obs.object is not None and self.config.stationary
              and 0 <= now-obs.object.stamp_s <= self.config.object_max_age_s):
            if np.linalg.norm(obs.object.pose[:3, 3]-self.anchor) > self.config.stationary_tolerance_m:
                raise RuntimeError('Tape moved during stationary alignment/descent')
        if self.stage in (Stage.APPROACH_ABOVE_TARGET, Stage.APPROACH_TARGET) and not obs.holding:
            raise RuntimeError('Held object lost during transport')

        target = self._target(obs)
        arrived = pose_reached(tcp, target, self.config, pregrasp=self.stage == Stage.APPROACH_ABOVE_OBJECT)
        stage = self.stage
        if progress:
            if stage == Stage.INIT and self._settled(obs.gripper_open, now, .28):
                self._advance(Stage.APPROACH_ABOVE_OBJECT, now)
            elif stage == Stage.APPROACH_ABOVE_OBJECT and self._settled(arrived and obs.pregrasp_ready, now):
                if self.config.stationary:
                    self.alignment_reference = tuple(pose.copy() for pose in self.grasp_targets(obs.object))
                    self.alignment_reference_stamp_s = float(obs.object.stamp_s)
                self._advance(Stage.ALIGN_GRASP, now)
            elif stage == Stage.ALIGN_GRASP and self._settled(obs.alignment_complete and arrived, now):
                if obs.aligned_grasp is None:
                    raise ValueError('Alignment must supply the corrected grasp pose')
                self.aligned_grasp = validate_transform(obs.aligned_grasp).copy()
                self.aligned_at = self.alignment_targets(obs.object)[1][:3, 3].copy()
                self._advance(Stage.APPROACH_OBJECT, now)
            elif stage == Stage.APPROACH_OBJECT and self._settled(arrived, now, .5):
                self.grasp = target.copy()
                self._advance(Stage.GRASP_OBJECT, now)
            elif stage == Stage.GRASP_OBJECT:
                if self._settled(obs.grasp_contact, now):
                    self.lift = self.grasp.copy()
                    self.lift[2, 3] = max(self.lift[2, 3]+self.config.lift_m,
                        self.above_target[2, 3], self.config.transport_z_m)
                    self._advance(Stage.LIFT_OBJECT, now)
                elif now-self.stage_started > 8:
                    raise RuntimeError('Grasp failed: no supported gripper contact')
            elif stage == Stage.LIFT_OBJECT and self._settled(arrived, now):
                self._advance(Stage.VERIFY_LIFT, now)
            elif stage == Stage.VERIFY_LIFT:
                if self._confirmed(obs.holding, obs) and now-self.stage_started > .4:
                    self._advance(Stage.APPROACH_ABOVE_TARGET, now)
                elif now-self.stage_started > 5:
                    raise RuntimeError('Lift verification failed')
            elif stage == Stage.APPROACH_ABOVE_TARGET and self._settled(arrived, now):
                self._advance(Stage.APPROACH_TARGET, now)
            elif stage == Stage.APPROACH_TARGET and self._settled(arrived and obs.release_ready, now):
                self._advance(Stage.PLACE_OBJECT, now)
            elif stage == Stage.PLACE_OBJECT and self._settled(obs.gripper_open, now):
                self._advance(Stage.VERIFY_PLACE, now)
            elif stage == Stage.VERIFY_PLACE:
                if self._confirmed(obs.placed, obs) and now-self.stage_started > .5:
                    self.retreat = self.release.copy()
                    self.retreat[2, 3] += self.config.retreat_m
                    self._advance(Stage.RETREAT, now)
                elif now-self.stage_started > 10:
                    raise RuntimeError('Object released; placement is visually unverified')
            elif stage == Stage.RETREAT and self._settled(arrived, now):
                self._advance(Stage.RETURN_HOME if self.config.return_home else Stage.COMPLETE, now)
            elif stage == Stage.RETURN_HOME and self._settled(arrived, now):
                self._advance(Stage.COMPLETE, now)

        stage = self.stage
        gripper = ('CLOSE' if stage == Stage.GRASP_OBJECT else 'HOLD' if stage in
            (Stage.LIFT_OBJECT, Stage.VERIFY_LIFT, Stage.APPROACH_ABOVE_TARGET, Stage.APPROACH_TARGET) else 'OPEN')
        self.action = Action(stage, self._target(obs), gripper,
            contact=stage in (Stage.APPROACH_OBJECT, Stage.GRASP_OBJECT, Stage.LIFT_OBJECT),
            carrying=gripper == 'HOLD',
            # Containment is a release condition; approach must be allowed to
            # converge, and retreat no longer carries the released object.
            over_box=stage in (Stage.PLACE_OBJECT, Stage.VERIFY_PLACE),
            done=stage == Stage.COMPLETE,
            metadata={'stationary': self.config.stationary, 'teacher_states_only': True,
                      'alignment_reference_stamp_s': self.alignment_reference_stamp_s,
                      'wrist_only_alignment': stage == Stage.ALIGN_GRASP and obs.object is None})
        return self.action
