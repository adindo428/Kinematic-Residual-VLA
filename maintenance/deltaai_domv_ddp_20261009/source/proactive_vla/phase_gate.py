"""Causal attachment gate shared by offline prefix replay and live control.

Only current/past object and EE observations and the previously *applied*
gripper command are accepted. Chunk prediction extrapolates an existing ramp;
it never anticipates a future grasp from demonstration actions.
"""
from collections import deque
from dataclasses import dataclass, asdict
import math

import numpy as np


PHASES = ("approach", "acquire", "fade", "carry", "released", "lost", "reset")


def rotation_xyz(angles):
    """Extrinsic xyz Euler matrix, matching DOM's scipy xyz convention."""
    x, y, z = angles
    cx, cy, cz = np.cos([x, y, z]); sx, sy, sz = np.sin([x, y, z])
    return np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                     [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx],
                     [-sy, cy*sx, cy*cx]])


@dataclass(frozen=True)
class PhaseGateConfig:
    close_duration_s: float = 0.2
    history_s: float = 0.2
    lift_m: float = 0.02
    relative_position_m: float = 0.01
    relative_speed_m_s: float = 0.05
    fade_s: float = 0.2
    loss_duration_s: float = 0.12
    loss_position_m: float = 0.03
    loss_speed_m_s: float = 0.1

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Phase thresholds must be finite and positive")


def gate_from_state(state, horizon, dt):
    """Expand [start, target, elapsed, duration] without future observations."""
    start, target, elapsed, duration = state
    u = np.clip((elapsed + np.arange(horizon) * dt) / duration, 0, 1)
    blend = 0.5 - 0.5 * np.cos(np.pi * u)
    return (start + (target - start) * blend).astype(np.float32)[:, None]


class CausalPhaseGate:
    def __init__(self, config=None):
        self.config = config or PhaseGateConfig()
        self.reset()

    def reset(self):
        self.history = deque()
        self.phase = "approach"
        self.close_since = self.close_z = self.confirmed_relative = None
        self.loss_since = self.last_time = None
        self.ramp = (1.0, 1.0, 0.0)

    def _level(self, now):
        start, target, since = self.ramp
        return float(gate_from_state((start, target, max(0, now-since), self.config.fade_s), 1, 0)[0, 0])

    def _ramp_to(self, target, now):
        self.ramp = (self._level(now), float(target), now)

    def observe(self, timestamp, ee_position, object_position, object_velocity,
                previous_gripper_command, resetting=False, ee_rotation_xyz=None):
        t = float(timestamp)
        ee, obj, vel = (np.asarray(x, dtype=np.float64).reshape(3)
                        for x in (ee_position, object_position, object_velocity))
        if not all(np.isfinite(x).all() for x in (ee, obj, vel)) or not math.isfinite(t):
            raise ValueError("Nonfinite phase observation")
        if self.last_time is not None and t < self.last_time:
            raise ValueError("Phase observations must be chronological; reset between episodes")
        if self.last_time == t:
            return self.state(t)
        self.last_time = t
        if resetting:
            self.reset()
            self.last_time = t
            self.phase = "reset"
            self.ramp = (0., 0., t)
            return self.state(t)
        if self.phase == "reset":
            self.phase = "approach"
            self._ramp_to(1, t)
        rotation = np.eye(3) if ee_rotation_xyz is None else rotation_xyz(np.asarray(ee_rotation_xyz).reshape(3))
        if not np.isfinite(rotation).all():
            raise ValueError("Nonfinite EE rotation")
        relative = rotation.T@(obj-ee)
        self.history.append((t, ee.copy(), relative.copy(), vel.copy(), rotation))
        while len(self.history) > 1 and self.history[1][0] <= t-self.config.history_s:
            self.history.popleft()
        closed = previous_gripper_command is not None and float(previous_gripper_command) <= 0
        if not closed:
            self.close_since = self.close_z = None
            self.loss_since = None
            if self.confirmed_relative is not None:
                self.phase = "released"
                self._ramp_to(0, t) if self.ramp[1] != 0 else None
            elif self.phase != "released":
                self.phase = "approach"
        else:
            if self.close_since is None:
                self.close_since, self.close_z = t, float(obj[2])
                if self.phase == "released":
                    self.confirmed_relative = None
                    self._ramp_to(1, t)
                self.phase = "acquire" if self.confirmed_relative is None else self.phase
            span = t-self.history[0][0]
            ready = span + 1e-8 >= self.config.history_s
            relative_speed = float("inf")
            variation = float("inf")
            if ready:
                history = list(self.history)
                speeds = []
                for a, b in zip(history, history[1:]):
                    dt = b[0]-a[0]
                    global_relative = b[4]@b[2]
                    local_speed = b[4].T@(b[3]-(b[1]-a[1])/dt)+(b[4].T-a[4].T)@global_relative/dt
                    speeds.append(np.linalg.norm(local_speed))
                relative_speed = float(max(speeds))
                positions = np.stack([row[2] for row in history])
                variation = float(np.linalg.norm(positions[:, None]-positions[None, :], axis=-1).max())
            if self.confirmed_relative is None:
                if (ready and t-self.close_since+1e-8 >= self.config.close_duration_s
                        and obj[2]-self.close_z+1e-8 >= self.config.lift_m
                        and variation <= self.config.relative_position_m
                        and relative_speed <= self.config.relative_speed_m_s):
                    self.confirmed_relative = relative.copy()
                    self.phase = "fade"
                    self._ramp_to(0, t)
            else:
                lost = (np.linalg.norm(relative-self.confirmed_relative) > self.config.loss_position_m
                        or (ready and relative_speed > self.config.loss_speed_m_s))
                self.loss_since = (t if self.loss_since is None else self.loss_since) if lost else None
                if self.loss_since is not None and t-self.loss_since+1e-8 >= self.config.loss_duration_s:
                    self.confirmed_relative = None
                    self.close_since, self.close_z = t, float(obj[2])
                    self.phase = "lost"
                    self._ramp_to(1, t)
                elif self.phase == "fade" and self._level(t) < 1e-6:
                    self.phase = "carry"
        return self.state(t)

    def state(self, timestamp):
        start, target, since = self.ramp
        return np.array([start, target, max(0, float(timestamp)-since), self.config.fade_s], dtype=np.float32)


def replay_prefix_states(states, environment_states, actions, timestamps, config=None):
    """Return compact per-anchor states/phase IDs; action[t] is never read at t."""
    gate = CausalPhaseGate(config)
    compact = np.empty((len(states), 4), np.float32)
    phases = np.empty(len(states), np.int8)
    for i in range(len(states)):
        env = environment_states[i]
        compact[i] = gate.observe(timestamps[i], states[i][:3], env[:3], env[6:9],
                                 None if i == 0 else actions[i-1][-1], ee_rotation_xyz=states[i][3:6])
        phases[i] = PHASES.index(gate.phase)
    return compact, phases
