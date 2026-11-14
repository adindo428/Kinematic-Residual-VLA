"""Causal, training-free Cartesian correction; no object pose inputs."""
import json
from pathlib import Path

import numpy as np


def ball_clip(vector, radius):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    return vector * min(1., radius / max(norm, 1e-15))


class VelocityInnovation:
    def __init__(self, mode, dt, gain=1., radius=.05, max_step=.005):
        assert mode in ('none', 'innovation', 'naive')
        assert dt > 0 and gain >= 0 and radius > 0 and max_step > 0
        self.mode, self.dt = mode, float(dt)
        self.gain, self.radius, self.max_step = gain, radius, max_step
        self.velocities, self.integrals, self.rows = [], [], []
        self.origin = None
        self.closed = False
        self.offset = np.zeros(3)

    def observe(self, index, velocity):
        assert index == len(self.velocities), (index, len(self.velocities))
        velocity = np.asarray(velocity, dtype=np.float64).reshape(3).copy()
        assert np.isfinite(velocity).all()
        integral = (self.integrals[-1] + .5*self.dt*(self.velocities[-1]+velocity)
                    if self.velocities else np.zeros(3))
        self.velocities.append(velocity)
        self.integrals.append(integral)

    def set_origin(self, index):
        assert isinstance(index, int) and 0 <= index < len(self.velocities), index
        self.origin = index

    def correct(self, action):
        """Absolute robot-frame pose [xyz, quaternion, gripper]; returns copy."""
        base = np.asarray(action).copy()
        assert base.shape == (1, 8) and np.isfinite(base).all()
        index = len(self.velocities)-1
        assert index >= 0
        if self.origin is not None and base[0, -1] <= 0:
            self.closed = True  # Latched for this one-object episode; no pose-based gate.
        raw = np.zeros(3)
        age = 0.
        if self.origin is not None:
            age = (index-self.origin)*self.dt
            if self.mode == 'innovation':
                raw = (self.integrals[index]-self.integrals[self.origin]
                       -self.velocities[self.origin]*age)
            elif self.mode == 'naive':
                raw = self.velocities[index]*age
        target = (ball_clip(self.gain*raw, self.radius)
                  if not self.closed and self.mode != 'none' else np.zeros(3))
        self.offset += ball_clip(target-self.offset, self.max_step)
        base[0, :3] += self.offset
        assert np.linalg.norm(self.offset) <= self.radius+1e-9
        self.rows.append(dict(index=index, origin=self.origin, age_s=age,
            velocity=self.velocities[index].tolist(), raw_error_m=raw.tolist(),
            offset_m=self.offset.tolist(), gate_open=not self.closed,
            base_action=np.asarray(action).tolist(), applied_action=base.tolist()))
        return base

    def finish(self, folder):
        folder = Path(folder)
        (folder/'velocity_correction.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        offsets = np.array([r['offset_m'] for r in self.rows])
        norms = np.linalg.norm(offsets, axis=-1)
        summary = dict(mode=self.mode, steps=len(self.rows), max_offset_m=float(norms.max()),
            corrected_steps=int((norms > 1e-7).sum()), closing_seen=self.closed,
            gain=self.gain, radius_m=self.radius, max_offset_change_per_step_m=self.max_step,
            velocity_source='oracle simulator velocity rotated into robot frame',
            object_position_used=False, destination_position_used=False,
            numerical_integration='causal trapezoidal prefix integral; chunk observation index anchor',
            gate='latched off on first nonpositive gripper command; rate-limited fade')
        (folder/'velocity_correction_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
