"""Causal, device-free execution correction from paper sec/3_method.tex.

Inputs are measured velocity in right_base_estimate, already the action frame.
Prediction during occlusion is deliberately not counted as measured motion.
"""
from collections import deque
import numpy as np


class VelocityInnovation:
    settings = dict(gain=1., magnitude_limit_m=.05, step_limit_m=.005,
                    integration='causal_trapezoidal_prefix_then_last_velocity_to_control_time',
                    frame='right_base_estimate', gate='latched_first_gripper_command_le_zero',
                    predicted_velocity_used=False)

    def __init__(self):
        self.reset()

    def reset(self):
        self.samples=deque(maxlen=256)
        self.offset=np.zeros(3)
        self.closed=False
        self.diagnostics={}

    def interrupt(self):
        # Keep episode gate and applied offset; never integrate across a gap.
        self.samples.clear()

    def observe(self, stamp, velocity):
        velocity=np.asarray(velocity,float)
        if velocity.shape!=(3,) or not np.isfinite(velocity).all() or not np.isfinite(stamp):
            raise ValueError('Expected finite robot-frame velocity and timestamp')
        if self.samples:
            previous,v,prefix=self.samples[-1]
            if stamp==previous:return
            if stamp<previous or stamp-previous>.2:self.interrupt()
        prefix=np.zeros(3) if not self.samples else prefix+.5*(v+velocity)*(stamp-previous)
        self.samples.append((float(stamp),velocity.copy(),prefix.copy()))

    @staticmethod
    def saturate(value, radius):
        return value*min(1.,radius/max(float(np.linalg.norm(value)),1e-30))

    def apply(self, action, source_stamp, now):
        action=np.asarray(action,float)
        if action.shape!=(7,) or not np.isfinite(action).all() or not np.isfinite([source_stamp,now]).all():
            raise ValueError('Invalid action or innovation timestamp')
        self.closed |= bool(action[6]<=0)
        error=np.zeros(3);reason='closing_latched' if self.closed else 'missing_measured_anchor'
        if not self.closed and self.samples:
            anchors=[row for row in self.samples if row[0]==source_stamp]
            stamp,velocity,prefix=self.samples[-1]
            if anchors and source_stamp<=stamp<=now and now-stamp<=.2:
                t0,v0,p0=anchors[0]
                error=prefix-p0+velocity*(now-stamp)-v0*(now-t0)
                reason='measured_velocity_innovation'
        desired=self.saturate(error,.05)
        self.offset+=self.saturate(desired-self.offset,.005)
        result=action.copy();result[:3]+=self.offset
        self.diagnostics=dict(source_monotonic_s=float(source_stamp),control_monotonic_s=float(now),
            raw_innovation_m=error.tolist(),applied_offset_m=self.offset.tolist(),
            closing_latched=self.closed,reason=reason)
        return result
