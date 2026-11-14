"""Local online joint trajectories between IK and the existing motion guards.

Ruckig replans from the last *accepted reference* position, velocity and
acceleration. Encoder noise must not reset these derivatives on every update.
No intermediate waypoints or remote/cloud calculation are used. A safety hold
resets derivatives immediately; safety stops take precedence over smoothness.
"""
import numpy as np


def learned_governor_enabled(profile,request,execute=False):
    enabled=request=='on' or (request=='auto' and profile in ('hardware','smooth'))
    if execute and not enabled:
        raise ValueError('Unfiltered learned-policy execution is disabled after the recorded vibration incident. Use a joint acceleration/jerk governor; expert-action replay remains a separate diagnostic.')
    return enabled


class JointTrajectoryGovernor:
    def __init__(self,position,velocity_deg_s,acceleration_deg_s2=120.,jerk_deg_s3=600.):
        from ruckig import Ruckig,InputParameter,Trajectory
        self.generator=Ruckig(6);self.input=InputParameter(6);self.trajectory=Trajectory(6)
        for name,value in [('max_velocity',velocity_deg_s),('max_acceleration',acceleration_deg_s2),('max_jerk',jerk_deg_s3)]:
            values=np.broadcast_to(np.asarray(value,float),(6,))
            if not np.isfinite(values).all() or np.any(values<=0):raise ValueError('Positive finite trajectory limits required')
            setattr(self.input,name,np.deg2rad(values).tolist())
        self.input.target_velocity=[0.]*6;self.input.target_acceleration=[0.]*6
        self.reset(position)

    def reset(self,position):
        position=np.asarray(position,float)
        if position.shape!=(6,) or not np.isfinite(position).all():raise ValueError('Expected six finite joint positions')
        self.position=position.copy();self.velocity=np.zeros(6);self.acceleration=np.zeros(6)
        self.pending=None;self.diagnostics={}

    def propose(self,target,dt):
        from ruckig import Result
        target=np.asarray(target,float)
        if target.shape!=(6,) or not np.isfinite(target).all():raise ValueError('Expected six finite joint targets')
        if not np.isfinite(dt) or not 0<dt<=.15:raise ValueError('Invalid trajectory interval')
        self.input.current_position=self.position.tolist()
        self.input.current_velocity=self.velocity.tolist()
        self.input.current_acceleration=self.acceleration.tolist()
        self.input.target_position=target.tolist()
        result=self.generator.calculate(self.input,self.trajectory)
        if result not in (Result.Working,Result.Finished):raise RuntimeError(f'Joint trajectory calculation failed: {result}')
        position,velocity,acceleration=map(np.asarray,self.trajectory.at_time(dt))
        if not np.isfinite(np.r_[position,velocity,acceleration]).all():raise RuntimeError('Nonfinite joint trajectory')
        self.pending=(position,velocity,acceleration)
        self.diagnostics=dict(backend='ruckig_joint_trajectory',duration_s=float(self.trajectory.duration),
            reference_velocity_deg_s=np.rad2deg(velocity).tolist(),
            reference_acceleration_deg_s2=np.rad2deg(acceleration).tolist(),
            max_velocity_deg_s=np.rad2deg(self.input.max_velocity).tolist(),
            max_acceleration_deg_s2=np.rad2deg(self.input.max_acceleration).tolist(),
            max_jerk_deg_s3=np.rad2deg(self.input.max_jerk).tolist(),
            target_lag_deg=np.rad2deg(target-position).tolist())
        return position.copy()

    def accept(self,position):
        if self.pending is None or not np.array_equal(position,self.pending[0]):
            raise ValueError('Only the checked, proposed trajectory command may be accepted')
        self.position,self.velocity,self.acceleration=(v.copy() for v in self.pending)
        self.pending=None
