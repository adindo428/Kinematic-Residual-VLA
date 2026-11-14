"""Short causal quintic pose-target blend before policy differential IK.

Replans from the current reference position/velocity/acceleration, preserving C2
continuity at each update. Rotations use continuous rotation vectors about a
fixed reference, avoiding component-wise Euler interpolation. This is a target
filter with a small lag, not a collision planner or a change to policy weights.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import validate_transform


class PoseTargetSmoother:
    def __init__(self,horizon_s=.08):
        if not np.isfinite(horizon_s) or horizon_s<.04:raise ValueError('Smoothing horizon must be >=40ms')
        self.horizon=horizon_s;self.anchor=None;self.x=self.v=self.a=None;self.diagnostics={}

    def reset(self,pose):
        pose=validate_transform(pose);self.anchor=pose[:3,:3].copy()
        self.x=np.r_[pose[:3,3],np.zeros(3)];self.v=np.zeros(6);self.a=np.zeros(6)
        self.diagnostics={}

    def _rotation_vector(self,rotation):
        vector=Rotation.from_matrix(rotation@self.anchor.T).as_rotvec();angle=np.linalg.norm(vector)
        if angle<1e-9:
            previous=self.x[3:];norm=np.linalg.norm(previous)
            if norm<np.pi:return vector
            axis=previous/norm
            options=[axis*(2*np.pi*k) for k in range(-2,3)]
        else:
            axis=vector/angle;options=[vector+2*np.pi*k*axis for k in range(-2,3)]
        return min(options,key=lambda x:np.linalg.norm(x-self.x[3:]))

    def step(self,target_pose,dt):
        target=validate_transform(target_pose)
        if self.anchor is None:raise RuntimeError('Reset the smoother to the current EE pose first')
        if not np.isfinite(dt) or not 0<dt<=.15:raise ValueError('Invalid smoothing interval')
        goal=np.r_[target[:3,3],self._rotation_vector(target[:3,:3])]
        t=max(self.horizon,dt)
        coefficients=np.zeros((6,6));coefficients[0]=self.x;coefficients[1]=self.v;coefficients[2]=self.a/2
        matrix=np.array([[t**3,t**4,t**5],[3*t*t,4*t**3,5*t**4],[6*t,12*t*t,20*t**3]])
        rhs=np.vstack([goal-self.x-self.v*t-self.a*t*t/2,-self.v-self.a*t,-self.a])
        coefficients[3:]=np.linalg.solve(matrix,rhs)
        self.x=np.array([1,dt,dt**2,dt**3,dt**4,dt**5])@coefficients
        self.v=np.array([0,1,2*dt,3*dt**2,4*dt**3,5*dt**4])@coefficients
        self.a=np.array([0,0,2,6*dt,12*dt**2,20*dt**3])@coefficients
        output=np.eye(4);output[:3,3]=self.x[:3]
        output[:3,:3]=Rotation.from_rotvec(self.x[3:]).as_matrix()@self.anchor
        lag=np.linalg.norm(target[:3,3]-output[:3,3])
        angular_lag=np.linalg.norm(Rotation.from_matrix(target[:3,:3]@output[:3,:3].T).as_rotvec())
        self.diagnostics=dict(horizon_s=self.horizon,position_lag_m=float(lag),rotation_lag_deg=float(np.rad2deg(angular_lag)),
            reference_velocity_mps=self.v[:3].tolist(),reference_acceleration_mps2=self.a[:3].tolist())
        return output
