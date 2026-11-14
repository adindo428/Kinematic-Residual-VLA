"""Fixed session start pose, measured settling gate and guarded return plans.

No hardware access. Return paths are local Ruckig joint trajectories, checked
before motion and again at each executed step. A blocked path is never bypassed.
"""
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from .policy_trajectory import JointTrajectoryGovernor
from .vx300s_kinematics import forward,jacobian,LOWER,UPPER
from .dom_dataset import pose6


def check_joint_path(geometry,retained,candidate,actual):
    offset=actual-retained
    segments=max(1,int(np.ceil(np.max(np.abs(candidate-retained))/np.deg2rad(2))))
    for fraction in np.linspace(0,1,segments+1):
        intermediate=retained+fraction*(candidate-retained)
        geometry.check(intermediate,.02)
        projected=intermediate+offset
        if np.any(projected<LOWER) or np.any(projected>UPPER):raise RuntimeError('Projected measured joint motion exceeds limits')
        geometry.check(projected,.02)


class StartPoseGate:
    def __init__(self,joints,gripper,tcp,*,settle_s=.4,command_joints=None,command_gripper=None,
            position_tolerance_m=.005,joint_tolerance_deg=1.,rotation_tolerance_deg=2.):
        self.joints=np.asarray(joints,float).copy();self.gripper=int(gripper);self.tcp=np.asarray(tcp,float)
        if self.joints.shape!=(6,) or not np.isfinite(self.joints).all() or np.any(self.joints<LOWER) or np.any(self.joints>UPPER):
            raise ValueError('Invalid fixed start joints')
        # Under load the measured pose need not equal the position register's
        # holding goal. Replay the command that produced the measured reference;
        # commanding measured angles directly would apply that bias a second time.
        self.command_joints=np.asarray(self.joints if command_joints is None else command_joints,float).copy()
        self.command_gripper=self.gripper if command_gripper is None else int(command_gripper)
        if (self.command_joints.shape!=(6,) or not np.isfinite(self.command_joints).all()
                or np.any(self.command_joints<LOWER) or np.any(self.command_joints>UPPER)
                or np.max(abs(self.command_joints-self.joints))>np.deg2rad(6)+1e-9):
            raise ValueError('Invalid start holding commands or excessive initial following error')
        if not np.isfinite([position_tolerance_m,joint_tolerance_deg,rotation_tolerance_deg,settle_s]).all() or min(position_tolerance_m,joint_tolerance_deg,rotation_tolerance_deg,settle_s)<=0:
            raise ValueError('Start tolerances and settling duration must be finite and positive')
        self.position_tolerance_m=float(position_tolerance_m);self.joint_tolerance_deg=float(joint_tolerance_deg)
        self.rotation_tolerance_deg=float(rotation_tolerance_deg)
        self.position_release_m=self.position_tolerance_m*1.1
        self.instantaneous_position_limit_m=self.position_tolerance_m*1.2
        self.pose=forward(self.joints,self.tcp);self.settle_s=settle_s;self.invalidate()

    def invalidate(self):
        self.stable_since=None;self.previous=None;self.ready=False;self.diagnostics={};self.samples=deque()

    def observe(self,joints,gripper,now):
        joints=np.asarray(joints,float);pose=forward(joints,self.tcp)
        speed=float('inf')
        if self.previous is not None:
            stamp,q=self.previous;dt=now-stamp
            if 0<dt<=.15:speed=float(np.max(np.abs(joints-q))/dt)
            else:self.samples.clear()
        self.previous=(now,joints.copy())
        position=float(np.linalg.norm(pose[:3,3]-self.pose[:3,3]))
        angle=float(np.linalg.norm(Rotation.from_matrix(pose[:3,:3]@self.pose[:3,:3].T).as_rotvec()))
        joint_error=float(np.max(abs(joints-self.joints)));gripper_error=abs(int(gripper)-self.gripper)
        # One encoder tick caused ~0.6mm TCP error variation in the real reset.
        # Use a full quiet window's median for the configured position check,
        # with a separate instantaneous ceiling and 2mm maximum window excursion.
        # Joint/orientation/gripper/speed limits still apply to every sample.
        within=(joint_error<=np.deg2rad(self.joint_tolerance_deg) and position<=self.instantaneous_position_limit_m and angle<=np.deg2rad(self.rotation_tolerance_deg)
                and gripper_error<=12 and speed<=np.deg2rad(5))
        self.samples.append((now,pose[:3,3].copy(),within))
        while len(self.samples)>1 and now-self.samples[1][0]>=self.settle_s:self.samples.popleft()
        stamps,positions,valid=zip(*self.samples);positions=np.asarray(positions)
        filtered_error=float(np.linalg.norm(np.median(positions,axis=0)-self.pose[:3,3]))
        window_span=float(np.linalg.norm(np.ptp(positions,axis=0)))
        span=now-stamps[0];blocking=[]
        # Once settled, do not withdraw READY for a sub-tick threshold crossing.
        # A real movement/raw-limit violation still clears readiness immediately.
        position_threshold=self.position_release_m if self.ready else self.position_tolerance_m
        if span<self.settle_s-1e-9:blocking.append(f'collecting {self.settle_s:g}s feedback')
        if filtered_error>position_threshold:blocking.append(f'median XYZ >{position_threshold*1000:g}mm')
        if window_span>.002:blocking.append('TCP still moving >2mm/window')
        if not all(valid):blocking.append('raw pose/gripper/speed settling')
        self.ready=not blocking
        self.stable_since=stamps[0] if all(valid) and filtered_error<=position_threshold and window_span<=.002 else None
        self.diagnostics=dict(ready=bool(self.ready),joint_error_deg=float(np.rad2deg(joint_error)),
            position_error_m=position,rotation_error_deg=float(np.rad2deg(angle)),gripper_error_ticks=gripper_error,
            filtered_position_error_m=filtered_error,position_window_span_m=window_span,blocking_checks=blocking,
            filtered_position_threshold_m=position_threshold,
            joint_threshold_deg=self.joint_tolerance_deg,
            rotation_threshold_deg=self.rotation_tolerance_deg,
            stable_s=0. if self.stable_since is None else now-self.stable_since)
        return self.ready

    def record(self):
        return dict(schema='krvla_eval_start_pose_v1',robot='right_follower',joints_rad=self.joints.tolist(),
            tcp_pose6=pose6(self.pose).tolist(),tcp_offset_m=self.tcp.tolist(),gripper_raw=self.gripper,
            command_joints_rad=self.command_joints.tolist(),command_gripper_raw=self.command_gripper,
            reference_kind='measured_pose_with_holding_commands',
            tolerances=dict(joints_deg=self.joint_tolerance_deg,tcp_m=self.position_tolerance_m,tcp_statistic='median XYZ over settle window',
                settled_tcp_release_m=self.position_release_m,
                instantaneous_tcp_m=self.instantaneous_position_limit_m,max_window_span_m=.002,
                rotation_deg=self.rotation_tolerance_deg,gripper_ticks=12,settle_s=self.settle_s,max_joint_speed_deg_s=5.))


class StartPoseReturn:
    def __init__(self,retained,start,hardware_rates,tcp_limit,angular_limit,geometry,workspace_min,workspace_max,now):
        self.start=start;self.began=now
        if (workspace_min is None)!=(workspace_max is None):raise ValueError('Provide both workspace bounds or neither')
        self.low=None if workspace_min is None else np.asarray(workspace_min)
        self.high=None if workspace_max is None else np.asarray(workspace_max)
        generator=JointTrajectoryGovernor(retained,np.minimum(hardware_rates,15.),60.,300.)
        generator.propose(start.command_joints,.04);self.trajectory=generator.trajectory
        self.scale=1.;duration=self.trajectory.duration
        # Sample at <=20ms before time dilation; bounded 15deg/s joints imply
        # <=0.3deg nominal geometry spacing, finer than the execution path check.
        self.samples=np.linspace(0,duration,max(2,int(np.ceil(duration/.02))+1));self.checked=0
        self.prepared=False;self.tcp_limit=tcp_limit;self.angular_limit=angular_limit
        self.planning_deadline=now+30.;self.duration=duration

    def advance_preflight(self,geometry,now,max_samples=5):
        if self.prepared:return True
        if now>self.planning_deadline:raise RuntimeError('Reset path validation timed out')
        for stamp in self.samples[self.checked:self.checked+max_samples]:
            q,v,_=map(np.asarray,self.trajectory.at_time(float(stamp)))
            self.check_pose(q,geometry)
            twist=jacobian(q,self.start.tcp)@v
            if self.tcp_limit is not None:self.scale=max(self.scale,float(np.linalg.norm(twist[:3])/self.tcp_limit))
            if self.angular_limit is not None:self.scale=max(self.scale,float(np.linalg.norm(twist[3:])/np.deg2rad(self.angular_limit)))
            self.checked+=1
        if self.checked==len(self.samples):
            self.scale*=1.02;self.duration=self.trajectory.duration*self.scale
            self.began=now;self.deadline=now+self.duration+10.;self.prepared=True
        return self.prepared

    def check_pose(self,q,geometry):
        if np.any(q<LOWER) or np.any(q>UPPER):raise RuntimeError('Reset trajectory exceeds joint limits')
        pose=forward(q,self.start.tcp)
        if self.low is not None and (np.any(pose[:3,3]<self.low) or np.any(pose[:3,3]>self.high)):raise RuntimeError('Reset trajectory leaves the configured workspace')
        geometry.check(q,.02)

    def sample(self,now):
        if not self.prepared:raise RuntimeError('Reset path must be checked before motion')
        if now>self.deadline:
            d=self.start.diagnostics
            raise RuntimeError('Reset did not reach and settle at the fixed start pose: '
                f'XYZ error {1000*d.get("position_error_m",float("nan")):.1f}mm, '
                f'joint error {d.get("joint_error_deg",float("nan")):.2f}deg, '
                f'gripper error {d.get("gripper_error_ticks","unknown")} ticks')
        q,_,_=self.trajectory.at_time(max(0.,min(self.duration,now-self.began))/self.scale)
        return np.asarray(q)

    def finished(self,now):return self.prepared and now-self.began>=self.duration and self.start.ready
