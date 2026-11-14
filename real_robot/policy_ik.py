"""Measured-state bounded weighted DLS for timed VLA targets; no actuator access.

The policy requests an absolute pose for this 25 Hz step. Solve
q_des = q_measured + J^T (J J^T + lambda^2 I)^-1 pose_error.
The retained command is used only for actuator slew/clearance bounds, never as
an integrator of measured-pose error. Speed is induced by targets and timing;
limits constrain it without creating a separate fixed-speed trajectory. The
default bounded solve includes individual joint bounds in the optimization;
the historical uniform scaling implementation remains for explicit replay.
"""
import cv2
import numpy as np
from scipy.optimize import lsq_linear
from .geometry import validate_transform
from .vx300s_kinematics import forward,jacobian,LOWER,UPPER,JOINT_NAMES,MOTOR_IDS


def rotation_error(target,current):
    return cv2.Rodrigues(target@current.T)[0].ravel()


class PolicyDifferentialIK:
    def __init__(self,actual,retained,tcp_offset,hardware_joint_rates_deg_s,*,
                 max_tcp_speed_mps=.5,max_angular_speed_deg_s=90.,
                 max_following_error_deg=6.,damping=.01,method='bounded_dls',orientation_weight_m=.1):
        self.goal=np.asarray(retained,float).copy();self.offset=np.asarray(tcp_offset,float)
        actual=np.asarray(actual,float);forward(actual,self.offset);forward(self.goal,self.offset)
        violations=[]
        for source,values in [('measured',actual),('retained goal',self.goal)]:
            for i in np.flatnonzero((values<LOWER)|(values>UPPER)):
                violations.append(f'{source} {JOINT_NAMES[i]} (motor {MOTOR_IDS[i]}) '
                    f'{np.rad2deg(values[i]):.4f} deg outside '
                    f'[{np.rad2deg(LOWER[i]):.4f}, {np.rad2deg(UPPER[i]):.4f}] deg')
        if violations:
            raise ValueError('Initial joint state/goal outside limits: '+'; '.join(violations))
        rates=np.asarray(hardware_joint_rates_deg_s,float)
        if rates.shape!=(6,) or not np.isfinite(rates).all() or np.any(rates<=0):
            raise ValueError('Six positive hardware joint velocity limits required')
        if any(v is None for v in [max_following_error_deg,damping,orientation_weight_m]):raise ValueError('Invalid IK safety bounds')
        values=[max_following_error_deg,damping,orientation_weight_m]+[v for v in [max_tcp_speed_mps,max_angular_speed_deg_s] if v is not None]
        if not np.isfinite(values).all() or min(values)<=0:raise ValueError('Invalid IK safety bounds')
        self.rates=np.deg2rad(rates);self.tcp_limit=max_tcp_speed_mps
        self.angular_limit=None if max_angular_speed_deg_s is None else np.deg2rad(max_angular_speed_deg_s)
        self.following_limit=np.deg2rad(max_following_error_deg);self.damping=damping
        if method not in ('bounded_dls','uniform_dls'):raise ValueError('Unknown policy IK method')
        self.method=method;self.orientation_weight=orientation_weight_m
        self.diagnostics={}

    def hold(self):
        return self.goal.copy()

    def step(self,target_pose,actual,dt):
        target=validate_transform(target_pose);actual=np.asarray(actual,float)
        if not np.isfinite(dt) or not 0<dt<=.15:raise ValueError('Invalid IK control interval')
        measured=forward(actual,self.offset)
        if np.any(actual<LOWER) or np.any(actual>UPPER):raise RuntimeError('Measured joints outside limits')
        following=self.goal-actual
        if np.max(np.abs(following))>self.following_limit+1e-9:
            raise RuntimeError('Joint following error exceeds configured stall bound')
        error=np.r_[target[:3,3]-measured[:3,3],rotation_error(target[:3,:3],measured[:3,:3])]
        j=jacobian(actual,self.offset);singular=np.linalg.svd(j,compute_uv=False)
        low=np.maximum(LOWER,actual-self.following_limit)
        high=np.minimum(UPPER,actual+self.following_limit)
        retained_pose=forward(self.goal,self.offset)
        solver_info={}
        if self.method=='uniform_dls':
            damping=self.damping+max(0.,.025-singular[-1])
            delta=j.T@np.linalg.solve(j@j.T+damping*damping*np.eye(6),error)
            command_step=actual+delta-self.goal
            scale=min(1.,float(np.min(self.rates*dt/np.maximum(np.abs(command_step),1e-12))))
            for k,value in enumerate(command_step):
                if value>0:scale=min(scale,max(0.,float((high[k]-self.goal[k])/value)))
                elif value<0:scale=min(scale,max(0.,float((low[k]-self.goal[k])/value)))
        else:
            # Limit each task component relative to the retained command, not
            # measured q. Clipping relative to a lagging measurement could pull
            # a healthy retained reference backwards every control cycle.
            position_step=target[:3,3]-retained_pose[:3,3]
            if self.tcp_limit is not None:position_step*=min(1.,self.tcp_limit*dt/max(np.linalg.norm(position_step),1e-12))
            angular_step=rotation_error(target[:3,:3],retained_pose[:3,:3])
            if self.angular_limit is not None:angular_step*=min(1.,self.angular_limit*dt/max(np.linalg.norm(angular_step),1e-12))
            limited_rotation=cv2.Rodrigues(angular_step)[0]@retained_pose[:3,:3]
            limited_error=np.r_[retained_pose[:3,3]+position_step-measured[:3,3],
                rotation_error(limited_rotation,measured[:3,:3])]
            # Express rotation residuals in metres using a 0.1m characteristic
            # length (also used by the offline pose IK). Orientation remains
            # an objective, but no longer implicitly weighs 1rad like 1metre.
            weights=np.r_[np.ones(3),np.full(3,self.orientation_weight)]
            weighted=j*weights[:,None];weighted_singular=np.linalg.svd(weighted,compute_uv=False)
            damping=self.damping+max(0.,.025-weighted_singular[-1])
            matrix=np.vstack([weighted,damping*np.eye(6)])
            rhs=np.r_[weights*limited_error,np.zeros(6)]
            lower=np.maximum(low,self.goal-self.rates*dt)-actual
            upper=np.minimum(high,self.goal+self.rates*dt)-actual
            if np.any(lower>upper+1e-12):raise RuntimeError('No feasible bounded joint update')
            free=upper-lower>1e-12;delta=(lower+upper)/2
            if free.any():
                solved=lsq_linear(matrix[:,free],rhs-matrix[:,~free]@delta[~free],
                    bounds=(lower[free],upper[free]),method='bvls',tol=1e-10,max_iter=50)
                if not solved.success or not np.isfinite(solved.x).all():raise RuntimeError('Bounded DLS did not converge')
                delta[free]=solved.x
                solver_info.update(solver_iterations=int(solved.nit),solver_optimality=float(solved.optimality))
            command_step=actual+delta-self.goal;scale=1.
            solver_info.update(orientation_weight_m=self.orientation_weight,
                weighted_minimum_singular_value=float(weighted_singular[-1]),
                position_reference_step_m=float(np.linalg.norm(position_step)),
                angular_reference_step_deg=float(np.rad2deg(np.linalg.norm(angular_step))),
                active_joint_bounds=np.flatnonzero(np.isclose(delta,lower,atol=1e-8,rtol=0)|np.isclose(delta,upper,atol=1e-8,rtol=0)).tolist())
        # Bound finite Cartesian steps too, rather than assuming J's local
        # approximation remains exact after a larger correction near a singularity.
        for _ in range(16):
            candidate=self.goal+scale*command_step
            commanded=forward(candidate,self.offset)
            distance=np.linalg.norm(commanded[:3,3]-retained_pose[:3,3])
            angle=np.linalg.norm(rotation_error(commanded[:3,:3],retained_pose[:3,:3]))
            factors=[1.]
            if self.tcp_limit is not None:factors.append(self.tcp_limit*dt/max(distance,1e-12))
            if self.angular_limit is not None:factors.append(self.angular_limit*dt/max(angle,1e-12))
            factor=min(factors)
            if factor>=1.-1e-9:break
            scale*=factor*.999
        else:raise RuntimeError('Could not enforce Cartesian command-step bounds')
        self.diagnostics=dict(backend='policy_bounded_weighted_dls' if self.method=='bounded_dls' else 'policy_measured_state_dls',
            method=self.method,position_error_m=float(np.linalg.norm(error[:3])),
            rotation_error_deg=float(np.rad2deg(np.linalg.norm(error[3:]))),
            requested_tcp_speed_mps=float(np.linalg.norm(error[:3])/dt),
            requested_angular_speed_deg_s=float(np.rad2deg(np.linalg.norm(error[3:]))/dt),
            command_scale=float(scale),minimum_singular_value=float(singular[-1]),damping=float(damping),
            command_scale_scope='finite_cartesian_backtracking' if self.method=='bounded_dls' else 'global_joint_and_cartesian_scaling',
            following_error_rad=following.tolist(),joint_rate_ceilings_deg_s=np.rad2deg(self.rates).tolist(),
            tcp_speed_ceiling_mps=self.tcp_limit,angular_speed_ceiling_deg_s=None if self.angular_limit is None else float(np.rad2deg(self.angular_limit)),**solver_info)
        self.check_candidate(candidate,actual,dt)
        return candidate

    def accept(self,candidate):
        candidate=np.asarray(candidate,float)
        forward(candidate,self.offset)
        if np.any(candidate<LOWER-1e-10) or np.any(candidate>UPPER+1e-10):raise ValueError('Joint goal outside limits')
        self.goal=candidate.copy()

    def check_candidate(self,candidate,actual,dt):
        """Recheck a trajectory-shaped command; never clip its derivatives."""
        candidate=np.asarray(candidate,float);actual=np.asarray(actual,float)
        if candidate.shape!=(6,) or not np.isfinite(candidate).all():raise RuntimeError('Invalid joint trajectory command')
        if np.any(candidate<LOWER) or np.any(candidate>UPPER):raise RuntimeError('Joint trajectory exceeds joint limits')
        if np.max(np.abs(candidate-actual))>self.following_limit+1e-9:raise RuntimeError('Joint trajectory exceeds following error bound')
        if np.any(np.abs(candidate-self.goal)>self.rates*dt+1e-9):raise RuntimeError('Joint trajectory exceeds velocity bound')
        old,new=forward(self.goal,self.offset),forward(candidate,self.offset)
        if ((self.tcp_limit is not None and np.linalg.norm(new[:3,3]-old[:3,3])>self.tcp_limit*dt+1e-9)
                or (self.angular_limit is not None and np.linalg.norm(rotation_error(new[:3,:3],old[:3,:3]))>self.angular_limit*dt+1e-9)):
            raise RuntimeError('Joint trajectory exceeds Cartesian motion bound')
