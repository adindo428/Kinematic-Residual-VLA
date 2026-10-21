"""Hardware-free Cartesian PID + analytic IK for continuous object following."""
import numpy as np
from .vx300s_kinematics import forward, inverse_tool_pitch, LOWER, UPPER

DEFAULT_SERVO=dict(workspace_min_m=[.20,-.30,.09],workspace_max_m=[.58,.30,.35],
    hover_height_m=.05,surface_top_offset_m=.042,kp=2.,ki=0.,kd=0.,
    integral_limit_m_s=.02,max_tcp_speed_mps=.03,max_tcp_acceleration_mps2=.08,
    max_joint_speed_deg_s=6.,max_command_lead_m=.05,max_observation_age_s=.20,target_smoothing_s=.12,
    tool_pitch_deg=90.,tool_pitch_rate_deg_s=4.,max_following_error_deg=4.,
    # Legacy configuration key; follow/pick-place do not install current guards.
    experimental_current_guard=False,
    seconds=600,control_hz=30,collision_model='scratch/real_robot/right-handeye/hover-plan/offline-collision-model.xml')

def validate_servo(values=None):
    if values is None: values={}
    if not isinstance(values,dict) or set(values)-set(DEFAULT_SERVO): raise ValueError('Unknown continuous servo setting')
    c={**DEFAULT_SERVO,**values}
    lo,hi=np.asarray(c['workspace_min_m'],float),np.asarray(c['workspace_max_m'],float)
    if lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite([*lo,*hi]).all() or np.any(lo>=hi): raise ValueError('Invalid TCP workspace')
    if np.any(lo<[.15,-.35,.07]) or np.any(hi>[.60,.35,.45]): raise ValueError('Workspace exceeds the tabletop tracking envelope')
    bounds=dict(hover_height_m=(.03,.10),surface_top_offset_m=(0.,.08),kp=(.1,4.),ki=(0.,1.),kd=(0.,.3),
        integral_limit_m_s=(0.,.03),max_tcp_speed_mps=(.001,.5),max_tcp_acceleration_mps2=(.005,5.),
        max_joint_speed_deg_s=(.1,360.),max_command_lead_m=(.02,.20),max_observation_age_s=(.05,.25),target_smoothing_s=(0.,.5),
        tool_pitch_deg=(20.,90.),tool_pitch_rate_deg_s=(.1,30.),max_following_error_deg=(4.,12.),
        seconds=(1.,1800.),control_hz=(10.,30.))
    for name,(minimum,maximum) in bounds.items():
        value=c[name]
        if value is None and name in ('max_joint_speed_deg_s','max_tcp_acceleration_mps2','max_command_lead_m'):continue
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or not minimum<=value<=maximum:
            raise ValueError(f'{name} must be within [{minimum}, {maximum}]')
    if not isinstance(c['experimental_current_guard'],bool):raise ValueError('experimental_current_guard must be boolean')
    return c

class CartesianIKServo:
    def __init__(self,retained_goal,settings=None,joint_velocity_limits_deg_s=None):
        self.c=validate_servo(settings); self.q=np.asarray(retained_goal,float).copy()
        if self.q.shape!=(6,) or not np.isfinite(self.q).all() or np.any(self.q<LOWER) or np.any(self.q>UPPER):
            raise ValueError('Invalid retained motor goals')
        self.velocity=np.zeros(3);self.integral=np.zeros(3);self.previous_actual=None;self.measured_velocity=np.zeros(3)
        self.filtered_target=None
        hardware=None if joint_velocity_limits_deg_s is None else np.asarray(joint_velocity_limits_deg_s,float)
        if hardware is not None and (hardware.shape!=(6,) or not np.isfinite(hardware).all() or np.any(hardware<=0) or np.any(hardware>360)):
            raise ValueError('Invalid configured motor velocity limits')
        if self.c['max_joint_speed_deg_s'] is None:
            if hardware is None:raise ValueError('Read motor velocity limits before disabling the pilot joint-rate limit')
            limits=hardware
        else:limits=np.full(6,self.c['max_joint_speed_deg_s']) if hardware is None else np.minimum(hardware,self.c['max_joint_speed_deg_s'])
        self.joint_rate_rad_s=np.deg2rad(limits)

    def hold(self):
        self.velocity[:]=0;self.integral[:]=0;self.previous_actual=None;self.measured_velocity[:]=0
        self.filtered_target=None
        return self.q.copy()

    def step(self,target,actual,dt):
        c=self.c;target=np.asarray(target,float);actual=np.asarray(actual,float)
        if not 0<dt<=.15 or target.shape!=(3,) or not np.isfinite(target).all(): raise ValueError('Invalid servo target/timing')
        if np.any(target<c['workspace_min_m']) or np.any(target>c['workspace_max_m']):
            raise ValueError('Target outside configured TCP workspace')
        inverse_tool_pitch(target,self.q,c['tool_pitch_deg'])
        if self.filtered_target is None:self.filtered_target=target.copy()
        else:self.filtered_target+=(1-np.exp(-dt/max(c['target_smoothing_s'],1e-8)))*(target-self.filtered_target)
        target=self.filtered_target
        measured=forward(actual,[.16,0,0])[:3,3];reference=forward(self.q,[.16,0,0])[:3,3]
        error=target-measured
        if self.previous_actual is not None:
            raw=(measured-self.previous_actual)/dt;alpha=dt/(.10+dt)
            self.measured_velocity+=alpha*(raw-self.measured_velocity)
        self.previous_actual=measured.copy()
        self.integral=np.clip(self.integral+dt*error,-c['integral_limit_m_s'],c['integral_limit_m_s'])
        requested=c['kp']*error+c['ki']*self.integral-c['kd']*self.measured_velocity
        requested*=min(1.,c['max_tcp_speed_mps']/max(np.linalg.norm(requested),1e-12))
        change=requested-self.velocity
        if c['max_tcp_acceleration_mps2'] is None:self.velocity=requested
        else:self.velocity+=change*min(1.,c['max_tcp_acceleration_mps2']*dt/max(np.linalg.norm(change),1e-12))
        lead=reference-measured
        if c['max_command_lead_m'] is not None and np.linalg.norm(lead)>c['max_command_lead_m'] and np.dot(self.velocity,lead)>0:
            self.hold();return self.q.copy()
        position=reference+self.velocity*dt
        position=np.clip(position,c['workspace_min_m'],c['workspace_max_m'])
        current_pitch=np.rad2deg(self.q[1]+self.q[2]+self.q[4])
        pitch=current_pitch+np.clip(c['tool_pitch_deg']-current_pitch,-c['tool_pitch_rate_deg_s']*dt,c['tool_pitch_rate_deg_s']*dt)
        solution=inverse_tool_pitch(position,self.q,float(np.clip(pitch,20,90)))
        delta=solution-self.q
        delta*=min(1.,float(np.min(self.joint_rate_rad_s*dt/np.maximum(np.abs(delta),1e-12))))
        # Joint interpolation during initial orientation alignment can change
        # TCP speed; bound the actual nonlinear FK step as well.
        for _ in range(4):
            distance=np.linalg.norm(forward(self.q+delta,[.16,0,0])[:3,3]-reference)
            if distance<=c['max_tcp_speed_mps']*dt+1e-10:break
            delta*=c['max_tcp_speed_mps']*dt/distance
        candidate=self.q+delta;point=forward(candidate,[.16,0,0])[:3,3]
        if np.linalg.norm(point-reference)>c['max_tcp_speed_mps']*dt+1e-9:
            raise ValueError('Proposed TCP command exceeds speed ceiling')
        if np.any(point<np.array(c['workspace_min_m'])-1e-8) or np.any(point>np.array(c['workspace_max_m'])+1e-8):
            raise ValueError('Proposed command outside TCP workspace')
        if np.max(np.abs(candidate-actual))>np.deg2rad(c['max_following_error_deg']):
            raise RuntimeError('Joint following error exceeds configured stall bound')
        return candidate

    def accept(self,candidate):
        self.q=np.array(candidate,float).copy()
