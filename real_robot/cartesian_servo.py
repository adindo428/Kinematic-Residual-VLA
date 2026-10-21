"""Following target adapter around the shared Cartesian pose/velocity servo."""
import numpy as np
from .vx300s_kinematics import forward, inverse_tool_pitch
from .task_space_servo import PoseVelocityServo

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
    """Retain the follow CLI/configuration while sharing its servo with picking."""
    def __init__(self, retained_goal, settings=None, joint_velocity_limits_deg_s=None, actual=None):
        self.c = validate_servo(settings)
        retained = np.asarray(retained_goal, float)
        self.joint_rates = joint_velocity_limits_deg_s if self.c['max_joint_speed_deg_s'] is None else self.c['max_joint_speed_deg_s']
        if self.joint_rates is None:
            raise ValueError('Read motor velocity limits before disabling the pilot joint-rate limit')
        self.core = PoseVelocityServo(retained if actual is None else actual, retained, [.16,0,0],
            kp=self.c['kp'], ki=self.c['ki'], kd=self.c['kd'], integral_limit=self.c['integral_limit_m_s'],
            max_following_error_deg=self.c['max_following_error_deg'],
            hardware_joint_rates_deg_s=joint_velocity_limits_deg_s)
        self.filtered_target = None
        self.pitch_reference = None

    @property
    def q(self): return self.core.goal

    @property
    def diagnostics(self): return self.core.diagnostics

    def hold(self):
        self.filtered_target = None
        self.pitch_reference = None
        return self.core.hold()

    def step(self, target, actual, dt, target_velocity_m_s=None):
        c = self.c
        target, actual = np.asarray(target,float), np.asarray(actual,float)
        if (not 0 < dt <= .15 or target.shape != (3,) or not np.isfinite(target).all()
                or actual.shape != (6,) or not np.isfinite(actual).all()):
            raise ValueError('Invalid servo target/timing')
        velocity = np.zeros(3) if target_velocity_m_s is None else np.asarray(target_velocity_m_s,float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError('Expected finite target velocity XYZ')
        if np.any(target < c['workspace_min_m']) or np.any(target > c['workspace_max_m']):
            raise ValueError('Target outside configured TCP workspace')
        inverse_tool_pitch(target, self.q, c['tool_pitch_deg'])
        if self.filtered_target is None:
            self.filtered_target = target.copy()
        else:
            self.filtered_target += velocity*dt
            self.filtered_target += (1-np.exp(-dt/max(c['target_smoothing_s'],1e-8)))*(target-self.filtered_target)
        self.filtered_target = np.clip(self.filtered_target,c['workspace_min_m'],c['workspace_max_m'])
        if self.pitch_reference is None:
            self.pitch_reference = float(np.rad2deg(actual[1]+actual[2]+actual[4]))
        self.pitch_reference += np.clip(c['tool_pitch_deg']-self.pitch_reference,
            -c['tool_pitch_rate_deg_s']*dt,c['tool_pitch_rate_deg_s']*dt)
        orientation_q = inverse_tool_pitch(self.filtered_target, actual, float(np.clip(self.pitch_reference,20,90)))
        pose = forward(orientation_q,[.16,0,0])
        pose[:3,3] = self.filtered_target
        measured = forward(actual,[.16,0,0])[:3,3]
        lead = forward(self.q,[.16,0,0])[:3,3]-measured
        drive = c['kp']*(self.filtered_target-measured)+velocity
        if c['max_command_lead_m'] is not None and np.linalg.norm(lead)>c['max_command_lead_m'] and np.dot(drive,lead)>0:
            return self.hold()
        candidate = self.core.step(pose,actual,dt,tcp_rate_m_s=c['max_tcp_speed_mps'],
            joint_rate_deg_s=self.joint_rates,angular_rate_deg_s=90.,
            acceleration_m_s2=c['max_tcp_acceleration_mps2'],target_velocity_m_s=velocity)
        point = forward(candidate,[.16,0,0])[:3,3]
        if np.any(point < np.asarray(c['workspace_min_m'])-1e-8) or np.any(point > np.asarray(c['workspace_max_m'])+1e-8):
            raise ValueError('Proposed command outside TCP workspace')
        return candidate

    def accept(self,candidate): self.core.accept(candidate)
