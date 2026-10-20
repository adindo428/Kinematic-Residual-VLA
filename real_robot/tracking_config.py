"""Configuration for the deliberately slow right-arm tracking pilot."""
from pathlib import Path
import math
import yaml

DEFAULT_CONTROL = dict(kp=1.5, ki=0., kd=0., integral_limit_m_s=.01,
    max_tcp_speed_mps=.015, max_tcp_acceleration_mps2=.05,
    max_joint_speed_deg_s=3., max_object_speed_mps=.05, max_lag_m=.05,
    orientation_kp=1.5, damping=.02)

def validate_control(values=None):
    values = values or {}
    if set(values)-set(DEFAULT_CONTROL): raise ValueError('Unknown tracking control setting')
    result = {**DEFAULT_CONTROL, **values}
    bounds = dict(kp=(0.,2.),ki=(0.,1.),kd=(0.,1.),integral_limit_m_s=(0.,.01),
        max_tcp_speed_mps=(.001,.015),max_tcp_acceleration_mps2=(.001,.05),
        max_joint_speed_deg_s=(.1,3.),max_object_speed_mps=(.001,.05),max_lag_m=(.005,.05),
        orientation_kp=(.1,1.5),damping=(.02,.2))
    for key,(lo,hi) in bounds.items():
        value=result[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not lo<=value<=hi:
            raise ValueError(f'{key} must be in [{lo}, {hi}] for this pilot')
    return result

def load_config(path):
    data=yaml.safe_load(Path(path).read_text())
    if not isinstance(data,dict) or data.get('schema')!='right_arm_tracking_v1':
        raise ValueError('Expected right_arm_tracking_v1 YAML configuration')
    if set(data)-{'schema','perception','arm','control','servo'}: raise ValueError('Unknown configuration section')
    data['control']=validate_control(data.get('control'))
    if 'servo' in data:
        from .cartesian_servo import validate_servo
        data['servo']=validate_servo(data['servo'])
    return data
