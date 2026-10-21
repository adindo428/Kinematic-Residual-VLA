"""Configuration for fixed-camera perception and continuous right-arm following."""
from pathlib import Path
import yaml

def load_config(path):
    data=yaml.safe_load(Path(path).read_text())
    if not isinstance(data,dict) or data.get('schema')!='right_arm_tracking_v1':
        raise ValueError('Expected right_arm_tracking_v1 YAML configuration')
    if set(data)-{'schema','perception','arm','servo'}: raise ValueError('Unknown configuration section')
    if 'servo' in data:
        from .cartesian_servo import validate_servo
        data['servo']=validate_servo(data['servo'])
    return data
