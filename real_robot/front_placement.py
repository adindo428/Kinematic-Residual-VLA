"""Front-camera placement evidence, independent of wrist/side association."""
import numpy as np

from .geometry import validate_transform, transform_point
from .pick_geometry import inside_polygon


class FrontPlacementVerifier:
    def __init__(self, template, box):
        self.rig_sha = template['rig_sha256']
        self.front_serial = template['rig']['front_serial']
        self.T_base_front = validate_transform(template['rig']['T_base_front_candidate']).copy()
        self.box = box

    def evaluate(self, record, now, released_after_s):
        evidence = {'source':'front_efficienttam_rgbd','confirmed':False,
                    'reason':'waiting for gripper release','frame_stamp_s':None}
        if released_after_s is None:
            return False,None,evidence
        if (not isinstance(record,dict) or record.get('schema')!='front_side_observation_v1'
                or not record.get('running') or record.get('rig_sha256')!=self.rig_sha
                or record.get('output_frame')!='right_base_estimate'):
            evidence['reason']='front perception unavailable or rig/frame mismatch'
            return False,None,evidence
        front = (record.get('views') or {}).get('front')
        if not isinstance(front,dict):
            evidence['reason']='restart perception to publish front placement feedback'
            return False,None,evidence
        timing = front.get('timing') or {}
        captured = timing.get('host_arrival_monotonic_s')
        native = timing.get('sensor_timestamp_s')
        if (timing.get('camera_serial')!=self.front_serial
                or not isinstance(captured,(int,float)) or not np.isfinite(captured)
                or not isinstance(native,(int,float)) or not np.isfinite(native)
                or not 0<=now-captured<=.25):
            evidence['reason']='waiting for a fresh front-camera frame'
            return False,None,evidence
        evidence.update(frame_stamp_s=native,capture_monotonic_s=captured,
                        depth_frame_number=timing.get('depth_frame_number'))
        if captured<=released_after_s:
            evidence['reason']='front frame predates the gripper release'
            return False,native,evidence
        model = front.get('model_tracking') or {}
        if (model.get('tracker')!='EfficientTAM' or model.get('object_probability',0)<.5
                or model.get('mask_reason')):
            evidence['reason']='tape identity not visible in front camera'
            return False,native,evidence
        # These are raw per-view candidates, before the two-view association
        # gate. A hidden side view must not discard valid front evidence.
        candidates = [c for c in front.get('candidates',[])
                      if c.get('point_camera_m') is not None and not (c.get('quality') or {}).get('reason')]
        if len(candidates)!=1:
            evidence['reason']='waiting for a unique front tape depth measurement'
            return False,native,evidence
        try:
            point = transform_point(self.T_base_front,candidates[0]['point_camera_m'])
        except ValueError:
            evidence['reason']='invalid front tape depth measurement'
            return False,native,evidence
        inside = inside_polygon(point[:2],self.box['interior_polygon_xy_m'])
        landed_height = self.box['floor_z_base_m']-.015<=point[2]<=self.box['rim_z_base_m']+.010
        confirmed = bool(inside and landed_height)
        evidence.update(confirmed=confirmed,point_base_m=point.tolist(),inside_box=bool(inside),
            below_rim=bool(landed_height),
            reason='tape visible inside box below rim' if confirmed else 'tape has not been observed inside box below rim')
        return confirmed,native,evidence
