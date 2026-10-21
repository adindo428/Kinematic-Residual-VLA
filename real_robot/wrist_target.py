"""Camera-motion-compensated wrist target estimates; no device access.

The wrist surface is registered to the fixed-camera surface proxy while both
are visible. Encoder poses are interpolated at the wrist host capture time.
These estimates are demonstrator inputs, not policy observations.
"""
import numpy as np

from .automatic_demonstrator import ObjectState
from .vx300s_kinematics import forward


def camera_from_tcp(tcp, template):
    gripper = np.asarray(tcp, float).copy()
    gripper[:3,3] -= gripper[:3,:3]@np.asarray(template['tcp_offset_m'])
    return gripper@np.asarray(template['T_gripper_wrist_camera'])


def feature_point(feature, intrinsics):
    u,v = feature['center_px']
    z = feature['depth_m']
    return np.array([(u-intrinsics['ppx'])*z/intrinsics['fx'],
                     (v-intrinsics['ppy'])*z/intrinsics['fy'], z])


def project(point, direction, camera, intrinsics):
    p = camera[:3,:3].T@(point-camera[:3,3])
    axis = camera[:3,:3].T@direction
    if p[2] <= 0:
        return None
    fx,fy = intrinsics['fx'],intrinsics['fy']
    pixel = np.array([fx*p[0]/p[2]+intrinsics['ppx'],fy*p[1]/p[2]+intrinsics['ppy']])
    derivative = np.array([fx*(axis[0]*p[2]-p[0]*axis[2]),
                           fy*(axis[1]*p[2]-p[1]*axis[2])])/p[2]**2
    if np.linalg.norm(derivative) < 1e-9:
        return None
    return {'center_px':pixel.tolist(), 'axis_rad':float(np.arctan2(derivative[1],derivative[0])),
            'depth_m':float(p[2]), 'axis_reliable':True}


def capture_joints(history, stamp):
    """Interpolate feedback bracketing an image; never claim a future pose."""
    if not history or stamp < history[0][0] or stamp > history[-1][0]:
        return None
    times = np.array([t for t,_ in history])
    right = int(np.searchsorted(times,stamp,side='right'))
    if right == len(history):
        return np.asarray(history[-1][1]).copy()
    left = max(0,right-1)
    a,qa = history[left]; b,qb = history[right]
    return np.asarray(qa)+(np.asarray(qb)-qa)*((stamp-a)/(b-a) if b>a else 0.)


class WristTargetEstimator:
    def __init__(self, template, reference_pose):
        self.template = template
        self.reference_pose = np.asarray(reference_pose,float).copy()
        self.offset = None
        self.last_stamp = self.last_point = None
        self.velocity = np.zeros(3)
        self.last_state = None
        self.source = 'waiting'
        self.capture_pose = None
        self.live_registration = False
        self.prompt_generation = None

    def update(self, fixed, wrist, q_at_capture, now):
        self.capture_pose = None
        generation=((wrist or {}).get('tracker') or {}).get('prompt_generation')
        if generation is not None and generation!=self.prompt_generation:
            # Reacquisition must not turn a previous wrong mask's position into
            # a large object velocity or keep its surface registration offset.
            self.prompt_generation=generation
            self.last_stamp=self.last_point=None
            self.offset=None
            self.last_state=fixed
            self.velocity=np.zeros(3) if fixed is None else fixed.velocity_m_s.copy()
            self.live_registration=False
        feature = (wrist or {}).get('feature') or {}
        depth = feature.get('depth_m')
        valid = (q_at_capture is not None and isinstance(depth,(int,float))
                 and np.isfinite(depth) and depth > 0)
        stamp = ((wrist or {}).get('timing') or {}).get('host_arrival_monotonic_s')
        if valid:
            self.capture_pose = forward(q_at_capture)@np.asarray(self.template['T_gripper_wrist_camera'])
        if valid and stamp != self.last_stamp:
            intr = wrist['timing']['intrinsics']
            camera = self.capture_pose
            point = camera[:3,:3]@feature_point(feature,intr)+camera[:3,3]
            self.capture_pose = camera
            if self.offset is None:
                # The recorded pregrasp supplies an initial surface-to-proxy
                # offset if the robot occludes fixed views before first overlap.
                # Replace/refine it from simultaneous live views below.
                taught = np.asarray(self.template['stages']['pregrasp']['T_base_tcp'])
                taught_camera = camera_from_tcp(taught,self.template)
                surface = (taught_camera[:3,:3]@feature_point(self.template['wrist_reference'],intr)
                           +taught_camera[:3,3])
                self.offset = self.reference_pose[:3,3]-surface
            if fixed is not None:
                # Compare the two cameras at the same host time.
                reference = fixed.pose[:3,3]+fixed.velocity_m_s*(stamp-fixed.stamp_s)
                offset = reference-point
                self.offset = offset if self.last_stamp is None else .9*self.offset+.1*offset
                self.live_registration = True
            if self.last_stamp is not None and stamp > self.last_stamp:
                dt = stamp-self.last_stamp
                observed = (point-self.last_point)/dt
                self.velocity += dt/(.15+dt)*(observed-self.velocity)
            elif fixed is not None:
                self.velocity = fixed.velocity_m_s.copy()
            self.last_point,self.last_stamp = point.copy(),stamp
            if self.offset is not None:
                if fixed is not None:
                    pose = fixed.pose.copy()
                elif self.last_state is not None:
                    pose = self.last_state.pose.copy()
                else:
                    pose = self.reference_pose.copy()
                pose[:3,3] = point+self.offset
                # Upright tape rolls on the same tabletop; mask changes must
                # not create an artificial vertical interception velocity.
                if self.last_state is not None:
                    pose[2,3] = self.last_state.pose[2,3]
                self.velocity[2] = 0.
                self.last_state = ObjectState(pose,self.velocity,stamp,
                    semantics='fused_visible_surface_proxy')
        if valid and self.last_state is not None and 0 <= now-self.last_state.stamp_s <= .25:
            self.source = ('wrist_depth_registered_to_fixed_proxy' if self.live_registration
                           else 'wrist_depth_with_demonstrated_surface_offset')
            return self.last_state
        if fixed is not None:
            self.source = 'fixed_cameras'
            self.last_state = fixed
            return fixed
        self.source = 'unavailable'
        return None
