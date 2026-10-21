"""VX300s base_link -> gripper_link model; no actuator access.

Joint origins/axes follow the manufacturer description:
https://github.com/Interbotix/interbotix_ros_manipulators/blob/main/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/vx300s.urdf.xacro
The physical fingertip offset is an explicit input, not the gripper_link origin.
"""
import numpy as np

JOINT_NAMES = ('waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate')
MOTOR_IDS = (1, 2, 4, 6, 7, 8)
AXES = ((0, 0, 1), (0, 1, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0))
ORIGINS = ((0, 0, .079), (0, 0, .04805), (.05955, 0, .3),
           (.2, 0, 0), (.1, 0, 0), (.069744, 0, 0))
LOWER = np.deg2rad([-180, -106, -101, -180, -107, -180])
UPPER = np.deg2rad([180, 72, 92, 180, 128, 180])


def forward(q, tcp_offset_m=None):
    """Return T_base_gripper, or T_base_tcp with explicit gripper-frame offset."""
    q = np.asarray(q, dtype=float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError('Expected six finite model joint angles in radians')
    t = np.eye(4)
    for angle, axis, origin in zip(q, AXES, ORIGINS):
        axis = np.asarray(axis, dtype=float)
        skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]])
        a = np.eye(4)
        a[:3, 3] = origin
        a[:3, :3] = np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew@skew)
        t = t@a
    if tcp_offset_m is not None:
        offset = np.asarray(tcp_offset_m, dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError('TCP offset must be finite XYZ in the gripper frame')
        t[:3, 3] += t[:3, :3]@offset
    return t


def inverse(target, seed, tcp_offset_m):
    """Offline bounded IK only. Does not check scene or self collisions."""
    from scipy.optimize import least_squares
    import cv2
    from .geometry import validate_transform
    target = validate_transform(target)
    # Validate TCP before starting an optimizer.
    forward(seed, tcp_offset_m)
    def residual(q):
        actual = forward(q, tcp_offset_m)
        orientation = cv2.Rodrigues(target[:3, :3].T@actual[:3, :3])[0].ravel()
        return np.r_[actual[:3, 3]-target[:3, 3], .1*orientation]
    result = least_squares(residual, np.clip(seed, LOWER+1e-6, UPPER-1e-6),
                           bounds=(LOWER+1e-6, UPPER-1e-6), max_nfev=300,
                           ftol=1e-12, xtol=1e-12, gtol=1e-12)
    err = residual(result.x)
    if np.linalg.norm(err[:3]) > .001 or np.linalg.norm(err[3:])/.1 > np.deg2rad(.5):
        raise ValueError('No accurate bounded IK solution for this seed and target')
    return result.x


def jacobian(q, tcp_offset_m):
    """Geometric TCP Jacobian, [base-frame XYZ; base-frame angular velocity]."""
    endpoint = forward(q, tcp_offset_m)[:3, 3]
    transform = np.eye(4)
    origins, axes = [], []
    for angle, axis, origin in zip(q, AXES, ORIGINS):
        axis = np.asarray(axis, float)
        origins.append(transform[:3, 3]+transform[:3, :3]@origin)
        axes.append(transform[:3, :3]@axis)
        skew = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]],
                         [-axis[1], axis[0], 0.]])
        joint = np.eye(4)
        joint[:3, 3] = origin
        joint[:3, :3] = np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)
        transform = transform@joint
    return np.vstack((np.column_stack([np.cross(a, endpoint-o) for a, o in zip(axes, origins)]),
                      np.column_stack(axes)))


def inverse_downward(position, seed, tcp_length_m=.16):
    """Analytic IK for a vertical downward VX300s tool, preserving world yaw.

    Uses the exact link origins above, forearm roll zero, and both elbow
    branches. No optimizer, hardware access or collision checking.
    """
    result=inverse_tool_pitch(position,seed,90.,tcp_length_m)
    result[5]=result[0]
    return result


def inverse_tool_pitch(position,seed,pitch_deg=60.,tcp_length_m=.16):
    """Analytic position IK with a radial tool pitch; fingers stay level.

    Pitch is down from the horizontal. This frees a position-only tracker from
    a vertical-tool posture; caller must still check collision and load.
    """
    p=np.asarray(position,float); seed=np.asarray(seed,float)
    if p.shape!=(3,) or seed.shape!=(6,) or not np.isfinite([*p,*seed,tcp_length_m]).all() or not .05<=tcp_length_m<=.25:
        raise ValueError('Invalid downward IK position/seed/TCP')
    if not np.isfinite(pitch_deg) or not 20<=pitch_deg<=90:raise ValueError('Tool pitch must be 20..90 degrees')
    pitch=np.deg2rad(pitch_deg);tool=.069744+tcp_length_m
    radius=np.hypot(p[0],p[1]); waist=np.arctan2(p[1],p[0])
    radius-=tool*np.cos(pitch)
    height=p[2]-(.079+.04805)+tool*np.sin(pitch)
    first=np.hypot(.05955,.3); second=.3; offset=np.arctan2(.05955,.3)
    cosine=(radius**2+height**2-first**2-second**2)/(2*first*second)
    if abs(cosine)>1+1e-10: raise ValueError('Downward target is outside arm reach')
    candidates=[]
    for relative in [np.arccos(np.clip(cosine,-1,1)),-np.arccos(np.clip(cosine,-1,1))]:
        shoulder=np.arctan2(radius,height)-np.arctan2(second*np.sin(relative),first+second*np.cos(relative))-offset
        elbow=relative-np.pi/2+offset
        q=np.array([waist,shoulder,elbow,0.,pitch-shoulder-elbow,0.])
        if np.all(q>=LOWER) and np.all(q<=UPPER): candidates.append(q)
    if not candidates: raise ValueError('Downward target violates joint limits')
    return min(candidates,key=lambda q:np.linalg.norm(q-seed))
