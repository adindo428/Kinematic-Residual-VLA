"""All quaternions are wxyz; rotations map local vectors into the parent frame."""
import numpy as np
from scipy.spatial.transform import Rotation


def rotation(q):
    q = np.asarray(q, dtype=float)
    return Rotation.from_quat(q[..., [1, 2, 3, 0]])


def quaternion(r):
    return r.as_quat()[..., [3, 0, 1, 2]]


def dom_euler(q):
    e = rotation(q).as_euler("xyz")
    e[..., [0, 2]] %= 2 * np.pi
    return e.astype(np.float32)


def mode_quaternion(mode, down):
    return quaternion(Rotation.from_rotvec([np.deg2rad(mode), 0, 0]) * rotation(down))


def angle_error(q_desired, q_measured):
    return (rotation(q_desired) * rotation(q_measured).inv()).as_rotvec()


def shift_point(position, q, old_offset, new_offset):
    return np.asarray(position) + rotation(q).apply(np.asarray(new_offset) - old_offset)


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def point_jacobian(jacobian, offset_world):
    j = np.array(jacobian, dtype=float, copy=True)
    j[:3] -= skew(offset_world) @ j[3:]
    return j


def limit_norm(v, maximum):
    v = np.asarray(v, dtype=float)
    return v * min(1.0, maximum / max(np.linalg.norm(v), 1e-12))
