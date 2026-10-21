"""Rigid calibration: T_destination_source maps source points into destination."""
import numpy as np


def interpolate_pose(first, second, fraction):
    """Linear translation and shortest-path rotation between two poses."""
    from scipy.spatial.transform import Rotation
    first,second = np.asarray(first,float),np.asarray(second,float)
    result = first.copy()
    result[:3,3] = (1-fraction)*first[:3,3]+fraction*second[:3,3]
    rotation = Rotation.from_matrix(second[:3,:3]@first[:3,:3].T).as_rotvec()
    result[:3,:3] = Rotation.from_rotvec(rotation*fraction).as_matrix()@first[:3,:3]
    return result


def validate_transform(transform):
    t = np.asarray(transform, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError('Expected a finite 4x4 rigid transform')
    r = t[:3, :3]
    if not (np.allclose(t[3], [0, 0, 0, 1], atol=1e-7)
            and np.allclose(r.T @ r, np.eye(3), atol=1e-5)
            and np.isclose(np.linalg.det(r), 1, atol=1e-5)):
        raise ValueError('Transform must have a proper rotation and homogeneous last row')
    return t


def transform_point(transform, point):
    t = validate_transform(transform)
    p = np.asarray(point, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError('Expected a finite XYZ point in meters')
    return t[:3, :3] @ p + t[:3, 3]


def fit_rigid_transform(camera_points, base_points, *, min_points=4):
    """Fit paired, independently measured 3D landmarks; no scale fitting.

    Three points may be requested for explicitly unvalidated display estimates.
    Planar non-collinear landmarks are supported. These residuals are fit errors,
    not independent calibration accuracy. Validate on withheld landmarks too.
    """
    a, b = np.asarray(camera_points, float), np.asarray(base_points, float)
    if min_points < 3:
        raise ValueError('At least three non-collinear landmarks are required')
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) < min_points:
        raise ValueError(f'Need at least {min_points} paired XYZ landmarks')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Landmarks must be finite')
    ac, bc = a - a.mean(0), b - b.mean(0)
    if np.linalg.matrix_rank(ac, tol=1e-6) < 2 or np.linalg.matrix_rank(bc, tol=1e-6) < 2:
        raise ValueError('Collinear or coincident landmarks are not sufficient')
    u, _, vt = np.linalg.svd(ac.T @ bc)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    r = vt.T @ correction @ u.T
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, b.mean(0) - r @ a.mean(0)
    errors = np.linalg.norm(a @ r.T + t[:3, 3] - b, axis=1)
    return validate_transform(t), errors
