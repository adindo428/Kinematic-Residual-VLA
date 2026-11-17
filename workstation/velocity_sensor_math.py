"""Causal velocity measurements and estimator diagnostics, in robot-base metres."""
import hashlib
from collections import deque
import numpy as np


def gaussian_velocity(truth, sigma, scene, repeat, index, seed=16311):
    """A stateless, model-independent draw for each scene/repeat/observation."""
    assert np.isfinite(sigma) and sigma >= 0
    key = f'{seed}:{scene}:{repeat}:{index}'.encode()
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')
    return np.asarray(truth) + sigma*np.random.default_rng(value).standard_normal(3)


def rotation(q):
    """Scalar-first quaternion, mapping local vectors into the parent frame."""
    w, x, y, z = np.asarray(q, dtype=float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def surface_position(mask, depth, intrinsic, camera_pos, camera_quat, robot_pos, robot_quat):
    """Visible-surface centroid, not an estimate of hidden object COM."""
    mask = np.asarray(mask, bool)
    depth = np.asarray(depth).squeeze()
    valid = mask & np.isfinite(depth) & (depth > .02) & (depth < 5.)
    ys, xs = np.nonzero(valid)
    if len(xs) < 20:
        return None
    z = depth[ys, xs]
    median = np.median(z)
    # Reject isolated depth/background pixels without consulting object state.
    keep = np.abs(z-median) <= max(.02, 3*np.median(np.abs(z-median)))
    xs, ys, z = xs[keep], ys[keep], z[keep]
    if len(xs) < 20:
        return None
    k = np.asarray(intrinsic)
    optical = np.stack([(xs-k[0, 2])*z/k[0, 0], (ys-k[1, 2])*z/k[1, 1], z], axis=1)
    world = optical@rotation(camera_quat).T + camera_pos
    base = (world-robot_pos)@rotation(robot_quat)
    return np.median(base, axis=0)


class CausalSlope:
    def __init__(self, window_s=.16):
        self.window_s = window_s
        self.points = deque()

    def update(self, timestamp, position):
        while self.points and timestamp-self.points[0][0] > self.window_s+1e-8:
            self.points.popleft()
        if position is None:
            self.points.clear()
            return np.zeros(3), False, 0.
        self.points.append((float(timestamp), np.asarray(position).copy()))
        if len(self.points) < 3:
            return np.zeros(3), False, 0.
        t = np.array([p[0] for p in self.points]);t -= t.mean()
        p = np.stack([p[1] for p in self.points])
        return t@p/(t@t), True, float(timestamp-np.mean([p[0] for p in self.points]))


def error_metrics(errors):
    a = np.asarray(errors, float).reshape(-1, 3)
    if not len(a):
        return {'samples': 0}
    return dict(samples=len(a), bias_xyz=a.mean(0).tolist(),
                std_xyz=a.std(0).tolist(), rmse_xyz=np.sqrt(np.mean(a*a, axis=0)).tolist(),
                sigma_equivalent=float(np.sqrt(np.mean(a*a))),
                rmse_vector=float(np.sqrt(np.mean(np.sum(a*a, axis=1)))),
                p95_vector=float(np.percentile(np.linalg.norm(a, axis=1), 95)))
