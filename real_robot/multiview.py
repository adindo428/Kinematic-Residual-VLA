"""Camera-only acquisition, relative calibration, and gated 3D proxy fusion.

No actuator imports. T_front_side maps side optical coordinates into front
optical coordinates. A shared global timestamp domain does not synchronize
camera exposures; the complete RGB/depth timestamp span is checked explicitly.
"""
from collections import deque
import threading
import time

import cv2
import numpy as np

from .geometry import validate_transform, transform_point

FRONT_SERIAL = '234222300865'
SIDE_SERIAL = '218722271409'


def timestamp_span(front, side):
    domains = [f['timing']['timestamp_domain'] for f in (front, side)]
    if domains[0] != domains[1] or not any(x in domains[0] for x in ('global_time', 'system_time')):
        raise ValueError('Cross-camera pairing needs a common global/system clock, not independent hardware clocks')
    stamps = np.array([f['timing'][k] for f in (front, side)
                       for k in ('sensor_timestamp_s', 'color_timestamp_s')])
    if not np.isfinite(stamps).all(): raise ValueError('Nonfinite camera timestamp')
    return float(stamps.max()-stamps.min()), float(stamps.max())


class FramePairer:
    """Consume each frame at most once, in causal capture order."""
    def __init__(self, max_skew_s=.035, max_age_s=.5):
        if not np.isfinite([max_skew_s, max_age_s]).all() or min(max_skew_s, max_age_s) <= 0:
            raise ValueError('Pairing thresholds must be positive')
        self.max_skew, self.max_age = max_skew_s, max_age_s
        self.queues = {'front': deque(), 'side': deque()}
        self.last_stamp = {}; self.dropped = {'front': 0, 'side': 0}

    def add(self, role, frames):
        for frame in frames:
            t = frame['timing']['sensor_timestamp_s']
            if not np.isfinite(t) or (role in self.last_stamp and t <= self.last_stamp[role]):
                raise ValueError('Camera timestamps must increase strictly')
            self.last_stamp[role] = t; self.queues[role].append(frame)
            if len(self.queues[role]) > 60:
                self.queues[role].popleft(); self.dropped[role] += 1

    def pop(self, now):
        for role, queue in self.queues.items():
            while queue and now-queue[0]['timing']['host_arrival_monotonic_s'] > self.max_age:
                queue.popleft(); self.dropped[role] += 1
        front, side = self.queues['front'], self.queues['side']
        while front and side:
            span, stamp = timestamp_span(front[0], side[0])
            if span <= self.max_skew:
                return front.popleft(), side.popleft(), {'span_s': span, 'timestamp_s': stamp}
            # Too-old samples cannot acquire a closer future partner from the
            # other monotonic stream. Never reuse an already matched frame.
            role = 'front' if front[0]['timing']['sensor_timestamp_s'] < side[0]['timing']['sensor_timestamp_s'] else 'side'
            self.queues[role].popleft(); self.dropped[role] += 1
        return None


class CameraPair:
    """Independent read threads prevent one camera wait from blocking the other."""
    def __init__(self, front=FRONT_SERIAL, side=SIDE_SERIAL, fps=15, max_skew_s=.035):
        if front == side: raise ValueError('Need two different cameras')
        self.serials = {'front': front, 'side': side}; self.fps = fps
        self.pairer = FramePairer(max_skew_s)
        self.stop = threading.Event(); self.lock = threading.Lock()
        self.pending = {'front': deque(maxlen=30), 'side': deque(maxlen=30)}
        self.latest = {}; self.errors = {}; self.overruns = {'front': 0, 'side': 0}
        self.threads = []

    def start(self):
        for role in self.serials:
            thread = threading.Thread(target=self._read, args=(role,), daemon=True)
            self.threads.append(thread); thread.start()
        return self

    def _read(self, role):
        from .camera import RGBDCamera
        camera = None; previous = None; domain = None
        try:
            camera = RGBDCamera(self.serials[role], fps=self.fps)
            while not self.stop.is_set():
                bgr, depth, timing, deproject = camera.read()
                stamp = timing['sensor_timestamp_s']
                if ((previous is not None and stamp <= previous)
                        or (domain is not None and timing['timestamp_domain'] != domain)):
                    raise RuntimeError('Camera clock repeated or changed domain')
                previous, domain = stamp, timing['timestamp_domain']
                frame = dict(bgr=bgr, depth=depth, timing=timing, deproject=deproject)
                with self.lock:
                    if len(self.pending[role]) == self.pending[role].maxlen: self.overruns[role] += 1
                    self.pending[role].append(frame); self.latest[role] = frame
        except BaseException as exc:
            with self.lock: self.errors[role] = f'{type(exc).__name__}: {exc}'
        finally:
            if camera is not None: camera.close()

    def poll(self):
        with self.lock:
            errors = dict(self.errors)
            drained = {r: list(q) for r, q in self.pending.items()}
            for q in self.pending.values(): q.clear()
        if errors: raise RuntimeError(str(errors))
        for role, frames in drained.items(): self.pairer.add(role, frames)
        return self.pairer.pop(time.monotonic())

    def views(self):
        with self.lock: return dict(self.latest)

    def close(self):
        self.stop.set()
        for thread in self.threads: thread.join(timeout=15)
        if any(t.is_alive() for t in self.threads):
            raise RuntimeError('Camera reader has not stopped; do not open another pipeline')


def charuco_detector(square_m=.05, marker_m=.0375):
    board = cv2.aruco.CharucoBoard((6, 6), square_m, marker_m,
                                 cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    params = cv2.aruco.DetectorParameters(); params.detectInvertedMarker = True
    return board, cv2.aruco.CharucoDetector(board, detectorParams=params)


def estimate_board(frame, board, detector, min_corners=12):
    cc, ci, mc, mi = detector.detectBoard(frame['bgr'])
    result = {'corner_count': 0 if ci is None else len(ci), 'valid': False,
              'reason': 'not_enough_corners'}
    if ci is not None:
        result.update(ids=ci.ravel().astype(int).tolist(), pixels=cc.reshape(-1, 2).tolist())
    if ci is None or len(ci) < min_corners: return result
    ids = ci.ravel().astype(int); pixels = cc.reshape(-1, 2).astype(float)
    obj = board.getChessboardCorners()[ids].astype(float)
    if np.linalg.matrix_rank(obj-obj.mean(0)) < 2:
        return {**result, 'reason': 'collinear_corners'}
    rays = np.array([frame['deproject'](p.tolist(), 1.) for p in pixels], float)
    norm = rays[:, :2]/rays[:, 2:]
    ok, rv, tv = cv2.solvePnP(obj, norm, np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return {**result, 'reason': 'pnp_failed'}
    rv, tv = cv2.solvePnPRefineLM(obj, norm, np.eye(3), None, rv, tv)
    pose = np.eye(4); pose[:3, :3] = cv2.Rodrigues(rv)[0]; pose[:3, 3] = tv.ravel()
    xyz = obj@pose[:3, :3].T+pose[:3, 3]
    if np.any(xyz[:, 2] <= .03): return {**result, 'reason': 'board_behind_camera'}
    intr = frame['timing']['intrinsics']; scale = np.array([intr['fx'], intr['fy']])
    residual = (xyz[:, :2]/xyz[:, 2:]-norm)*scale
    rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    result.update(ids=ids.tolist(), pixels=pixels.tolist(), normalized_pixels=norm.tolist(),
                  object_points_m=obj.tolist(), T_camera_board=pose.tolist(),
                  ray_rmse_equivalent_px=rmse, timing=frame['timing'],
                  valid=rmse < 1.5, reason=None if rmse < 1.5 else 'large_ray_residual')
    return result


def pose_difference(a, b):
    a, b = validate_transform(a), validate_transform(b)
    return (float(np.linalg.norm(a[:3, 3]-b[:3, 3])),
            float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(a[:3, :3].T@b[:3, :3])[0]))))


def fit_relative(observations, fit_indices, heldout_indices):
    """Joint ray bundle fit, with completely held-out board poses for checking."""
    from scipy.optimize import least_squares
    if len(fit_indices) < 5 or len(heldout_indices) < 2 or set(fit_indices)&set(heldout_indices):
        raise ValueError('Need >=5 fit poses and >=2 distinct held-out poses')
    if len(set(fit_indices+heldout_indices)) != len(fit_indices)+len(heldout_indices):
        raise ValueError('Repeated observation indices')
    selected = fit_indices+heldout_indices
    for k, i in enumerate(selected):
        for j in selected[:k]:
            distance, angle = pose_difference(observations[i]['front']['T_camera_board'],
                                              observations[j]['front']['T_camera_board'])
            if distance < .02 and angle < 8:
                raise ValueError('Need distinct board poses, not repeated frames of one pose')
    def pack(t): return np.r_[cv2.Rodrigues(np.asarray(t)[:3, :3])[0].ravel(), np.asarray(t)[:3, 3]]
    def unpack(x):
        t = np.eye(4); t[:3, :3] = cv2.Rodrigues(x[:3])[0]; t[:3, 3] = x[3:6]; return t
    relative = []
    for row in observations:
        relative.append(validate_transform(row['front']['T_camera_board'])@
                        np.linalg.inv(validate_transform(row['side']['T_camera_board'])))
    # Initialize from a transform medoid; alternative planar PnP solutions become
    # large residuals rather than being silently averaged into the rig transform.
    medoid = min(fit_indices, key=lambda i: sum(pose_difference(relative[i], relative[j])[0]
                   +.01*pose_difference(relative[i], relative[j])[1] for j in fit_indices))
    x0 = np.concatenate([pack(relative[medoid])]+[pack(observations[i]['front']['T_camera_board']) for i in fit_indices])
    def ray_error(pose, row):
        p = np.asarray(row['object_points_m'])@pose[:3, :3].T+pose[:3, 3]
        z = np.maximum(p[:, 2:], .01)
        intr = row['timing']['intrinsics']
        return ((p[:, :2]/z-np.asarray(row['normalized_pixels']))*[intr['fx'], intr['fy']]).ravel()
    def residual(x):
        fs = unpack(x[:6]); out = []
        for k, i in enumerate(fit_indices):
            fb = unpack(x[6+6*k:12+6*k]); row = observations[i]
            out.extend(ray_error(fb, row['front'])); out.extend(ray_error(np.linalg.inv(fs)@fb, row['side']))
        return np.asarray(out)
    result = least_squares(residual, x0, loss='soft_l1', f_scale=.7, max_nfev=300,
                           ftol=1e-10, xtol=1e-10, gtol=1e-10)
    fs = validate_transform(unpack(result.x[:6])); checks = []
    for i in heldout_indices:
        row = observations[i]
        fb = np.array(row['front']['T_camera_board'])
        predicted_side = np.linalg.inv(fs)@fb
        errors = ray_error(predicted_side, row['side']).reshape(-1, 2)
        translation, angle = pose_difference(fs, relative[i])
        checks.append({'index': i, 'side_ray_rmse_equivalent_px': float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))),
                       'relative_translation_difference_m': translation, 'relative_rotation_difference_deg': angle})
    fit_rmse = float(np.sqrt(np.mean(residual(result.x)**2)))
    validated = bool(result.success and fit_rmse < 1.0
                     and all(x['side_ray_rmse_equivalent_px'] < 2.0
                             and x['relative_translation_difference_m'] < .015
                             and x['relative_rotation_difference_deg'] < 1.5 for x in checks))
    return {'T_front_side': fs.tolist(), 'fit_indices': fit_indices, 'heldout_indices': heldout_indices,
            'fit_ray_rmse_equivalent_px_per_axis': fit_rmse, 'heldout_checks': checks,
            'relative_calibration_validated': validated, 'solver_success': bool(result.success)}


def fuse_proxy(front_point, side_point, T_front_side, front_sigma=.015, side_sigma=.015,
               max_disagreement_m=.06):
    """Fuse visible-surface position proxies, not a true volumetric centroid.

    A 15 mm uncertainty floor and observed view spread are retained: correlated
    calibration/shape bias is not made artificially precise by averaging.
    """
    transform = validate_transform(T_front_side)
    values = [front_sigma, side_sigma, max_disagreement_m]
    if not np.isfinite(values).all() or min(values) <= 0: raise ValueError('Invalid fusion thresholds')
    points, sigmas, sources = [], [], []
    for role, point, sigma in [('front', front_point, front_sigma), ('side', side_point, side_sigma)]:
        if point is None: continue
        p = np.asarray(point, float)
        if p.shape != (3,) or not np.isfinite(p).all(): raise ValueError('Nonfinite per-camera position')
        points.append(p if role == 'front' else transform_point(transform, p)); sigmas.append(sigma); sources.append(role)
    out = {'position_front_m': None, 'covariance_front_m2': None, 'sources': sources,
           'position_semantics': 'fused_visible_surface_proxy', 'motion_sent': False}
    if not points: return {**out, 'status': 'lost'}
    disagreement = float(np.linalg.norm(points[0]-points[1])) if len(points) == 2 else None
    if disagreement is not None and disagreement > max_disagreement_m:
        return {**out, 'status': 'view_disagreement', 'view_disagreement_m': disagreement}
    p = np.array(points); weights = 1/np.square(sigmas); weights /= weights.sum()
    center = weights@p; residual = p-center
    covariance = np.eye(3)*max(.015**2, 1/sum(1/np.square(sigmas)))
    covariance += (residual.T*weights)@residual
    return {**out, 'status': 'fused' if len(points) == 2 else 'single_view',
            'position_front_m': center.tolist(), 'covariance_front_m2': covariance.tolist(),
            'view_disagreement_m': disagreement}


class ProxyAssociation:
    """Acquire from a unique cross-view match; retain identity through short gaps.

    Color regions alone do not establish target identity. After 0.5 s without
    an associated measurement, reacquisition requires both cameras again.
    """
    def __init__(self, T_front_side, max_disagreement_m=.06, max_speed_mps=.5, grace_s=.5):
        self.transform = validate_transform(T_front_side)
        if not np.isfinite([max_disagreement_m, max_speed_mps, grace_s]).all() or min(max_disagreement_m, max_speed_mps, grace_s) <= 0:
            raise ValueError('Invalid association limits')
        self.disagreement, self.speed, self.grace = max_disagreement_m, max_speed_mps, grace_s
        self.last_point = self.last_good = self.last_time = None

    def update(self, stamp, candidates):
        if not np.isfinite(stamp) or (self.last_time is not None and stamp <= self.last_time):
            raise ValueError('Association timestamps must increase')
        self.last_time = stamp
        recent = self.last_good is not None and stamp-self.last_good <= self.grace
        gate = .06+self.speed*(stamp-self.last_good) if recent else None
        points = {}
        for role in ('front', 'side'):
            points[role] = []
            for i, point in enumerate(candidates[role]):
                p = np.asarray(point, float)
                if p.shape != (3,) or not np.isfinite(p).all(): raise ValueError('Invalid candidate position')
                p = p if role == 'front' else transform_point(self.transform, p)
                if not recent or np.linalg.norm(p-self.last_point) <= gate:
                    points[role].append((i, p))
        matches = [(i, j) for i, a in points['front'] for j, b in points['side']
                   if np.linalg.norm(a-b) <= self.disagreement]
        selected = {'front': None, 'side': None}; reason = None
        if len(matches) == 1:
            selected['front'], selected['side'] = matches[0]
        elif len(matches) > 1:
            reason = 'association_ambiguous'
        elif recent and sum(map(len, points.values())) == 1:
            role = 'front' if points['front'] else 'side'
            selected[role] = points[role][0][0]
        else:
            reason = 'awaiting_two_view_reacquisition' if not recent else 'association_lost_or_disagreeing'
        result = fuse_proxy(*(None if selected[r] is None else candidates[r][selected[r]] for r in ('front', 'side')),
                            self.transform, max_disagreement_m=self.disagreement)
        if reason is not None: result['status'] = reason
        if result['position_front_m'] is not None:
            self.last_point = np.array(result['position_front_m']); self.last_good = stamp
        result['selected_candidate_indices'] = selected
        result['identity_recent'] = recent
        return result
