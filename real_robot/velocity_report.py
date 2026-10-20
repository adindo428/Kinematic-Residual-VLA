"""Report causal filtered surface-proxy velocities, with reference speed bands."""
import numpy as np


def velocity_report(track, stale=False, reference=(.2, .8)):
    result = {'velocity_mps': None, 'speed_mps': None, 'speed_cm_s': None,
              'speed_std_mps': None, 'reference_range_mps': list(reference),
              'reference': 'DynamicVLA v2 Appendix D real-world release speeds',
              'reference_url': 'https://arxiv.org/html/2601.22153v2',
              'status': 'unavailable', 'range_assessment': 'unavailable',
              'semantics': 'filtered visible-surface proxy velocity; not independently measured object-centroid ground truth'}
    if stale or track is None or not track.accepted or not track.ready:
        return {**result, 'status': 'stale' if stale else ('unavailable' if track is None else track.status)}
    v = np.asarray(track.velocity_mps); cov = np.asarray(track.covariance)[3:, 3:]
    speed = float(np.linalg.norm(v))
    std = float(np.sqrt(max(0., (v/speed)@cov@(v/speed) if speed > 1e-6 else np.linalg.eigvalsh(cov).max())))
    lo, hi = reference
    band = 'below' if speed+2*std < lo else ('above' if speed-2*std > hi else
            ('inside' if speed-2*std >= lo and speed+2*std <= hi else 'boundary_or_uncertain'))
    return {**result, 'velocity_mps': v.tolist(), 'speed_mps': speed, 'speed_cm_s': 100*speed,
            'speed_std_mps': std, 'status': 'estimated', 'range_assessment': band,
            'uncertainty_note': 'Kalman state uncertainty excludes shape bias and unmeasured capture latency.'}
