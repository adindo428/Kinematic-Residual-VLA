"""Wrist observation freshness and the bounded first-mask startup handshake."""
import json
from pathlib import Path

import numpy as np


def read_wrist_observation(path, now_s, max_age_s=.25):
    """Return only fresh data, plus diagnostics that never count as observations."""
    try:
        record = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        return None, {'status': 'unavailable', 'reason': str(exc)}
    if not isinstance(record, dict) or record.get('schema') != 'wrist_pick_observation_v1':
        return None, {'status': 'invalid', 'reason': 'Invalid wrist observation schema'}
    if not record.get('running'):
        return None, {'status': 'stopped', 'reason': record.get('error') or 'Wrist feed stopped'}
    timing = record.get('timing') or {}
    stamp = timing.get('host_arrival_monotonic_s')
    if not isinstance(stamp, (int, float)) or not np.isfinite(stamp):
        return None, {'status': 'invalid', 'reason': 'Missing or invalid wrist capture timestamp'}
    age = float(now_s-stamp)
    diagnostics = {'status': 'fresh' if 0 <= age <= max_age_s else 'stale',
                   'age_s': age, 'max_age_s': max_age_s, 'reason': record.get('reason'),
                   'processing_state': record.get('processing_state'),
                   'processing_s': record.get('processing_s')}
    return (record if diagnostics['status'] == 'fresh' else None), diagnostics


class WristMaskStartup:
    """Wait at pregrasp without new goals, then require a genuinely fresh mask.

    This grace period applies only to the first prompt. It does not extend the
    freshness limit, reuse stale features, or retry an interrupted attempt.
    """
    def __init__(self, timeout_s=5.):
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('Wrist mask startup timeout must be positive')
        self.timeout_s = float(timeout_s)
        self.started_s = None
        self.ready = False

    def request(self, now_s):
        if self.started_s is None:
            self.started_s = float(now_s)

    def waiting(self, record, diagnostics, now_s):
        if self.started_s is None or self.ready:
            return False
        if diagnostics.get('status') in ('stopped', 'invalid'):
            raise RuntimeError('Wrist feed failed during mask initialization: '+str(diagnostics.get('reason')))
        if now_s-self.started_s > self.timeout_s:
            reason = diagnostics.get('reason') or diagnostics.get('status') or 'No reliable mask'
            raise RuntimeError(f'Wrist mask initialization timed out after {self.timeout_s:g}s: {reason}')
        # The caller passes only freshness-checked records. Require the flag
        # too so a stale snapshot cannot accidentally end the startup hold.
        feature = (record or {}).get('feature') or {}
        tracker = (record or {}).get('tracker') or {}
        depth = feature.get('depth_m')
        stamp = ((record or {}).get('timing') or {}).get('host_arrival_monotonic_s')
        fresh = (diagnostics.get('status') == 'fresh' and isinstance(stamp, (int, float))
                 and np.isfinite(stamp) and 0 <= now_s-stamp <= .25)
        # Wait for the first online propagation after seeding. The seed result
        # itself may use almost the entire 250ms budget before publication.
        if (fresh and tracker.get('seed') is False and feature.get('axis_reliable')
                and isinstance(depth, (int, float)) and np.isfinite(depth) and depth > 0):
            self.ready = True
            return False
        return True
