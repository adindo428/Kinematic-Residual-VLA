"""Causal constant-velocity Kalman filter. All timestamps share one clock."""
from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass
class Track:
    timestamp_s: float
    position_m: np.ndarray
    velocity_mps: np.ndarray
    covariance: np.ndarray
    measurement_age_s: float
    accepted: bool
    ready: bool
    status: str
    frame: str
    stationary_assumed: bool = False

    def as_dict(self):
        return {k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in vars(self).items()}


class VelocityTracker:
    def __init__(self, frame='camera_optical', measurement_std_m=0.004,
                 acceleration_std_mps2=0.8, max_gap_s=0.25, min_hits=8,
                 innovation_gate=16.27, max_speed_mps=0.5,
                 max_position_std_m=0.025, max_velocity_std_mps=0.15,
                 stationary_updates=True, stationary_window_s=1.0,
                 stationary_speed_mps=0.003, stationary_radius_m=0.006,
                 stationary_exit_displacement_m=0.004, zero_velocity_std_mps=0.003):
        values = [measurement_std_m, acceleration_std_mps2, max_gap_s,
                  innovation_gate, max_speed_mps, max_position_std_m, max_velocity_std_mps,
                  stationary_window_s, stationary_speed_mps, stationary_radius_m,
                  stationary_exit_displacement_m, zero_velocity_std_mps]
        if not np.isfinite(values).all() or min(values) <= 0 or min_hits < 2:
            raise ValueError('Tracker thresholds must be finite and positive; min_hits >= 2')
        self.frame = frame
        self.r = measurement_std_m ** 2 * np.eye(3)
        self.acceleration_std = acceleration_std_mps2
        self.max_gap, self.min_hits, self.gate = max_gap_s, min_hits, innovation_gate
        self.max_speed = max_speed_mps
        self.max_position_std, self.max_velocity_std = max_position_std_m, max_velocity_std_mps
        self.x = self.p = self.time = self.last_good = None
        self.hits = 0
        self.stationary_settings = dict(enabled=bool(stationary_updates),
            window_s=stationary_window_s, entry_speed_mps=stationary_speed_mps,
            radius_m=stationary_radius_m, exit_displacement_m=stationary_exit_displacement_m,
            zero_velocity_std_mps=zero_velocity_std_mps)
        self.recent = deque()
        self.stationary = False
        self.stationary_anchor = None
        self.stationary_cooldown_until = -float('inf')

    def _observe_motion(self, now, measurement):
        """Infer stillness from recent accepted positions, not filter speed alone.

        This is a motion assumption, not independent velocity ground truth. Very
        slow motion below the entry threshold cannot reliably be distinguished
        from position noise. No stationary constraint survives a missing sample.
        """
        settings = self.stationary_settings
        if not settings['enabled'] or measurement is None:
            self.recent.clear()
            self.stationary, self.stationary_anchor = False, None
            return
        self.recent.append((now, measurement.copy()))
        while now-self.recent[0][0] > settings['window_s']:
            self.recent.popleft()
        if len(self.recent) < 8 or now-self.recent[0][0] < .8*settings['window_s']:
            return
        times = np.array([t for t, _ in self.recent])
        positions = np.array([p for _, p in self.recent])
        times -= times.mean()
        center = positions.mean(axis=0)
        slope = times @ (positions-center) / (times @ times)
        spread = np.percentile(np.linalg.norm(positions-center-times[:, None]*slope, axis=1), 90)
        speed = np.linalg.norm(slope)
        if self.stationary:
            displacement = np.linalg.norm(positions[-3:].mean(axis=0)-self.stationary_anchor)
            if (displacement > settings['exit_displacement_m']
                    or speed > 2*settings['entry_speed_mps'] or spread > settings['radius_m']):
                self.stationary, self.stationary_anchor = False, None
                self.stationary_cooldown_until = now+.5*settings['window_s']
                # Restart velocity from measured recent motion rather than
                # turning accumulated position lag into a catch-up speed spike.
                self.x[3:] = slope
            return
        if (now >= self.stationary_cooldown_until
                and speed < settings['entry_speed_mps'] and spread < settings['radius_m']):
            self.stationary, self.stationary_anchor = True, center.copy()

    def _zero_velocity_update(self):
        h = np.c_[np.zeros((3, 3)), np.eye(3)]
        r = self.stationary_settings['zero_velocity_std_mps']**2*np.eye(3)
        innovation = h @ self.p @ h.T+r
        k = np.linalg.solve(innovation, h @ self.p).T
        self.x += k @ (-self.x[3:])
        a = np.eye(6)-k @ h
        self.p = a @ self.p @ a.T+k @ r @ k.T
        self.p = (self.p+self.p.T)/2

    def update(self, timestamp_s, measurement=None):
        now = float(timestamp_s)
        if not np.isfinite(now) or (self.time is not None and now <= self.time):
            raise ValueError('Timestamps must be finite and strictly increasing')
        z = None if measurement is None else np.asarray(measurement, float)
        if z is not None and (z.shape != (3,) or not np.isfinite(z).all()):
            raise ValueError('Measurement must be finite XYZ or None')
        expired = self.last_good is not None and now - self.last_good > self.max_gap
        if self.x is None or expired:
            self.time = now
            self.recent.clear()
            self.stationary, self.stationary_anchor = False, None
            self.stationary_cooldown_until = -float('inf')
            if z is None:
                self.x = self.p = self.last_good = None
                self.hits = 0
                return None
            self.x = np.r_[z, np.zeros(3)]
            self.p = np.diag([self.r[0, 0]] * 3 + [0.3 ** 2] * 3)
            self.last_good, self.hits = now, 1
            self._observe_motion(now, z)
            return self._snapshot(True)
        dt = now - self.time
        f = np.eye(6)
        f[:3, 3:] = dt * np.eye(3)
        g = np.vstack([0.5 * dt ** 2 * np.eye(3), dt * np.eye(3)])
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + self.acceleration_std ** 2 * (g @ g.T)
        self.time, accepted = now, False
        if z is not None:
            residual = z - self.x[:3]
            innovation = self.p[:3, :3] + self.r
            distance = float(residual @ np.linalg.solve(innovation, residual))
            if distance <= self.gate:
                k = np.linalg.solve(innovation, self.p[:3, :]).T
                self.x += k @ residual
                h = np.c_[np.eye(3), np.zeros((3, 3))]
                a = np.eye(6) - k @ h
                self.p = a @ self.p @ a.T + k @ self.r @ k.T
                self.p = (self.p + self.p.T) / 2
                self.last_good, self.hits, accepted = now, self.hits + 1, True
        self._observe_motion(now, z if accepted else None)
        if accepted and self.stationary:
            self._zero_velocity_update()
        return self._snapshot(accepted)

    def _snapshot(self, accepted):
        age = self.time - self.last_good
        position_std = np.sqrt(np.linalg.eigvalsh(self.p[:3, :3]).max())
        velocity_std = np.sqrt(np.linalg.eigvalsh(self.p[3:, 3:]).max())
        status = 'tracking'
        if self.hits < self.min_hits:
            status = 'warming_up'
        elif not accepted:
            status = 'coasting'
        elif position_std > self.max_position_std or velocity_std > self.max_velocity_std:
            status = 'uncertain'
        elif np.linalg.norm(self.x[3:]) > self.max_speed:
            status = 'too_fast'
        # New measurement required for candidate controller commands.
        return Track(self.time, self.x[:3].copy(), self.x[3:].copy(), self.p.copy(),
                     age, accepted, status == 'tracking', status, self.frame, self.stationary)
