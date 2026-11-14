"""Clutched right-leader control for manual data collection only."""
import time
import numpy as np

from .grasp_capture import TeachingControl


class ClutchedIO:
    """Anchor leader motion at the current follower pose, with no enable jump."""
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.follower = wrapped.follower
        self.offset = np.zeros(8)
        self.gripper_anchor = (0., 0.)

    def anchor(self, leader, follower):
        leader, follower = np.asarray(leader, float), np.asarray(follower, float)
        self.offset = follower[:8] - leader[:8]
        self.gripper_anchor = (float(np.clip(leader[-1],0,100)),
                               float(np.clip(follower[-1],0,100)))

    def map_leader(self, leader):
        result = np.asarray(leader, float).copy()
        result[:8] += self.offset
        # Continuous at clutch-in, while retaining the full gripper travel.
        initial, actual = self.gripper_anchor
        current = float(np.clip(result[-1], 0, 100))
        if current >= initial:
            result[-1] = actual + (current-initial)/max(100-initial,1e-6)*(100-actual)
        else:
            result[-1] = actual - (initial-current)/max(initial,1e-6)*actual
        return result

    def read(self):
        row = self.wrapped.read()
        row['leader_input_calibrated'] = list(row['leader_calibrated'])
        row['leader_calibrated'] = self.map_leader(row['leader_input_calibrated']).tolist()
        row['leader_clutch_offset_deg'] = self.offset.tolist()
        row['gripper_clutch_anchor_percent'] = list(self.gripper_anchor)
        return row

    def health(self):
        return self.wrapped.health()

    def goal(self, target):
        self.wrapped.goal(target)


class CollectionControl(TeachingControl):
    """Retain heartbeat/rate/health guards; replace absolute matching with clutching."""
    def __init__(self, io, execute, tcp_offset, fps=30, max_relative_target=5.):
        super().__init__(ClutchedIO(io),execute,tcp_offset,fps,max_relative_target=max_relative_target)
        self.torque_enabled_here = False

    def toggle(self):
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            if self.enabled:
                self.enabled = False
                return False
            if not self.execute:
                raise ValueError('Preview only; restart with --execute to follow')
            if self.stop.is_set() or self.latest is None:
                raise ValueError('Waiting for robot feedback')
            if time.monotonic()-self.latest['host_read_end_monotonic_s'] > .15:
                raise ValueError('Robot feedback is stale')
            health = self.io.health()
            torque = np.asarray(health['torque'])
            if np.any(health['hardware_errors']):
                raise ValueError('Follower reports a hardware fault')
            if not (np.all(torque == 0) or np.all(torque == 1)):
                raise ValueError('Mixed follower torque states; inspect the motor fault before enabling')
            row = self.io.read()
            if max(row['temperatures_c']) >= 55:
                raise ValueError('Follower temperature guard')
            actual = np.asarray(row['follower_calibrated'])
            if np.all(torque == 0):
                # T is explicit clutch-in authorization. Seed the current pose
                # before energizing; never send a distant leader pose on enable.
                self.io.goal(actual)
                self.io.follower.write('Torque_Enable',np.ones(9,np.int32))
                self.torque_enabled_here = True
                if not np.all(self.io.health()['torque'] == 1):
                    raise RuntimeError('Follower holding torque did not enable')
            self.io.anchor(row['leader_input_calibrated'],actual)
            self.enabled = True
            self.heartbeat = time.monotonic()
            return True
