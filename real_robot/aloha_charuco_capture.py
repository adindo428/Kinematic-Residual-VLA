"""Right ALOHA leader teleoperation and raw hand-eye observations.

Run in the LeRobot Python environment, with both repositories on PYTHONPATH.
No torque, operating-mode, or calibration-register writes are performed.
Only operator-enabled leader following writes Goal_Position to the right arm.
These observations are not a validated robot/camera calibration.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import threading
import time
from collections import deque

import cv2
import numpy as np


def following_target(leader, follower, max_relative_target=5.):
    """Use the same per-step relative-target clamp as LeRobot teleoperation."""
    leader, follower = np.asarray(leader), np.asarray(follower)
    if not np.isfinite(leader).all() or not np.isfinite(follower).all():
        raise ValueError('Nonfinite joint reading')
    if not np.isfinite(max_relative_target) or max_relative_target <= 0:
        raise ValueError('Relative target bound must be positive and finite')
    return follower + np.clip(leader-follower, -max_relative_target, max_relative_target)


def raw_positions(bus):
    # Temporarily bypass only the software angle conversion, retaining raw ticks.
    calibration = bus.calibration
    try:
        bus.calibration = None
        return bus.read('Present_Position')
    finally:
        bus.calibration = calibration


def calibrated_positions(bus, raw):
    # No automatic adjustment of calibration offsets during capture.
    return bus.apply_calibration(raw.copy(), bus.motor_names)


class LeaderFollowing:
    """Single serial owner; camera waits and image saves do not set control rate."""
    def __init__(self, leader, follower, fps=30):
        self.leader, self.follower, self.fps = leader, follower, fps
        self.lock = threading.Lock()
        self.enabled = threading.Event()
        self.stop = threading.Event()
        self.history = deque(maxlen=120)
        self.mismatch = 0.
        self.error = None
        self.thread = threading.Thread(target=self._run, name='right-leader-following', daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                with self.lock:
                    raw = raw_positions(self.follower)
                    actual = calibrated_positions(self.follower, raw)
                    target = calibrated_positions(self.leader, raw_positions(self.leader))
                    self.mismatch = float(np.max(np.abs(target[:-1]-actual[:-1])))
                    self.history.append((time.monotonic(), actual.copy()))
                    if self.enabled.is_set():
                        self.follower.write('Goal_Position', following_target(target, actual).astype(np.float32))
                self.stop.wait(max(0., 1/self.fps-(time.monotonic()-started)))
        except Exception as exc:
            self.error = exc
            self.enabled.clear()
            self.stop.set()

    def snapshot(self):
        with self.lock:
            if self.error is not None:
                raise RuntimeError('Teleoperation failed; following stopped') from self.error
            started = time.monotonic()
            raw = raw_positions(self.follower)
            return started, time.monotonic(), raw, calibrated_positions(self.follower, raw)

    def toggle(self):
        with self.lock:
            if self.error is not None:
                raise RuntimeError('Teleoperation failed') from self.error
            if self.enabled.is_set():
                self.enabled.clear()
                return False
            if not np.all(self.follower.read('Torque_Enable') == 1):
                return False
            self.enabled.set()
            return True

    def diagnostics(self):
        with self.lock:
            history = list(self.history)
            hz = ((len(history)-1)/(history[-1][0]-history[0][0])) if len(history)>1 else 0.
            recent = [(t,p) for t,p in history if t >= time.monotonic()-.8]
            still = (len(recent)>=8 and recent[-1][0]-recent[0][0]>=.5
                     and np.max(np.ptp(np.array([p[:-1] for _,p in recent]), axis=0)) <= .3)
            return hz, bool(still), self.mismatch

    def close(self):
        # Lock serial writes before clearing enable; no command after this point.
        with self.lock:
            self.enabled.clear()
            self.stop.set()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError('Teleoperation thread has not finished; serial ports remain owned')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--calibration-dir', default='/home/ailab/lerobot/.cache/calibration/aloha_default')
    parser.add_argument('--marker-length', type=float, default=.0375,
                        help='Provisional marker length in meters; must be physically verified before fitting')
    parser.add_argument('--seconds', type=float, default=1800)
    args = parser.parse_args()
    if not 0 < args.marker_length < .05 or not 0 < args.seconds <= 3600:
        raise ValueError('Invalid marker length or session duration')
    sdk_version = importlib.metadata.version('pyrealsense2')
    if tuple(map(int, sdk_version.split('.')[:2])) < (2, 59):
        raise RuntimeError('Use the repository .venv with pyrealsense2>=2.59. The older LeRobot SDK reports inconsistent D405 color intrinsics.')
    from lerobot.common.robot_devices.robots.configs import AlohaRobotConfig
    from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
    from .camera import RGBDCamera
    cv2.setNumThreads(1)
    config = AlohaRobotConfig()
    buses = {}
    for role, bus_config in [('leader', config.leader_arms['right']),
                             ('follower', config.follower_arms['right'])]:
        owner = subprocess.run(['fuser', bus_config.port], capture_output=True, text=True)
        if owner.returncode == 0:
            raise RuntimeError(f'{bus_config.port} already in use: {owner.stdout.strip()}')
        buses[role] = DynamixelMotorsBus(bus_config)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 'aloha_charuco_observations_v1', 'arm': 'right',
                'robot_model_candidate': 'vx300s', 'kinematics_validated': False,
                'board': {'squares': [6, 6], 'square_length_m': .05,
                          'marker_length_m': args.marker_length,
                          'marker_length_physically_verified': False,
                          'dictionary': 'DICT_4X4_50', 'legacy_pattern': False},
                'wrist_serial': '218622278247', 'front_serial': '234222300865',
                'calibration_validated': False, 'torque_writes': False,
                'pyrealsense2_version': sdk_version,
                'note': 'Raw observations only. Joint conventions and physical board dimensions require validation.',
                'software_calibrations': {}}
    camera = control = None
    count = 0
    message = 'T starts 30 Hz following; stop moving briefly and press S to save'
    try:
        for role, bus in buses.items():
            source = Path(args.calibration_dir) / f'right_{role}.json'
            raw = source.read_bytes()
            calibration = json.loads(raw)
            manifest['software_calibrations'][role] = {
                'sha256': hashlib.sha256(raw).hexdigest(), 'data': calibration}
            bus.connect()
            bus.set_bus_baudrate(1_000_000)  # Host serial setting only.
            bus.set_calibration(calibration)
        follower, leader = buses['follower'], buses['leader']
        torque = follower.read('Torque_Enable')
        manifest['initial_follower_torque'] = torque.tolist()
        manifest['teleoperation'] = {'requested_fps': 30, 'max_relative_target': 5,
                                     'save_pauses_following': False}
        (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        camera = RGBDCamera(manifest['wrist_serial'])
        params = cv2.aruco.DetectorParameters()
        params.detectInvertedMarker = True
        board = cv2.aruco.CharucoBoard((6, 6), .05, args.marker_length,
                    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        detector = cv2.aruco.CharucoDetector(board, detectorParams=params)
        control = LeaderFollowing(leader, follower)
        control.start()
        start = time.monotonic()
        status_saved = 0.
        print('Window ready: T starts following, S saves without pausing, Q exits.', flush=True)
        while time.monotonic()-start < args.seconds:
            before_t, before_end, before_raw, before = control.snapshot()
            bgr, depth, timing, _ = camera.read()
            after_start, after_t, after_raw, after = control.snapshot()
            cc, ci, mc, mi = detector.detectBoard(bgr)
            corners = 0 if ci is None else len(ci)
            hz, still, delta = control.diagnostics()
            active = control.enabled.is_set()
            if time.monotonic()-status_saved >= 1:
                status = {'host_monotonic_s':time.monotonic(), 'following':active,
                          'measured_control_hz':hz, 'stationary':still,
                          'leader_mismatch_deg':delta, 'saved_poses':count,
                          'charuco_corners':corners}
                status_tmp = root/'live_status.tmp'
                status_tmp.write_text(json.dumps(status,indent=2)+'\n')
                status_tmp.replace(root/'live_status.json')
                status_saved = time.monotonic()
            view = bgr.copy()
            if mi is not None:
                cv2.aruco.drawDetectedMarkers(view, mc, mi)
            if ci is not None:
                cv2.aruco.drawDetectedCornersCharuco(view, cc, ci)
            panel = np.zeros((145, view.shape[1], 3), np.uint8)
            lines = [f'RIGHT ARM: {"LEADER FOLLOWING" if active else "PAUSED"} | saved {count} | corners {corners}/25',
                     f'{hz:.1f} Hz | {"STILL" if still else "MOVING"} | T toggle | S save | Q quit',
                     message[:88], 'Board/base/camera fixed. Torque settings unchanged on exit.']
            for i, line in enumerate(lines):
                cv2.putText(panel, line, (8, 25+32*i), cv2.FONT_HERSHEY_SIMPLEX, .46, (0,255,255), 1)
            cv2.imshow('Right ALOHA ChArUco calibration capture', np.vstack([view,panel]))
            key = cv2.waitKey(1) & 0xff
            if key == ord('q'):
                break
            if key == ord('t'):
                active = control.toggle()
                message = ('Following enabled; S saves without interrupting it' if active
                           else 'Following off; torque settings unchanged')
            if key == ord('s'):
                if corners < 12:
                    message = f'Not saved: only {corners} corners; need at least 12'
                    continue
                if not still or np.max(np.abs(after[:-1]-before[:-1])) > .3:
                    message = 'Not saved: hold the leader still for about one second, then S'
                    continue
                filename = f'pose_{count:03d}'
                np.savez_compressed(root/f'{filename}.npz', bgr=bgr, depth_m=depth.astype(np.float32))
                record = {'index':count, 'timing':timing,
                          'host_joint_bracket_s':[before_t,after_t],
                          'joint_read_intervals_s':[[before_t,before_end],[after_start,after_t]],
                          'teleoperation_hz':hz,
                          'motor_names':follower.motor_names,
                          'raw_ticks_before':before_raw.tolist(), 'raw_ticks_after':after_raw.tolist(),
                          'calibrated_joint_values_before':before.tolist(),
                          'calibrated_joint_values_after':after.tolist(),
                          'charuco_ids':ci.ravel().tolist(), 'charuco_corners':cc.reshape(-1,2).tolist(),
                          'marker_ids':mi.ravel().tolist(), 'marker_corners':[c.reshape(-1,2).tolist() for c in mc],
                          'rgbd_file':f'{filename}.npz', 'transform_validated':False}
                (root/f'{filename}.json').write_text(json.dumps(record,indent=2)+'\n')
                count += 1
                message = f'Saved pose {count}; continue to the next pose'
                print(message, flush=True)
    finally:
        if control is not None:
            control.close()
        if camera is not None:
            camera.close()
        for bus in buses.values():
            if bus.is_connected:
                bus.disconnect()
        cv2.destroyAllWindows()
        print(f'Capture closed; {count} poses saved in {root}. Torque settings unchanged.', flush=True)


if __name__ == '__main__':
    main()
