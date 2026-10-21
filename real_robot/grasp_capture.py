"""Leader-assisted blue-tape grasp teaching with front, side and wrist RGB-D.

The tracker owns the fixed cameras. This process owns both right-arm serial
ports and the right wrist camera. Starts paused; --execute plus T enables
bounded leader following. Never enables torque or changes motor configuration.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import threading
import time

import cv2
import numpy as np

from .aloha_charuco_capture import calibrated_positions, following_target
from .grasp_prior import EVENTS, follower_pose, fresh_observation, tape_frame
from .tracking_config import load_config
from .rgbd_archive import save_rgbd

PROJECT = Path(__file__).resolve().parent.parent
WRIST_SERIAL = '218622278247'
KEY_EVENTS = dict(zip('pgclarv', EVENTS))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def signed(values, bits):
    a = np.asarray(values, np.int64)
    return (a+2**(bits-1)) % 2**bits-2**(bits-1)


class RegisterBlock:
    """Read a verified complete packet without software calibration changes."""
    def __init__(self, bus, address, length):
        import dynamixel_sdk as dxl
        self.dxl, self.address, self.length = dxl, address, length
        self.ids = [bus.motors[name][0] for name in bus.motor_names]
        self.reader = dxl.GroupSyncRead(bus.port_handler, bus.packet_handler, address, length)
        for motor in self.ids:
            if not self.reader.addParam(motor):
                raise RuntimeError('Cannot prepare complete motor read')

    def read(self, fields):
        r = self.reader
        if r.txRxPacket() != self.dxl.COMM_SUCCESS:
            raise RuntimeError('Motor telemetry communication failed')
        if not all(r.isAvailable(i, self.address, self.length) for i in self.ids):
            raise RuntimeError('Incomplete motor telemetry packet')
        return {name: np.array([r.getData(i, address, width) for i in self.ids])
                for name, (address, width) in fields.items()}


class RobotIO:
    def __init__(self, leader, follower):
        self.leader, self.follower = leader, follower
        for bus in (leader, follower):
            if [bus.motors[n][0] for n in bus.motor_names] != list(range(1, 10)):
                raise ValueError('Require the right ALOHA nine-motor ID order')
        self.leader_reader = RegisterBlock(leader, 132, 4)
        self.follower_reader = RegisterBlock(follower, 126, 21)
        self.health_reader = RegisterBlock(follower, 64, 7)
        modes = RegisterBlock(follower, 10, 3).read(
            {'drive': (10, 1), 'mode': (11, 1), 'secondary': (12, 1)})
        if (np.any(modes['drive'] & 8) or np.any(modes['mode'][:8] != 4)
                or modes['mode'][8] != 5 or modes['secondary'][2] != 2
                or modes['secondary'][4] != 4):
            raise RuntimeError('Follower drive/mode/paired-motor setup differs from audited arm')
        models = RegisterBlock(follower, 0, 2).read({'model': (0, 2)})['model']
        if models.tolist() != [1120]*7+[1020]*2:
            raise RuntimeError('Follower motor models differ from audited arm')
        self.provenance = {**{k: v.tolist() for k, v in modes.items()}, 'models': models.tolist(),
            'current_limits_a': (RegisterBlock(follower, 38, 2).read({'limit': (38, 2)})['limit']*.00269).tolist(),
            'goal_currents_a': (signed(RegisterBlock(follower, 102, 2).read({'goal': (102, 2)})['goal'], 16)*.00269).tolist(),
            'initial_goal_raw_ticks': signed(RegisterBlock(follower, 116, 4).read({'goal': (116, 4)})['goal'], 32).tolist()}

    def health(self):
        row = self.health_reader.read({'torque': (64, 1), 'hardware_errors': (70, 1)})
        if np.any(row['hardware_errors']):
            raise RuntimeError(f"Follower hardware fault: {row['hardware_errors'].tolist()}")
        return row

    def read(self):
        leader_raw = signed(self.leader_reader.read({'position': (132, 4)})['position'], 32)
        row = self.follower_reader.read({'raw': (132, 4), 'current': (126, 2),
            'velocity': (128, 4), 'voltage': (144, 2), 'temperature': (146, 1)})
        raw = signed(row['raw'], 32)
        return {'leader_raw_ticks': leader_raw.tolist(), 'follower_raw_ticks': raw.tolist(),
                'leader_calibrated': calibrated_positions(self.leader, leader_raw).tolist(),
                'follower_calibrated': calibrated_positions(self.follower, raw).tolist(),
                'currents_a': (signed(row['current'], 16)*.00269).tolist(),
                'velocities_rad_s': (signed(row['velocity'], 32)*.229*2*np.pi/60).tolist(),
                'voltages_v': (row['voltage']*.1).tolist(),
                'temperatures_c': row['temperature'].tolist()}

    def goal(self, target):
        self.follower.write('Goal_Position', np.asarray(target, np.float32))


class TeachingControl:
    """Serial owner. Logging/camera work stays off the command thread."""
    def __init__(self, io, execute, tcp_offset, fps=30):
        self.io, self.execute, self.tcp_offset, self.fps = io, execute, tcp_offset, fps
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.enabled = False
        self.error = None
        self.latest = None
        self.history = deque(maxlen=40)
        self.pending = queue.Queue(maxsize=256)
        self.heartbeat = time.monotonic()
        self.thread = threading.Thread(target=self._run, name='grasp-teleoperation', daemon=True)

    def start(self):
        self.thread.start()

    def pulse(self):
        with self.lock:
            self.heartbeat = time.monotonic()
            if self.error:
                raise RuntimeError(self.error)

    def _run(self):
        health = None
        last_health = -float('inf')
        last_cycle = time.monotonic()
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                with self.lock:
                    if self.stop.is_set():
                        break
                    if started-self.heartbeat > .5:
                        raise RuntimeError('Recorder heartbeat lost; leader following stopped')
                    if started-last_cycle > .20:
                        raise RuntimeError('Teleoperation interval exceeded 200 ms')
                    last_cycle = started
                    if started-last_health >= .25:
                        health = self.io.health()
                        last_health = time.monotonic()
                    row = self.io.read()
                    read_end = time.monotonic()
                    if read_end-started > .15:
                        raise RuntimeError('Motor feedback read exceeded 150 ms')
                    if max(row['temperatures_c']) >= 55:
                        raise RuntimeError('Follower temperature guard')
                    if self.enabled and not np.all(health['torque'] == 1):
                        raise RuntimeError('Follower torque lost; leader following stopped')
                    actual = np.array(row['follower_calibrated'])
                    leader = np.array(row['leader_calibrated'])
                    q, pose = follower_pose(row['follower_raw_ticks'], self.tcp_offset)
                    row.update(host_read_start_monotonic_s=started, host_read_end_monotonic_s=read_end,
                        q_model_rad=q.tolist(), T_base_tcp_candidate=pose.tolist(),
                        health_host_monotonic_s=last_health,
                        torque=health['torque'].tolist(), hardware_errors=health['hardware_errors'].tolist(),
                        following=self.enabled, commanded_calibrated=None)
                    if self.enabled:
                        target = following_target(leader, actual)
                        self.io.goal(target)
                        row['commanded_calibrated'] = target.tolist()
                    row['host_cycle_end_monotonic_s'] = time.monotonic()
                    self.latest = row
                    self.history.append(row)
                    # Stop on overflow rather than silently losing action/feedback records.
                    self.pending.put_nowait(row)
                self.stop.wait(max(0, 1/self.fps-(time.monotonic()-started)))
        except BaseException as exc:
            with self.lock:
                self.error = f'{type(exc).__name__}: {exc}'
                self.enabled = False
            self.stop.set()

    def snapshot(self):
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            row = self.latest
            if row is None or time.monotonic()-row['host_read_end_monotonic_s'] > .15:
                raise ValueError('Waiting for fresh robot feedback')
            return dict(row)

    def stationary(self):
        with self.lock:
            recent = [r for r in self.history if time.monotonic()-r['host_read_end_monotonic_s'] <= .8]
            if len(recent) < 8 or recent[-1]['host_read_end_monotonic_s']-recent[0]['host_read_end_monotonic_s'] < .5:
                return False
            # Include the gripper; a teaching event should not straddle closure.
            return bool(np.max(np.ptp([r['follower_calibrated'] for r in recent], axis=0)) <= .5)

    def toggle(self):
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            if self.enabled:
                self.enabled = False
                return False
            if not self.execute:
                raise ValueError('Preview only: restart with --execute for leader following')
            if self.stop.is_set():
                raise ValueError('Teaching controller has stopped')
            if self.latest is None or time.monotonic()-self.latest['host_read_end_monotonic_s'] > .15:
                raise ValueError('Waiting for robot feedback')
            if not np.all(self.io.health()['torque'] == 1):
                raise ValueError('All nine follower motors must already have holding torque')
            if np.max(np.abs(np.array(self.latest['leader_calibrated'])-
                                self.latest['follower_calibrated'])) > 5:
                raise ValueError('Match leader to follower within 5 degrees / gripper percentage points')
            self.enabled = True
            self.heartbeat = time.monotonic()
            return True

    def close(self):
        with self.lock:
            self.enabled = False
            self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('Robot thread still owns serial ports')


class WristRecording:
    """Save wrist RGB-D and native timing independently of the robot loop."""
    def __init__(self, root, serial=WRIST_SERIAL):
        self.root, self.serial = Path(root), serial
        self.root.mkdir()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.records = deque(maxlen=60)
        self.preview = None
        self.error = None
        self.thread = threading.Thread(target=self._run, name='grasp-wrist', daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        from .camera import RGBDCamera
        camera = None
        try:
            camera = RGBDCamera(self.serial)
            previous = None
            domain = None
            with (self.root/'frames.jsonl').open('x', buffering=1) as log:
                index = 0
                while not self.stop.is_set():
                    bgr, depth, timing, _ = camera.read()
                    stamp = timing['sensor_timestamp_s']
                    if ((previous is not None and stamp <= previous)
                            or (domain is not None and timing['timestamp_domain'] != domain)):
                        raise RuntimeError('Wrist camera clock repeated or changed domain')
                    previous, domain = stamp, timing['timestamp_domain']
                    name = f'frame_{index:07d}.npz'
                    save_rgbd(self.root/name, bgr, depth)
                    row = {'index': index, 'file': 'wrist/'+name, 'timing': timing}
                    log.write(json.dumps(row, allow_nan=False)+'\n')
                    with self.lock:
                        self.records.append(row)
                        self.preview = bgr
                    index += 1
        except BaseException as exc:
            with self.lock:
                self.error = f'{type(exc).__name__}: {exc}'
        finally:
            if camera is not None:
                camera.close()

    def snapshot(self):
        with self.lock:
            if self.error:
                raise RuntimeError('Wrist recording failed: '+self.error)
            return list(self.records), self.preview

    def close(self):
        self.stop.set()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError('Wrist camera thread did not stop')


def event_snapshot(sample, observation, wrist_records, observation_file, now):
    """Preserve honest timing and file links, without claiming exposure sync."""
    rgbd = observation.get('rgbd') or {}
    if rgbd.get('save_every') != 1 or set(rgbd.get('files', {})) != {'front', 'side'}:
        raise ValueError('Restart perception with --record-rgbd')
    views = rgbd.get('views', {})
    if set(views) != {'front', 'side'}:
        raise ValueError('Waiting for recorded front/side frames')
    for role, view in views.items():
        age = now-view['timing']['host_arrival_monotonic_s']
        if not 0 <= age <= .4:
            raise ValueError(f'{role} RGB-D frame stale')
        if not (Path(observation_file).parent/rgbd['files'][role]).is_file():
            raise ValueError(f'Waiting for {role} RGB-D file to finish saving; retry marker')
    if not wrist_records:
        raise ValueError('Waiting for saved wrist RGB-D')
    wrist = min(wrist_records, key=lambda r: abs(r['timing']['host_arrival_monotonic_s']-
                                                sample['host_read_end_monotonic_s']))
    if not 0 <= now-wrist['timing']['host_arrival_monotonic_s'] <= .3:
        raise ValueError('Wrist RGB-D frame stale')
    timings = [views[r]['timing'] for r in ('front', 'side')]+[wrist['timing']]
    domains = {t['timestamp_domain'] for t in timings}
    common_clock = len(domains) == 1 and any(s in next(iter(domains)) for s in ('global_time', 'system_time'))
    stamps = [t[k] for t in timings for k in ('sensor_timestamp_s', 'color_timestamp_s')]
    span = max(stamps)-min(stamps) if common_clock else None
    if span is not None and span > .15:
        raise ValueError('Three-camera timestamp span exceeds 150 ms; hold still and retry')
    return {'sample': sample, 'observation': observation, 'front_side_rgbd': rgbd,
            'front_side_session': str(Path(observation_file).parent.resolve()), 'wrist_rgbd': wrist,
            'host_event_monotonic_s': now, 'three_camera_native_span_s': span,
            'cross_camera_clock_comparable': common_clock,
            'wrist_minus_joint_host_s': wrist['timing']['host_arrival_monotonic_s']-
                                         sample['host_read_end_monotonic_s'],
            'timing_note': 'Nearest saved frames and bounded read intervals; not hardware exposure synchronization.'}


def observation_path(config, override=None):
    if override:
        return Path(override).resolve()
    root = Path(config['perception']['output_root'])
    if not root.is_absolute():
        root = PROJECT/root
    pointer = json.loads((root/'latest-perception.json').read_text())
    if pointer['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise ValueError('Perception session is from another boot')
    return Path(pointer['observation_file']).resolve()


def connect_buses(calibration_dir):
    # Direct bus connect/disconnect only opens/closes serial. Robot.connect()
    # would toggle torque and must not be used for a handoff from an active hold.
    from lerobot.common.robot_devices.robots.configs import AlohaRobotConfig
    from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
    config = AlohaRobotConfig()
    configs = {'leader': config.leader_arms['right'], 'follower': config.follower_arms['right']}
    buses, provenance = {}, {}
    try:
        for role, cfg in configs.items():
            result = subprocess.run(['fuser', cfg.port], capture_output=True, text=True)
            if result.returncode == 0:
                raise RuntimeError(f'{cfg.port} already owned: {result.stdout.strip()}')
            if result.returncode != 1:
                raise RuntimeError(f'Cannot check serial owner for {cfg.port}: {result.stderr.strip()}')
        for role, cfg in configs.items():
            source = Path(calibration_dir)/f'right_{role}.json'
            raw = source.read_bytes()
            calibration = json.loads(raw)
            bus = DynamixelMotorsBus(cfg)
            buses[role] = bus
            bus.connect()
            bus.set_bus_baudrate(1_000_000)
            if calibration['motor_names'] != bus.motor_names:
                raise ValueError('Calibration motor order does not match bus')
            bus.set_calibration(calibration)
            provenance[role] = {'file': str(source.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
                                'calibration': calibration, 'port': cfg.port,
                                'motors': cfg.motors}
        return buses, provenance
    except BaseException:
        for bus in buses.values():
            if bus.is_connected:
                bus.disconnect()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(PROJECT/'real_robot/config.yaml'))
    parser.add_argument('--observation', help='Default: latest perception session from this boot')
    parser.add_argument('--output', help='New session directory')
    parser.add_argument('--execute', action='store_true', help='Allow T to enable leader following')
    parser.add_argument('--heading-deg', type=float, required=True,
                        help='Known rolling direction: 0=base +X, 90=base +Y; stationary tape axle perpendicular')
    parser.add_argument('--tcp-offset-m', type=float, nargs=3, default=[.16, 0, 0])
    parser.add_argument('--wrist-serial', default=WRIST_SERIAL)
    parser.add_argument('--calibration-dir', default='/home/ailab/lerobot/.cache/calibration/aloha_default')
    parser.add_argument('--seconds', type=float, default=1800)
    parser.add_argument('--perception-wait-s', type=float, default=15,
                        help='Wait for a fresh associated tape observation before opening devices')
    args = parser.parse_args(argv)
    if (not np.isfinite([args.heading_deg, *args.tcp_offset_m, args.seconds, args.perception_wait_s]).all()
            or not 0 <= args.perception_wait_s <= 60
            or not 0 < args.seconds <= 3600 or not .05 <= np.linalg.norm(args.tcp_offset_m) <= .25):
        parser.error('Invalid heading, TCP offset, or duration')
    config = load_config(args.config)
    try:
        observation_file = observation_path(config, args.observation)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(f'Start perception with --record-rgbd first: {exc}')
    rigpath = Path(config['perception']['rig'])
    if not rigpath.is_absolute():
        rigpath = PROJECT/rigpath
    rigraw = rigpath.read_bytes()
    rig_sha = hashlib.sha256(rigraw).hexdigest()
    deadline = time.monotonic()+args.perception_wait_s
    waiting = False
    while True:
        try:
            initial = json.loads(observation_file.read_text())
            fresh_observation(initial, rig_sha, time.monotonic())
            break
        except (OSError, ValueError) as exc:
            if time.monotonic() >= deadline:
                parser.error(f'Cannot start grasp capture: {exc}. Observation: {observation_file}. '
                             'Restart perception with --record-rgbd, select the tape in both views, '
                             'and keep that process running.')
            if not waiting:
                print(f'Waiting for fresh tape perception (up to {args.perception_wait_s:g}s): {exc}', flush=True)
                waiting = True
            time.sleep(.05)
    if (initial.get('rgbd') or {}).get('save_every') != 1:
        parser.error('Restart perception with --record-rgbd before grasp capture')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    root = Path(args.output) if args.output else PROJECT/'scratch/real_robot/grasps'/('blue-tape-'+stamp)
    root.mkdir(parents=True, exist_ok=False)
    (root/'episodes').mkdir()
    manifest = {'schema': 'blue_tape_grasp_capture_v1', 'object': 'blue_tape',
        'posture': 'upright_on_curved_edge', 'mode': 'execute' if args.execute else 'preview',
        'created_utc': stamp, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'rig_sha256': rig_sha, 'rig': json.loads(rigraw), 'tcp_offset_m': args.tcp_offset_m,
        'initial_heading_deg': args.heading_deg, 'wrist_serial': args.wrist_serial,
        'wrist_handeye_validated': False, 'wrist_use': 'Recorded optical-frame RGB-D and visual grasp evidence; no wrist-to-base fusion.',
        'observation_file': str(observation_file), 'front_side_session_is_required_dependency': True,
        'joint_units': 'LeRobot calibrated arm degrees; gripper percent. Model angles separately in radians.',
        'teleoperation': {'hz': 30, 'max_relative_target': 5, 'enable_match_bound': 5},
        'torque_writes': False, 'configuration_writes': False, 'validated_for_execution': False,
        'timestamp_note': 'Native camera timestamps and host joint read intervals; cameras have independent exposures.',
        'config': Path(args.config).read_text()}
    write_json(root/'manifest.json', manifest)
    wrist = WristRecording(root/'wrist', args.wrist_serial)
    control = None
    buses = {}
    episode = None
    box = 'box_1'
    heading = args.heading_deg
    count = 0
    error = None
    message = 'Waiting for wrist camera. T toggles leader following.'
    logs = []
    try:
        wrist.start()
        # Wait for the wrist before acquiring serial ports, without enabling motion.
        deadline = time.monotonic()+30
        while True:
            records, _ = wrist.snapshot()
            if records:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('No wrist frames within 30 seconds')
            time.sleep(.05)
        buses, provenance = connect_buses(args.calibration_dir)
        manifest['buses'] = provenance
        write_json(root/'manifest.json', manifest)
        io = RobotIO(buses['leader'], buses['follower'])
        manifest['hardware_readback'] = io.provenance
        write_json(root/'manifest.json', manifest)
        control = TeachingControl(io, args.execute, args.tcp_offset_m)
        control.start()
        feedback_log = (root/'robot.jsonl').open('x', buffering=1)
        observations_log = (root/'observations.jsonl').open('x', buffering=1)
        events_log = (root/'events.jsonl').open('x', buffering=1)
        logs = [feedback_log, observations_log, events_log]
        last_observation = None
        last_preview = 0
        last_status = 0
        started = time.monotonic()
        print(f'Grasp session: {root}\nStarts PAUSED. T enables following only with --execute. Q stops new goals.', flush=True)
        while time.monotonic()-started < args.seconds:
            control.pulse()
            records, view = wrist.snapshot()
            if not records or time.monotonic()-records[-1]['timing']['host_arrival_monotonic_s'] > .5:
                raise RuntimeError('Wrist frame feed stale; teaching stopped')
            while True:
                try:
                    row = control.pending.get_nowait()
                except queue.Empty:
                    break
                feedback_log.write(json.dumps(row, allow_nan=False)+'\n')
            try:
                observation = json.loads(observation_file.read_text())
            except (OSError, json.JSONDecodeError):
                observation = {}
            rgbd_views = (observation.get('rgbd') or {}).get('views', {})
            if (not observation.get('running') or set(rgbd_views) != {'front', 'side'}
                    or any(not 0 <= time.monotonic()-v['timing']['host_arrival_monotonic_s'] <= 1.
                           for v in rgbd_views.values())):
                raise RuntimeError('Front/side recording stopped or frame feed stale; teaching stopped')
            version = observation.get('host_published_monotonic_s')
            if version is not None and version != last_observation:
                observations_log.write(json.dumps(observation, allow_nan=False)+'\n')
                last_observation = version
            now = time.monotonic()
            if now-last_status >= .5:
                with control.lock:
                    following = control.enabled
                write_json(root/'status.json', {'running': True, 'following': following,
                    'box_id': box, 'heading_deg': heading, 'episode_id': None if episode is None else episode['episode_id'],
                    'events_saved': [] if episode is None else list(episode['events']),
                    'message': message, 'host_monotonic_s': now})
                last_status = now
            if view is not None and now-last_preview >= .1:
                panel = cv2.resize(view, (640, 480))
                footer = np.full((190, 640, 3), 24, np.uint8)
                with control.lock:
                    following = control.enabled
                next_event = EVENTS[len(episode['events'])] if episode else 'N: new episode / freeze tape reference'
                labels = [f"RIGHT WRIST | {'FOLLOWING LEADER' if following else 'PAUSED'} | {box} | heading {heading:g} deg",
                    'T: following | N: start | 1/2: box | [/]: heading -/+15',
                    'P: pregrasp | G: before close | C: closed | L: lifted',
                    'A: above box | R: release (open) | V: placed/success',
                    'X: failed episode | Q: stop; existing torque retained',
                    'Next: '+next_event, message[:85]]
                for i, label in enumerate(labels):
                    cv2.putText(footer, label, (8, 23+25*i), cv2.FONT_HERSHEY_SIMPLEX, .43, (0,255,255), 1)
                cv2.imshow('Blue tape grasp teaching - wrist', np.vstack([panel, footer]))
                last_preview = now
            key = cv2.waitKey(1) & 0xff
            if key == ord('q') or (root/'STOP').exists():
                break
            if key == 255:
                continue
            try:
                if key == ord('t'):
                    message = 'Following enabled' if control.toggle() else 'Following paused; retained goals'
                elif key in (ord('1'), ord('2'), ord('['), ord(']')):
                    if episode:
                        raise ValueError('Finish or fail this episode before changing box/heading')
                    if key in (ord('1'), ord('2')):
                        box = 'box_'+chr(key)
                    else:
                        heading += -15 if key == ord('[') else 15
                    message = f'{box}, rolling heading {heading:g} deg'
                elif key == ord('n'):
                    if episode:
                        raise ValueError('An episode is already open')
                    if not control.stationary():
                        raise ValueError('Hold leader, follower and tape still before N')
                    point = fresh_observation(observation, rig_sha, time.monotonic())
                    if set(observation['fusion']['sources']) != {'front', 'side'}:
                        missing = {'front', 'side'}-set(observation['fusion']['sources'])
                        details = []
                        for role in sorted(missing):
                            quality = ((observation.get('rgbd') or {}).get('views', {}).get(role, {}).get('quality') or {})
                            detail = role+': '+str(quality.get('reason') or 'no accepted 3D point')
                            median = quality.get('nonzero_depth_median_m')
                            bounds = quality.get('accepted_depth_range_m')
                            if median is not None and bounds is not None:
                                detail += f' (depth {median:.2f} m; allowed {bounds[0]:.2f}..{bounds[1]:.2f} m)'
                            details.append(detail)
                        raise ValueError('Need two accepted 3D views; '+'; '.join(details))
                    speed = (observation.get('velocity_report') or {}).get('speed_mps')
                    if speed is None or not np.isfinite(speed) or speed > .02:
                        raise ValueError('Wait for a stable tape speed estimate below 0.02 m/s')
                    anchor = event_snapshot(control.snapshot(), observation, records, observation_file, time.monotonic())
                    episode = {'episode_id': count, 'box_id': box, 'heading_deg': heading,
                        'T_base_tape_reference': tape_frame(point, heading).tolist(),
                        'anchor': anchor, 'events': {}, 'outcome': 'incomplete',
                        'reference_note': 'Frozen pre-contact visible-surface proxy; tape must remain stationary until closure.'}
                    count += 1
                    write_json(root/'episodes'/f"episode_{episode['episode_id']:04d}.json", episode)
                    message = 'Reference frozen. P: hold pregrasp pose above stationary tape.'
                elif key == ord('x'):
                    if episode is None:
                        raise ValueError('No open episode')
                    episode['outcome'] = 'failed'
                    write_json(root/'episodes'/f"episode_{episode['episode_id']:04d}.json", episode)
                    events_log.write(json.dumps({'episode_id': episode['episode_id'], 'event': 'failed',
                                                'host_monotonic_s': time.monotonic()})+'\n')
                    episode = None
                    message = 'Failed episode saved. Reset manually, then N.'
                elif chr(key) in KEY_EVENTS:
                    if episode is None:
                        raise ValueError('Press N to start an episode first')
                    name = KEY_EVENTS[chr(key)]
                    if name != EVENTS[len(episode['events'])]:
                        raise ValueError('Expected '+EVENTS[len(episode['events'])])
                    if not control.stationary():
                        raise ValueError('Hold the follower and gripper still for 0.5 s before marking')
                    event = event_snapshot(control.snapshot(), observation, records, observation_file, time.monotonic())
                    episode['events'][name] = event
                    events_log.write(json.dumps({'episode_id': episode['episode_id'], 'event': name, **event}, allow_nan=False)+'\n')
                    if name == 'placed':
                        episode['outcome'] = 'success'
                    write_json(root/'episodes'/f"episode_{episode['episode_id']:04d}.json", episode)
                    if name == 'placed':
                        episode = None
                        message = 'Operator-confirmed success saved. Reset manually, then N.'
                    else:
                        message = name+' saved'
            except ValueError as exc:
                message = str(exc)
                print('Not saved:', message, flush=True)
    except KeyboardInterrupt:
        pass
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        print(error, flush=True)
    finally:
        cleanup_errors = []
        robot_stopped = True
        if control is not None:
            try:
                control.close()
                if control.error and error is None:
                    error = control.error
            except BaseException as exc:
                robot_stopped = False
                cleanup_errors.append(str(exc))
            if logs:
                while not control.pending.empty():
                    row = control.pending.get_nowait()
                    try:
                        logs[0].write(json.dumps(row, allow_nan=False)+'\n')
                    except BaseException as exc:
                        cleanup_errors.append('Final feedback log: '+str(exc))
                        break
        try:
            wrist.close()
        except BaseException as exc:
            cleanup_errors.append(str(exc))
        if robot_stopped:
            for bus in buses.values():
                try:
                    if bus.is_connected:
                        bus.disconnect()
                except BaseException as exc:
                    cleanup_errors.append(str(exc))
        for stream in logs:
            try:
                stream.close()
            except BaseException as exc:
                cleanup_errors.append('Log close: '+str(exc))
        try:
            cv2.destroyAllWindows()
        except BaseException as exc:
            cleanup_errors.append(str(exc))
        try:
            write_json(root/'summary.json', {'running': False, 'error': error, 'cleanup_errors': cleanup_errors,
                'episodes_started': count, 'torque_changed': False,
                'stop_behavior': 'No further goals. Existing goals retained; hardware protection can independently disable torque.'})
            write_json(root/'status.json', {'running': False, 'following': False, 'error': error,
                                            'cleanup_errors': cleanup_errors})
        except BaseException as exc:
            cleanup_errors.append('Final status write: '+str(exc))
        for failure in cleanup_errors:
            print('Cleanup:', failure, flush=True)
    return 1 if error or cleanup_errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
