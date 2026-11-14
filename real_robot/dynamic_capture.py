"""Right leader/follower dynamic demonstrations; separate from grasp teaching."""
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np

from .collection_audio import AudioCues
from .recording_lease import boot_id, renew, LeaseOwner
from .static_object import SCHEMA as STATIC_SCHEMA,AUTOMATIC_PHASE_METHOD,lock_position,observe_motion,validate_lock

PROJECT = Path(__file__).resolve().parent.parent


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


class Episodes:
    """Persistent attempt ledger; markers never require a stationary object."""
    def __init__(self, root, task_id, split):
        self.root, self.task_id, self.split = Path(root), task_id, split
        (self.root / 'episodes').mkdir()
        self.current = None
        self.count = 0

    def save(self):
        write_json(self.root / 'episodes' / f"episode_{self.current['episode_id']:06d}.json", self.current)

    def begin(self, now, static_object=None):
        if self.current is not None:
            raise ValueError('An attempt is already open')
        if static_object is not None:validate_lock(static_object,now)
        self.current = dict(episode_id=self.count, task_id=self.task_id, split=self.split,
                            start_monotonic_s=now, end_monotonic_s=None,
                            outcome='incomplete', events={})
        if static_object is not None:self.current['static_object']=static_object
        self.count += 1
        self.save()

    def event(self, name, now):
        if self.current is None:
            raise ValueError('Start an episode before marking an event')
        if name in self.current['events']:
            raise ValueError(f'{name} already marked')
        self.current['events'][name] = now
        self.save()

    def finish(self, outcome, now, reason=None):
        if self.current is None:
            raise ValueError('No open attempt; press Space to start a trial in default-success mode')
        self.current.update(outcome=outcome, end_monotonic_s=now, reason=reason)
        self.save()
        with (self.root / 'attempts.jsonl').open('a') as stream:
            stream.write(json.dumps(self.current, allow_nan=False) + '\n')
        self.current = None


from .wrist_capture import EpisodeWrist


def copy_fixed(observation, source, root):
    """Make recordings self-contained; source archive writes are atomic."""
    rgbd = observation.get('rgbd') or {}
    if set(rgbd.get('files', {})) != {'front', 'side'}:
        return None
    result = dict(observation)
    result['rgbd'] = dict(rgbd)
    result['rgbd']['files'] = {}
    for role, name in rgbd['files'].items():
        src = (source / name).resolve()
        if src.parent != source.resolve() or not src.is_file():
            return None
        relative = f'fixed/{role}/{src.name}'
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            try:
                os.link(src, dest)
            except OSError:
                temp = dest.with_suffix('.tmp')
                shutil.copyfile(src, temp)
                temp.replace(dest)
        result['rgbd']['files'][role] = relative
    result['host_recorded_monotonic_s'] = observation.get('host_received_monotonic_s',time.monotonic())
    result['archive_ready_monotonic_s']=time.monotonic()
    return result


class FixedRecordingBuffer:
    """Do not discard snapshots whose asynchronous RGB-D files are still writing."""
    def __init__(self,source,root,stream):
        self.source,self.root,self.stream=Path(source),Path(root),stream
        self.pending=deque()
        self.version=None

    def receive(self,observation,now):
        version=observation.get('host_published_monotonic_s')
        files=(observation.get('rgbd') or {}).get('files',{})
        if version==self.version or set(files)!={'front','side'}:
            return
        if len(self.pending)>=256:
            raise RuntimeError('Fixed RGB-D publication backlog exceeded 256 observations')
        self.version=version
        self.pending.append({**observation,'host_received_monotonic_s':now})

    def drain(self):
        count=0
        while self.pending and count<16:
            saved=copy_fixed(self.pending[0],self.source,self.root)
            if saved is None:
                break
            self.stream.write(json.dumps(saved,allow_nan=False)+'\n')
            self.pending.popleft()
            count+=1
        return count

    def finish(self,timeout=3.):
        deadline=time.monotonic()+timeout
        while self.pending and time.monotonic()<deadline:
            if not self.drain():
                time.sleep(.01)
        if self.pending:
            raise RuntimeError(f'{len(self.pending)} fixed RGB-D observations did not finish saving')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(PROJECT / 'real_robot/config.yaml'))
    parser.add_argument('--plan', default=str(PROJECT / 'real_robot/collection_plan.json'))
    parser.add_argument('--task', required=True)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], required=True)
    parser.add_argument('--output', type=Path, required=True, help='New dedicated session directory')
    parser.add_argument('--observation')
    parser.add_argument('--recording-lease', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='T may enable matched leader following')
    parser.add_argument('--heading-deg', type=float, default=0, help='Assumed initial object yaw, not measured spin')
    parser.add_argument('--object-instance', help='Physical object instance ID (defaults to task ID)')
    parser.add_argument('--box-id', default='box_2')
    parser.add_argument('--tcp-offset-m', type=float, nargs=3, default=[.16, 0, 0])
    parser.add_argument('--gripper-open-percent', type=float, default=100)
    parser.add_argument('--gripper-closed-percent', type=float, default=0)
    parser.add_argument('--wrist-serial', default='218622278247')
    parser.add_argument('--camera-fps', type=int, choices=(15,30), default=30)
    parser.add_argument('--viewer', choices=['opencv', 'rerun'], default='opencv')
    parser.add_argument('--keyboard-source', choices=['terminal','global'], default='terminal')
    parser.add_argument('--start-delay',type=float,default=1.,help='Countdown seconds before opening a default-success episode')
    parser.add_argument('--no-audio',action='store_true',help='Disable episode start/end sounds')
    parser.add_argument('--default-success', action='store_true', help='Space starts an episode; Enter stops/saves success; X stops/saves failure')
    parser.add_argument('--static-object',action='store_true',help='Stationary approach; per-attempt position lock; automatic offline lift/release labels')
    from .relative_target import parse_relative_target
    parser.add_argument('--robot-max-relative-target', '--robot.max_relative_target', type=parse_relative_target, default=5.,help='Positive scalar or JSON list of nine limits; gripper last')
    parser.add_argument('--clutched', action='store_true', help='Relative leader motion; start/T establishes holding torque if all motors are off')
    parser.add_argument('--max-successes',type=int,default=0,help='Stop after this many operator successes; 0 is unlimited')
    parser.add_argument('--calibration-dir', default='/home/ailab/lerobot/.cache/calibration/aloha_default')
    parser.add_argument('--seconds', type=float, default=1800)
    args = parser.parse_args(argv)
    if (not np.isfinite(args.start_delay) or not 0 <= args.start_delay <= 10
            or args.max_successes<0 or not np.isfinite([args.seconds, args.heading_deg, *args.tcp_offset_m,
                         args.gripper_open_percent, args.gripper_closed_percent]).all()
            or not 0 < args.seconds <= 3600 or not .05 <= np.linalg.norm(args.tcp_offset_m) <= .25
            or not 0 <= args.gripper_closed_percent <= 100
            or not 0 <= args.gripper_open_percent <= 100
            or abs(args.gripper_open_percent - args.gripper_closed_percent) < 10):
        parser.error('Invalid duration, TCP offset, heading or gripper endpoints')
    plan = json.loads(Path(args.plan).read_text())
    tasks = {row['id']: row for row in plan['tasks']}
    if args.task not in tasks:
        parser.error('Task must be one of ' + ', '.join(tasks))
    # Hardware imports and serial ownership are shared, but stationary capture is unchanged.
    from .grasp_capture import RobotIO, TeachingControl, connect_buses, observation_path
    from .tracking_config import load_config
    config = load_config(args.config)
    observation_file = observation_path(config, args.observation)
    initial = json.loads(observation_file.read_text())
    if not initial.get('running') or time.monotonic() - initial.get('host_published_monotonic_s', 0) > .5:
        parser.error('Start live perception first')
    if Path(initial.get('recording_lease') or '').resolve() != args.recording_lease.resolve():
        parser.error('Perception must use the same --recording-lease')
    if initial.get('camera_fps') != args.camera_fps:
        parser.error('Perception and wrist recording must use the same --camera-fps')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    rig_path = Path(config['perception']['rig'])
    if not rig_path.is_absolute():
        rig_path = PROJECT / rig_path
    rig_raw = rig_path.read_bytes()
    if initial.get('rig_sha256') != hashlib.sha256(rig_raw).hexdigest():
        parser.error('Live perception and configured calibration differ')
    if initial.get('output_frame') != 'right_base_estimate':
        parser.error('Perception must publish right_base_estimate')
    packages = {}
    for name in ('numpy', 'scipy', 'pyrealsense2', 'opencv-contrib-python', 'pyarrow', 'av'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest = dict(schema='dom_dynamic_capture_v1', session_id=root.name, boot_id=boot_id(),
                    task_id=args.task, task=tasks[args.task], split=args.split,
                    object_instance=args.object_instance or args.task,
                    box_id=None if tasks[args.task].get('task_type')=='pick' else args.box_id,
                    collector='human_leader', mode='execute' if args.execute else 'preview',
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    tcp_offset_m=args.tcp_offset_m, heading_deg=args.heading_deg,
                    gripper_open_percent=args.gripper_open_percent,
                    gripper_closed_percent=args.gripper_closed_percent,
                    wrist_serial=args.wrist_serial, packages=packages,
                    rig_sha256=hashlib.sha256(rig_raw).hexdigest(), rig=json.loads(rig_raw),
                    config=Path(args.config).read_text(), observation_file=str(observation_file),
                    recording_lease=str(args.recording_lease.resolve()),
                    state_quality='estimated; surface proxy; object orientation assumed',
                    native_control_hz=30, native_camera_hz=args.camera_fps, training_fps=25,
                    torque_writes='explicit episode-start/T hold-enable only' if args.clutched else False,
                    teleoperation_mapping='clutched_relative' if args.clutched else 'absolute',
                    state_semantics='FK of measured follower raw encoders',
                    action_semantics='FK of commanded follower joints, not measured next-state pose',
                    execution_backend='leader_follower_joint_position',
                    kinematics_validated=False, max_relative_target=args.robot_max_relative_target,
                    viewer=args.viewer, keyboard_source=args.keyboard_source,
                    start_delay_s=args.start_delay if args.default_success else 0,
                    audio_cues=args.default_success and not args.no_audio, episode_completion='space_start_enter_stop' if args.default_success else 'explicit_success')
    if args.static_object:
        manifest['static_object_collection']=dict(schema=STATIC_SCHEMA,velocity='zero_during_declared_stationary_approach',
            position='fresh_measurement_locked_per_episode',continuous_tracking_required=False,
            motion_monitor='invalidate on fresh measured displacement >20mm; hidden motion requires operator X',
            phase_labeling=AUTOMATIC_PHASE_METHOD,
            lift_confirmation='offline measured closure + sustained TCP lift + operator episode success; not manual visual review',
            release_label='first gripper reopening trend after supported lift; L/R optional')
    manifest['git_revision'] = subprocess.check_output(['git','rev-parse','HEAD'], cwd=PROJECT, text=True).strip()
    manifest['source_sha256'] = {name:hashlib.sha256((PROJECT/'real_robot'/name).read_bytes()).hexdigest()
                                for name in ('dynamic_capture.py','grasp_capture.py','dom_dataset.py',
                                             'vx300s_kinematics.py','recording_lease.py','collection_control.py')}
    for name in ('collection_audio.py','episode_feedback.py','collection_preview.py'):
        manifest['source_sha256'][name]=hashlib.sha256((PROJECT/'real_robot'/name).read_bytes()).hexdigest()
    manifest['source_sha256']['wrist_capture.py']=hashlib.sha256((PROJECT/'real_robot/wrist_capture.py').read_bytes()).hexdigest()
    manifest['source_sha256']['rgbd_archive.py']=hashlib.sha256((PROJECT/'real_robot/rgbd_archive.py').read_bytes()).hexdigest()
    manifest['rgbd_compression']='lossless NPZ ZIP_DEFLATED level 1'
    if args.static_object:manifest['source_sha256']['static_object.py']=hashlib.sha256((PROJECT/'real_robot/static_object.py').read_bytes()).hexdigest()
    write_json(root / 'manifest.json', manifest)
    episodes = Episodes(root, args.task, args.split)
    from .episode_feedback import EpisodeFeedback
    feedback = EpisodeFeedback(default_success=args.default_success)
    audio = None
    pending_start = None

    def finish_episode(outcome, now, reason=None):
        episode_id = episodes.current['episode_id'] if episodes.current is not None else None
        episodes.finish(outcome, now, reason)
        feedback.saved(episode_id, outcome)
        if audio is not None:audio.play('end' if outcome=='success' else 'failed')

    def begin_episode(observation):
        now=time.monotonic()
        context=lock_position(observation,now) if args.static_object else None
        episodes.begin(now,context)

    wrist = EpisodeWrist(root / 'wrist', args.wrist_serial, fps=args.camera_fps)
    buses, control, logs = {}, None, []
    error, cleanup_errors, robot_stopped = None, [], True
    lease_owner = None
    fixed_buffer=None
    preview=None
    try:
        lease_owner = LeaseOwner(args.recording_lease)
        renew(args.recording_lease, False, time.monotonic())
        if args.viewer == 'rerun':
            from .collection_preview import RerunPreview
            preview = RerunPreview(observation_file.parent, str(root), keyboard_source=args.keyboard_source)
        if args.default_success and not args.no_audio:audio=AudioCues()
        wrist.thread.start()
        deadline = time.monotonic() + 30
        while wrist.snapshot()[0] is None:
            if time.monotonic() > deadline:
                raise RuntimeError('No wrist preview within 30 seconds')
            time.sleep(.05)
        buses, provenance = connect_buses(args.calibration_dir)
        io = RobotIO(buses['leader'], buses['follower'])
        manifest.update(buses=provenance, hardware_readback=io.provenance)
        write_json(root / 'manifest.json', manifest)
        if args.clutched:
            from .collection_control import CollectionControl
            control = CollectionControl(io, args.execute, args.tcp_offset_m, max_relative_target=args.robot_max_relative_target)
        else:
            control = TeachingControl(io, args.execute, args.tcp_offset_m, max_relative_target=args.robot_max_relative_target)
        robot_log = (root / 'robot.jsonl').open('x', buffering=1)
        obs_log = (root / 'observations.jsonl').open('x', buffering=1)
        logs = [robot_log, obs_log]
        fixed_buffer=FixedRecordingBuffer(observation_file.parent,root,obs_log)
        control.start()
        successes=0
        wrist_gap=False
        fixed_gap=False
        started = time.monotonic()
        message = ('Space starts and clutches at the current follower pose' if args.default_success else 'N starts and clutches at the current follower pose') if args.clutched else 'T: match and enable leader; N: begin before rolling'
        if args.viewer == 'opencv': cv2.namedWindow('Dynamic DOM capture')
        print('Starts PAUSED. SPACE starts; ENTER stops/saves success; X stops/saves failure; Q finishes session; T pauses.' if args.default_success else 'Starts PAUSED. T toggles following. N begins; V saves success; X failure; Q stops.', flush=True)
        if args.viewer == 'rerun':
            print(f'Keyboard source: {args.keyboard_source}. Keep the launching terminal focused.' if args.keyboard_source == 'terminal' else 'Keyboard source: desktop global listener.', flush=True)
        while time.monotonic() - started < args.seconds:
            control.pulse()
            while not control.pending.empty():
                row = control.pending.get_nowait()
                robot_log.write(json.dumps(row, allow_nan=False) + '\n')
            now = time.monotonic()
            renew(args.recording_lease, episodes.current is not None, now)
            image, arrival = wrist.snapshot()
            if arrival is None or now-arrival>.75:
                if not wrist_gap:
                    if control.enabled:
                        control.toggle()
                    if episodes.current is not None:
                        finish_episode('incomplete',now,'wrist_camera_gap')
                    wrist.enabled.clear()
                    renew(args.recording_lease,False,now)
                pending_start=None
                wrist_gap=True
                message='Wrist reconnecting; arm holding. Wait for live preview, then '+('SPACE.' if args.default_success else 'N.')
            elif wrist_gap:
                wrist_gap=False
                message='Wrist ready again. '+('SPACE' if args.default_success else 'N')+' starts a new attempt at the current arm pose.'
            observation = json.loads(observation_file.read_text())
            if not observation.get('running'):
                raise RuntimeError('Perception stopped: '+str(observation.get('error') or 'see perception.log'))
            if observation.get('rig_sha256') != manifest['rig_sha256']:
                raise RuntimeError('Perception calibration changed during capture')
            if now - observation['host_published_monotonic_s'] > 2.:
                raise RuntimeError('Perception stopped publishing for more than 2 seconds; see perception.log')
            ready=observation.get('fixed_cameras_ready',True)
            if not ready:
                if not fixed_gap:
                    if control.enabled:control.toggle()
                    if episodes.current is not None:finish_episode('incomplete',now,'fixed_camera_gap')
                    pending_start=None
                    wrist.enabled.clear()
                    renew(args.recording_lease,False,now)
                    print('Fixed camera gap: recording stopped; following paused. Waiting for camera recovery.',flush=True)
                fixed_gap=True
                message='Fixed cameras reconnecting; arm paused. Wait for live frames.'
            elif fixed_gap:
                fixed_gap=False
                message='Fixed cameras ready again. Press SPACE to re-clutch and start a new trial.'
                print(message,flush=True)
            if episodes.current is not None:
                if args.static_object and 'operator_visible_held' not in episodes.current['events']:
                    if observe_motion(episodes.current['static_object'],observation,time.monotonic()):
                        episodes.save()
                        print('Static position assumption ended: measured object displacement. Lift/release phases are derived at export; X if nudged before grasp.',flush=True)
                fixed_buffer.receive(observation,time.monotonic())
            fixed_buffer.drain()
            view = image.copy() if image is not None else np.zeros((480,640,3),np.uint8)
            if args.clutched:
                visible=not observation.get('stale',True) and (observation.get('fusion') or {}).get('position_output_m') is not None
                labels=[f"{args.task} | {args.split} | following={control.enabled} | saved successes={successes}/{args.max_successes or 'unlimited'} | active episode={None if episodes.current is None else episodes.current['episode_id']+1}",
                        'SPACE start | ENTER stop/save | X failed | Q finish | T pause/re-clutch' if args.default_success else 'N record | V success | X failed | Q finish | T pause/re-clutch',
                        ('STATIC: automatic lift/release labels | ENTER success | X if nudged before grasp' if args.static_object else
                         ('Object tracked' if visible else 'Waiting for object / pose unavailable; recording is allowed')),message]
            else:
                labels = [f"{args.task} | {args.split} | following={control.enabled}",
                          'N new | SPACE launch | H hand-clear | P response | C contact | L lift | R release',
                          'V success after 2s dwell | X failed | T following | Q stop', message]
            for i, label in enumerate(labels):
                cv2.putText(view, label, (8, 24 + 24*i), cv2.FONT_HERSHEY_SIMPLEX, .45, (0,0,255) if wrist_gap else (0,255,0), 1)
            if preview is not None:
                preview.set_status(feedback.snapshot(episodes.current,control.enabled,successes,args.max_successes,
                                                    state='CAMERA RECONNECTING' if wrist_gap or fixed_gap else (f'START IN {max(0,pending_start-time.monotonic()):.1f}s' if pending_start is not None else None)))
                key = preview.poll(image, labels)
                time.sleep(.01)
            else:
                cv2.imshow('Dynamic DOM capture', view)
                key = cv2.waitKey(10) & 255
            try:
                if key in (ord('q'), 27):
                    print('Finishing collection; exporting after shutdown.', flush=True)
                    break
                if key == ord('t'):
                    pending_start=None
                    if wrist_gap or fixed_gap:
                        raise ValueError('Wait for all cameras to reconnect before following')
                    enabled = control.toggle()
                    if not enabled and episodes.current is not None:
                        finish_episode('failed', time.monotonic(), 'following_paused')
                        wrist.enabled.clear()
                    message = f'Following={enabled}'
                elif args.default_success and key in (10, 13):
                    if pending_start is not None:
                        pending_start=None
                        message='Start canceled. No episode recorded; SPACE to try again.'
                        print(message,flush=True)
                        continue
                    if wrist_gap or fixed_gap:
                        raise ValueError('Camera reconnecting; this attempt was interrupted')
                    finish_episode('success',time.monotonic(),'operator_stopped_episode')
                    wrist.enabled.clear()
                    renew(args.recording_lease,False,time.monotonic())
                    successes += 1
                    message = f'Saved SUCCESS. Completed successes: {successes}/{args.max_successes or "unlimited"}. Recording stopped; reset, then SPACE for next trial.'
                    print(message, flush=True)
                    if args.max_successes and successes >= args.max_successes:
                        break
                elif (not args.default_success and key == ord('n')) or (args.default_success and key == 32):
                    if pending_start is not None:
                        raise ValueError('Start countdown already active; wait for the start sound')
                    if episodes.current is not None:
                        raise ValueError('Episode already recording; press Enter to stop/save or X to fail before starting another')
                    if wrist_gap or fixed_gap:
                        raise ValueError('Camera reconnecting; wait for fresh paired frames and wrist preview before starting')
                    if args.static_object:lock_position(observation,time.monotonic())
                    if not control.enabled:
                        if args.clutched:
                            control.toggle()
                        else:
                            raise ValueError('Enable matched leader following with T first')
                    if args.default_success:
                        pending_start=time.monotonic()+args.start_delay
                        message=f'Starting in {args.start_delay:.1f}s; '+('keep object still; wait for the start sound.' if args.static_object else 'wait for the start sound before pushing.')
                        print(message,flush=True)
                    else:
                        begin_episode(observation)
                        wrist.enabled.set()
                        message=f'RECORDING episode {episodes.current["episode_id"]+1}; '+('keep object stationary and demonstrate the pick.' if args.static_object else 'wait 1 second, then roll in.')
                        print(message,flush=True)
                elif key in (ord('v'), ord('x')):
                    if pending_start is not None:
                        pending_start=None
                        message='Start canceled. No episode recorded; SPACE to try again.'
                        print(message,flush=True)
                        continue
                    finish_episode('success' if key == ord('v') else 'failed', time.monotonic(),
                                    None if key == ord('v') else 'operator_marked_failure')
                    wrist.enabled.clear()
                    message = 'Attempt saved as ' + ('SUCCESS' if key == ord('v') else 'FAILED') + '; reset, then ' + ('SPACE' if args.default_success else 'N') + ' for next attempt'
                    print(message, flush=True)
                    if key==ord('v'):
                        successes+=1
                        if args.max_successes and successes>=args.max_successes:
                            break
                elif key in (32, ord('h'), ord('p'), ord('c'), ord('l'), ord('r')):
                    event = {32:'launch', ord('h'):'hand_clear', ord('p'):'response',
                             ord('c'):'contact', ord('l'):'lift', ord('r'):'release'}[key]
                    episodes.event(event, time.monotonic())
                    if args.static_object and key==ord('l'):
                        episodes.event('operator_visible_held',episodes.current['events']['lift'])
                    message = 'Confirmed visible grasp/lift; R at release.' if args.static_object and key==ord('l') else f'Marked {event}; continue moving'
            except ValueError as exc:
                message = str(exc)
                print('Key not applied: '+message, flush=True)
            if pending_start is not None and time.monotonic() >= pending_start:
                # No sleep: heartbeat, camera checks, keys and previews continue during countdown.
                if wrist_gap or fixed_gap or not control.enabled:
                    pending_start=None
                    message='Start canceled: wait for live cameras and enabled following.'
                    print(message,flush=True)
                else:
                    pending_start=None
                    try:
                        begin_episode(observation)
                    except ValueError as exc:
                        message=str(exc)
                        print('Start canceled: '+message,flush=True)
                    else:
                        wrist.enabled.set()
                        renew(args.recording_lease,True,time.monotonic())
                        if audio is not None:audio.play('start')
                        message=f'RECORDING episode {episodes.current["episode_id"]+1}; '+('keep object stationary and demonstrate the pick.' if args.static_object else 'push now.')
                        print(message,flush=True)

    except (Exception, KeyboardInterrupt) as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        if control is not None:
            try:
                control.close()
                if control.error:
                    error = error or control.error
            except BaseException as exc:
                robot_stopped = False
                cleanup_errors.append(str(exc))
            if logs:
                try:
                    while not control.pending.empty():
                        logs[0].write(json.dumps(control.pending.get_nowait(), allow_nan=False) + '\n')
                except BaseException as exc:
                    cleanup_errors.append('Robot log drain: ' + str(exc))
        if episodes.current is not None:
            try:
                finish_episode('incomplete', time.monotonic(), error or 'operator_quit_or_duration')
            except BaseException as exc:
                cleanup_errors.append('Attempt finalization: ' + str(exc))
        if lease_owner is not None:
            try:
                renew(args.recording_lease, False, time.monotonic())
            except BaseException as exc:
                cleanup_errors.append('Lease release: ' + str(exc))
            lease_owner.close()
        try:
            wrist.close()
        except BaseException as exc:
            cleanup_errors.append(str(exc))
        if fixed_buffer is not None:
            try:
                fixed_buffer.finish()
            except BaseException as exc:
                cleanup_errors.append('Fixed RGB-D finalization: '+str(exc))
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
                cleanup_errors.append('Log close: ' + str(exc))
        if audio is not None:
            try:audio.close()
            except BaseException as exc:cleanup_errors.append('Audio close: '+str(exc))
        if preview is not None:
            try:
                preview.set_status(feedback.snapshot(None,False,locals().get('successes',0),args.max_successes,
                                                    state='STOPPED - CHECK ERROR' if error else 'FINISHED'))
                preview.close()
            except BaseException as exc: cleanup_errors.append('Rerun close: '+str(exc))
        try:
            if args.viewer == 'opencv': cv2.destroyAllWindows()
        except BaseException as exc:
            cleanup_errors.append('Preview close: ' + str(exc))
        write_json(root / 'summary.json', dict(running=False, error=error, cleanup_errors=cleanup_errors,
                                               attempts=episodes.count,
                                               torque_changed=bool(control is not None and getattr(control,'torque_enabled_here',False))))
    if error:
        print(error, flush=True)
    return 1 if error or cleanup_errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
