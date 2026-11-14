"""Export dynamic teleoperation sessions to DOM's LeRobot v2.1 format, offline."""
import argparse
from bisect import bisect_left, bisect_right
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .grasp_prior import follower_pose

FPS = 25
CAMERAS = {'opst_cam': 'front', 'side_cam': 'side', 'wrist_cam': 'wrist'}
DATA_PATH = 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'
VIDEO_PATH = 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'
POSE_NAMES = ['ee_pos_x', 'ee_pos_y', 'ee_pos_z', 'ee_rot_x', 'ee_rot_y', 'ee_rot_z']
ENV_NAMES = [f'object_{kind}_{axis}' for kind in ('pos', 'rot', 'vel') for axis in 'xyz']


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def jsonl_read(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def jsonl_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def local_file(root, name):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError(f'Missing or external raw dependency: {name}')
    return path


def dom_euler(matrix):
    """Match DynamicVLA helpers exactly: xyz radians; X/Z in [0, 2pi)."""
    euler = Rotation.from_matrix(np.asarray(matrix)[:3, :3]).as_euler('xyz')
    euler[[0, 2]] %= 2*np.pi
    return euler


def pose6(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Invalid TCP transform')
    return np.r_[matrix[:3, 3], dom_euler(matrix)]


def command_raw(calibrated, calibration):
    """Inverse of pinned DynamixelMotorsBus.revert_calibration (4096 ticks)."""
    # RobotIO.goal casts to float32 before the pinned bus rounds into raw ticks.
    values = np.asarray(calibrated, dtype=np.float32).copy()
    if values.shape != (9,) or not np.isfinite(values).all():
        raise ValueError('Expected nine finite calibrated command values')
    for i, mode in enumerate(calibration['calib_mode']):
        if mode == 'DEGREE':
            values[i] = values[i] / 180 * 2048 - calibration['homing_offset'][i]
            if calibration['drive_mode'][i]:
                values[i] *= -1
        elif mode == 'LINEAR':
            values[i] = (values[i] / 100 * (calibration['end_pos'][i] - calibration['start_pos'][i])
                         + calibration['start_pos'][i])
        else:
            raise ValueError(f'Unsupported calibration mode {mode}')
    return np.round(values).astype(np.int32)


def command_action(row, manifest):
    raw = command_raw(row['commanded_calibrated'], manifest['buses']['follower']['calibration'])
    _, target = follower_pose(raw, manifest['tcp_offset_m'])
    closed, opened = manifest['gripper_closed_percent'], manifest['gripper_open_percent']
    # DOM uses +1=open, -1=closed, retaining continuous intermediate commands.
    gripper = np.clip(2 * (row['commanded_calibrated'][-1] - closed) / (opened - closed) - 1, -1, 1)
    return np.r_[pose6(target), gripper]


class TimeIndex:
    def __init__(self, rows, time_key):
        self.rows = sorted(rows, key=time_key)
        self.times = [float(time_key(row)) for row in self.rows]
        if not np.isfinite(self.times).all():
            raise ValueError('Nonfinite stream timestamps')

    def past(self, now, max_age=math.inf):
        i = bisect_right(self.times, now) - 1
        if i < 0 or now - self.times[i] > max_age:
            raise ValueError(f'No causal sample within {max_age:g}s at {now:g}')
        return self.rows[i], now - self.times[i]

    def next(self, now, max_delay):
        i = bisect_left(self.times, now)
        if i == len(self.times) or self.times[i] - now > max_delay:
            raise ValueError('No issued action near anchor')
        return self.rows[i], self.times[i] - now


def object_position(row):
    fusion = row.get('fusion') or {}
    value = fusion.get('position_output_m')
    if value is None or fusion.get('stale', True) or row.get('stale', True):
        return None
    p = np.asarray(value, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all():
        return None
    return p


def observation_measurement_time(row):
    return float((row.get('timing') or {}).get('oldest_host_arrival_monotonic_s',
                 row['host_recorded_monotonic_s']))


def align_episode(root, manifest, episode, *, camera_age=.25, robot_age=.12, object_age=.3):
    """Regular anchors; reject gaps instead of compressing time by dropping rows."""
    root = Path(root)
    robot = jsonl_read(root / 'robot.jsonl')
    observations = jsonl_read(root / 'observations.jsonl')
    wrist = jsonl_read(root / 'wrist/frames.jsonl')
    robot_index = TimeIndex(robot, lambda r: r['host_read_end_monotonic_s'])
    commands = TimeIndex([r for r in robot if r.get('commanded_calibrated') is not None],
                         lambda r: r['host_cycle_end_monotonic_s'])
    obs_index = TimeIndex(observations, lambda r: r['host_recorded_monotonic_s'])
    positions = TimeIndex([r for r in observations if object_position(r) is not None],
                          lambda r: r['host_recorded_monotonic_s'])
    wrist_index = TimeIndex(wrist, lambda r: r['available_monotonic_s'])
    static = None
    if manifest.get('static_object_collection'):
        from .static_object import SCHEMA,validate_lock
        if manifest['static_object_collection'].get('schema') != SCHEMA:
            raise ValueError('Unknown static collection schema')
        static = episode.get('static_object')
        if static is None:raise ValueError('Static episode has no per-attempt position lock')
        validate_lock(static,episode['start_monotonic_s'])
    streams = [robot_index, commands, obs_index, wrist_index]+([] if static is not None else [positions])
    if any(not index.times for index in streams):
        raise ValueError('Episode/session is missing robot, command, fixed, wrist or object samples')
    begin, end = episode['start_monotonic_s'], episode['end_monotonic_s']
    if end is None or not begin < end:
        raise ValueError('Episode has no valid end')
    # No state or images from an earlier attempt may initialize a new one.
    firsts = []
    for index in streams:
        i = bisect_left(index.times, begin)
        if i == len(index.times) or index.times[i] >= end:
            raise ValueError('No stream data within attempt')
        firsts.append(index.times[i])
    start = max(begin, *firsts)
    # A post-V reset command must never become the previous episode's action label.
    last_command = commands.times[bisect_right(commands.times, end)-1]
    label_end = min(end, last_command)
    count = int(math.floor((label_end - start) * FPS))
    if count < 20:
        raise ValueError('Fewer than one action horizon of synchronized data')
    rows, quality, frame_paths = [], [], {key: [] for key in CAMERAS}
    assumed_rotation = np.array([0., 0., np.deg2rad(manifest.get('heading_deg', 0)) % (2*np.pi)])
    for index in range(count):
        now = start + index/FPS
        state, state_age = robot_index.past(now, robot_age)
        if state['host_read_end_monotonic_s'] < begin or not state.get('following'):
            raise ValueError('Robot not following during attempt')
        action, action_delay = commands.next(now, robot_age)
        if action['host_cycle_end_monotonic_s'] > end:
            raise ValueError('Action outside attempt')
        obs, obs_age = obs_index.past(now, camera_age)
        wrist_row, wrist_age = wrist_index.past(now, camera_age)
        if static is not None:
            position_row = obs
            position_age = now-static['measurement_monotonic_s']
            p = np.asarray(static['position_m'],float)
        else:
            position_row, position_age = positions.past(now)
            if position_row['host_recorded_monotonic_s'] < begin:
                raise ValueError('Object state belongs to previous attempt')
            position_age = now - observation_measurement_time(position_row)
            p = object_position(position_row)
        velocity = (obs.get('velocity_report') or {}).get('velocity_mps')
        velocity_age = now - observation_measurement_time(obs)
        valid_velocity = (velocity is not None and 0 <= velocity_age <= object_age
                          and object_position(obs) is not None)
        if static is not None:valid_velocity=False
        v = np.asarray(velocity if valid_velocity else [0, 0, 0], float)
        if v.shape != (3,) or not np.isfinite(v).all():
            raise ValueError('Malformed object velocity')
        images = obs['rgbd']
        camera_ages, native_times = {}, {}
        for camera, role in CAMERAS.items():
            if role == 'wrist':
                path = local_file(root, wrist_row['file'])
                arrival = wrist_row['timing']['host_arrival_monotonic_s']
                native_times[role] = wrist_row['timing']['color_timestamp_s']
            else:
                path = local_file(root, images['files'][role])
                arrival = images['views'][role]['timing']['host_arrival_monotonic_s']
                native_times[role] = images['views'][role]['timing']['color_timestamp_s']
            if not 0 <= now - arrival <= camera_age:
                raise ValueError(f'{role} capture is too old for anchor')
            camera_ages[role] = now - arrival
            frame_paths[camera].append(path)
        rows.append({'action': command_action(action, manifest).astype(np.float32).tolist(),
                     'observation.state': pose6(state['T_base_tcp_candidate']).astype(np.float32).tolist(),
                     'observation.environment_state': np.r_[p, assumed_rotation, v].astype(np.float32).tolist(),
                     'timestamp': index/FPS, 'frame_index': index})
        quality.append(dict(frame_index=index, anchor_monotonic_s=now, robot_age_s=state_age,
                            action_delay_s=action_delay, perception_age_s=obs_age,
                            wrist_archive_age_s=wrist_age, camera_ages_s=camera_ages,
                            native_color_timestamps_s=native_times,
                            position_age_s=position_age, velocity_age_s=velocity_age,
                            position_measurement_monotonic_s=observation_measurement_time(position_row),
                            velocity_measurement_monotonic_s=observation_measurement_time(obs),
                            object_frame=manifest['rig'].get('output_frame', 'unknown'),
                            robot_frame='right_base',
                            base_calibration_validated=bool(manifest['rig'].get('base_calibration_validated',False)),
                            position_current=0 <= position_age <= object_age,
                            velocity_estimated=bool(valid_velocity), velocity_zero_filled=not valid_velocity,
                            orientation_source='assumed_initial_heading', orientation_measured=False,
                            camera_source_files={camera:str(paths[-1].relative_to(root.resolve()))
                                                 for camera,paths in frame_paths.items()},
                            position_semantics='visible_surface_proxy'))
        if static is not None:
            from .static_object import assumption_active,SCHEMA
            active_assumption=assumption_active(static,episode.get('events',{}),now)
            quality[-1].update(static_collection_schema=SCHEMA,static_assumption_valid=active_assumption,
                static_position_seed_m=static['position_m'],static_locked_monotonic_s=static['locked_monotonic_s'],
                static_invalidated_monotonic_s=static.get('invalidated_monotonic_s'),
                position_current=False,velocity_estimated=False,velocity_zero_filled=False,
                position_measurement_monotonic_s=static['measurement_monotonic_s'],
                velocity_measurement_monotonic_s=None,velocity_age_s=position_age,
                velocity_input_source='operator_static_assumption' if active_assumption else 'unavailable_after_static_assumption',
                position_semantics='locked_initial_surface_proxy; assumption during approach; unused placeholder otherwise')
            measured_gripper=(state.get('follower_calibrated') or [None])[-1]
            if measured_gripper is not None:
                closed,opened=manifest['gripper_closed_percent'],manifest['gripper_open_percent']
                quality[-1]['measured_gripper_normalized']=float(2*(measured_gripper-closed)/(opened-closed)-1)
        depth_quality = ((position_row.get('views') or {}).get('front') or {}).get('quality') or {}
        if static is not None:depth_quality=static.get('position_quality',{})
        if position_row.get('offline_tracking_recovery'):
            quality[-1].update(offline_tracking_recovery=position_row['offline_tracking_recovery'],
                               recovered_depth_valid_fraction=depth_quality.get('valid_fraction'),
                               recovered_depth_mad_m=depth_quality.get('depth_mad_m'),
                               sparse_depth_recovery=depth_quality.get('sparse_depth_recovery', False))
        if depth_quality.get('depth_correction_model_sha256'):
            quality[-1].update(depth_correction_model_sha256=depth_quality['depth_correction_model_sha256'],
                               depth_correction_extrapolated=depth_quality.get('depth_correction_extrapolated',False))
    if static is not None:
        from .static_object import AUTOMATIC_PHASE_METHOD,annotate_automatic_phases
        if manifest['static_object_collection'].get('phase_labeling')==AUTOMATIC_PHASE_METHOD:
            annotate_automatic_phases(rows,quality,episode['outcome'])
    from .dataset_quality import phases
    labels, required = phases(rows, quality, episode.get('events'))
    from .static_object import quality_valid
    for q, label, needs_velocity in zip(quality, labels, required):
        q.update(manipulation_phase=label, phase_confirmed=label=='held_confirmed',
                 velocity_required=bool(needs_velocity),
                 tracking_usable=bool((q['position_current'] and q['velocity_estimated']) or quality_valid(q) or not needs_velocity))
    return rows, frame_paths, quality


def features(codec='av1', image_size_wh=(480,360)):
    width,height=image_size_wh
    result = {
        'action': dict(dtype='float32', shape=[7], names=POSE_NAMES + ['gripper']),
        'observation.state': dict(dtype='float32', shape=[6], names=POSE_NAMES),
        'observation.environment_state': dict(dtype='float32', shape=[9], names=ENV_NAMES),
    }
    for camera in CAMERAS:
        video_info = {'has_audio':False, 'video.codec':codec, 'video.fps':FPS,
                      'video.pix_fmt':'yuv420p', 'video.is_depth_map':False}
        result['observation.images.' + camera] = dict(
            dtype='video', shape=[height,width,3], names=['height','width','channel'],
            video_info=video_info,
            info={**{k:v for k,v in video_info.items() if k != 'video.is_depth_map'},
                  'video.height':height, 'video.width':width, 'video.channel':3})
    for key in ('timestamp','frame_index','episode_index','index','task_index'):
        result[key] = dict(dtype='float32' if key == 'timestamp' else 'int64', shape=[1], names=None)
    return result


def numeric_stats(data):
    a = np.asarray(data, dtype=np.float64)
    return {key: np.atleast_1d(value).tolist() for key, value in
            dict(min=a.min(axis=0), max=a.max(axis=0), mean=a.mean(axis=0),
                 std=a.std(axis=0), count=np.array([len(a)])).items()}


def encode_video(paths, output, hashes, *, preprocessing=None, role=None):
    """Streaming AV1 encode; source BGR -> RGB; no full-episode image buffer."""
    import av
    import cv2
    output.parent.mkdir(parents=True, exist_ok=True)
    width,height=(384,384) if preprocessing is not None else (480,360)
    sums = np.zeros(3)
    squares = np.zeros(3)
    minimum, maximum = np.ones(3), np.zeros(3)
    pixels = 0
    previous, rgb = None, None
    with av.open(str(output), 'w') as container:
        stream = container.add_stream('libsvtav1', rate=FPS)
        stream.width, stream.height, stream.pix_fmt = width, height, 'yuv420p'
        stream.options = {'preset':'10', 'crf':'28', 'svtav1-params':'lp=2:log-level=0'}
        for i, path in enumerate(paths):
            if path != previous:
                with np.load(path, allow_pickle=False) as raw:
                    bgr = raw['bgr']
                    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
                        raise ValueError(f'Invalid RGB-D image {path}')
                    if preprocessing is not None:
                        from .image_preprocessing import transform
                        bgr = transform(bgr,preprocessing,role)
                    else:
                        bgr = cv2.resize(bgr, (width,height), interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if str(path) not in hashes:
                    hashes[str(path)] = sha256(path)
                previous = path
            # All frames contribute to streaming moments, using the same spatial subsample.
            sample = rgb[::4, ::4].reshape(-1, 3).astype(np.float64)/255
            sums += sample.sum(axis=0)
            squares += (sample**2).sum(axis=0)
            minimum = np.minimum(minimum, sample.min(axis=0))
            maximum = np.maximum(maximum, sample.max(axis=0))
            pixels += len(sample)
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    mean = sums/pixels
    std = np.sqrt(np.maximum(0, squares/pixels - mean**2))
    return {key: value.reshape(3,1,1).tolist() for key, value in
            dict(min=minimum, max=maximum, mean=mean, std=std).items()} | {'count':[len(paths)]}


def write_parquet(rows, output):
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([
        pa.field('action', pa.list_(pa.float32(), 7)),
        pa.field('observation.state', pa.list_(pa.float32(), 6)),
        pa.field('observation.environment_state', pa.list_(pa.float32(), 9)),
        pa.field('timestamp', pa.float32()),
        *[pa.field(k, pa.int64()) for k in ('frame_index','episode_index','index','task_index')],
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), output)


def export_sessions(sessions, output, *, camera_age=.25, robot_age=.12, object_age=.3, image_preprocessing=None):
    preprocessing=None
    if image_preprocessing is not None:
        from .image_preprocessing import load_spec,validate_spec,geometry
        preprocessing=(load_spec(image_preprocessing) if not isinstance(image_preprocessing,dict) else validate_spec(image_preprocessing))
        import torch
        torch.set_num_threads(4)
    width,height=(384,384) if preprocessing is not None else (480,360)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Output already exists; choose a new version')
    staging = output.with_name(output.name + '.partial')
    staging.mkdir(parents=True, exist_ok=False)
    candidates, audit, seen_sessions = [], [], set()
    hashes = {}
    try:
        for root in map(lambda p: Path(p).resolve(), sessions):
            manifest = json.loads((root / 'manifest.json').read_text())
            if preprocessing is not None and manifest['rig']['front_serial'] != preprocessing['front_serial']:
                raise ValueError('Front crop was selected for another camera')
            if manifest.get('schema') != 'dom_dynamic_capture_v1' or manifest.get('mode') != 'execute':
                raise ValueError(f'Not an executed dynamic session: {root}')
            if manifest['split'] not in ('train','validation','test') or root in seen_sessions:
                raise ValueError('Invalid split or duplicate input session')
            seen_sessions.add(root)
            hashes[str(root / 'manifest.json')] = sha256(root / 'manifest.json')
            for name in ('robot.jsonl','observations.jsonl','wrist/frames.jsonl'):
                hashes[str(root / name)] = sha256(root / name)
            for path in sorted((root / 'episodes').glob('*.json')):
                episode = json.loads(path.read_text())
                hashes[str(path)] = sha256(path)
                if episode['split'] != manifest['split'] or episode['task_id'] != manifest['task_id']:
                    raise ValueError('Episode task/split differs from its recording session')
                if episode['outcome'] != 'success':
                    audit.append(dict(source=str(path), exported=False, reason=episode['outcome']))
                    continue
                candidates.append((root, manifest, episode, path))
        order = {'train':0, 'validation':1, 'test':2}
        candidates.sort(key=lambda x: (order[x[1]['split']], str(x[0]), x[2]['episode_id']))
        metadata, stats_rows, tasks, camera_rows, source_rows = [], [], [], [], []
        splits = {name: [] for name in order}
        total = 0
        for root, manifest, episode, source in candidates:
            # Reject the entire attempt on an alignment gap, preserving regular physical time.
            try:
                rows, paths, quality = align_episode(root, manifest, episode,
                                                    camera_age=camera_age, robot_age=robot_age, object_age=object_age)
            except ValueError as exc:
                audit.append(dict(source=str(source), exported=False, reason=str(exc)))
                continue
            ep = len(metadata)
            task_spec=manifest['task']
            task_type=task_spec.get('task_type','place')
            if task_type not in ('pick','place','long-horizon'):raise ValueError('Unsupported task type: '+str(task_type))
            objects=task_spec.get('objects') or task_spec['instruction'].split('Pick up the ')[-1].split(' and place')[0]
            instruction = dict(task=task_type,objects=objects,instruction=task_spec['instruction'])
            if task_type!='pick':instruction['containers']=task_spec.get('containers','box')
            task = json.dumps(instruction, sort_keys=True)
            if task not in tasks:
                tasks.append(task)
            task_index = tasks.index(task)
            for i, row in enumerate(rows):
                row.update(episode_index=ep, index=total+i, task_index=task_index)
            data_file = staging / DATA_PATH.format(episode_chunk=ep//1000, episode_index=ep)
            write_parquet(rows, data_file)
            ep_stats = {key: numeric_stats([row[key] for row in rows]) for key in rows[0]}
            for camera in CAMERAS:
                key = 'observation.images.' + camera
                dest = staging / VIDEO_PATH.format(episode_chunk=ep//1000, episode_index=ep, video_key=key)
                ep_stats[key] = encode_video(paths[camera], dest, hashes, preprocessing=preprocessing, role=CAMERAS[camera])
            stats_rows.append(dict(episode_index=ep, stats=ep_stats))
            metadata.append(dict(episode_index=ep, tasks=[task], length=len(rows)))
            splits[manifest['split']].append(ep)
            source_rows.append(dict(episode_index=ep, source_episode=str(source), session_id=manifest['session_id'],
                                    split=manifest['split'], raw_session=str(root),
                                    tcp_offset_m=manifest['tcp_offset_m'],
                                    position_semantics='estimated_visible_surface_proxy', orientation='assumed_initial_heading',
                                    collector=manifest['collector'], native_camera_hz=manifest['native_camera_hz']))
            if manifest.get('stream_snapshot_original_session'):
                original=Path(manifest['stream_snapshot_original_session'])
                source_rows[-1].update(raw_session=str(original),source_episode=str(original/'episodes'/source.name))
            if manifest.get('expert_recovery'):
                from .recovery_demonstrations import validate_context
                source_rows[-1]['expert_recovery'] = validate_context(manifest['expert_recovery'])
            if manifest.get('static_object_collection'):
                source_rows[-1]['static_object_collection']=manifest['static_object_collection']
                source_rows[-1]['static_object']=episode['static_object']
            if manifest.get('offline_tracking_recovery'):
                recovery = manifest['offline_tracking_recovery']
                source_rows[-1]['offline_tracking_recovery'] = recovery
                source_rows[-1]['original_source_episode'] = str(Path(recovery['source'])/'episodes'/source.name)
            camera_rows.append(dict(episode_index=ep, cameras=[
                dict(name=c, source_role=r, width=width, height=height,
                     **({'image_geometry': geometry(preprocessing,r),
                         'calibration_image_space':'original raw 640x480; apply image_geometry pixel transform'} if preprocessing is not None else {}),
                     calibration=manifest['rig'] if r != 'wrist' else {'serial':manifest['wrist_serial']})
                for c, r in CAMERAS.items()]))
            jsonl_write(staging / f'quality/episode_{ep:06d}.jsonl', quality)
            audit.append(dict(source=source_rows[-1]['source_episode'], exported=True, episode_index=ep,
                              rows=len(rows), stale_position_frames=sum(not q['position_current'] for q in quality),
                              recorded_camera_hz={role:observed_rate([q['native_color_timestamps_s'][role] for q in quality])
                                                  for role in ('front','side','wrist')},
                              zero_velocity_frames=sum(q['velocity_zero_filled'] for q in quality)))
            total += len(rows)
        if not metadata:
            raise ValueError('No successful, alignable episodes; inspect export audit')
        split_ranges = {}
        for name, ids in splits.items():
            if ids:
                split_ranges[name] = f'{ids[0]}:{ids[-1]+1}'
        info = dict(codebase_version='v2.1', robot_type='aloha_solo', fps=FPS,
                    total_episodes=len(metadata), total_frames=total, total_tasks=len(tasks),
                    total_videos=len(metadata)*len(CAMERAS), total_chunks=math.ceil(len(metadata)/1000),
                    chunks_size=1000, data_path=DATA_PATH, video_path=VIDEO_PATH,
                    splits=split_ranges, features=features(image_size_wh=(width,height)))
        if preprocessing is not None:
            json_write(staging/'meta/image_preprocessing.json',preprocessing)
        json_write(staging / 'meta/info.json', info)
        jsonl_write(staging / 'meta/episodes.jsonl', metadata)
        jsonl_write(staging / 'meta/tasks.jsonl', [dict(task_index=i, task=t) for i,t in enumerate(tasks)])
        jsonl_write(staging / 'meta/episodes_stats.jsonl', stats_rows)
        jsonl_write(staging / 'meta/camera.jsonl', camera_rows)
        jsonl_write(staging / 'meta/sources.jsonl', source_rows)
        json_write(staging / 'meta/splits.json', splits)
        # Explicit train-only aggregate, using the installed v2.1 aggregator if available.
        from lerobot.common.datasets.compute_stats import aggregate_stats
        train_stats = [s['stats'] for s in stats_rows if s['episode_index'] in set(splits['train'])]
        if train_stats:
            arrays = [{key:{k:np.asarray(v) for k,v in stat.items()} for key,stat in row.items()} for row in train_stats]
            aggregate = aggregate_stats(arrays)
            json_write(staging / 'meta/train_stats.json', {key:{k:v.tolist() for k,v in stat.items()} for key,stat in aggregate.items()})
        json_write(staging / 'meta/raw_hashes.json', hashes)
        json_write(staging / 'export_report.json', dict(complete=False, attempts=audit, frames=total,
                   episodes=len(metadata), camera_age_s=camera_age, robot_age_s=robot_age, object_age_s=object_age,
                   native_camera_requests_hz=sorted({m['native_camera_hz'] for _,m,_,_ in candidates}),
                   resampling='causal zero-order hold onto 25Hz; source timestamps and repeated frames retained',
                   rotation='xyz Euler radians; X/Z modulo 2pi; matches DOM helper',
                   labels='commanded Cartesian pose and +1=open/-1=closed, not next measured state'))
        validate_dataset(staging)
        report = json.loads((staging/'export_report.json').read_text())
        report['complete'] = True
        json_write(staging/'export_report.json', report)
        staging.rename(output)
        return report
    except BaseException as exc:
        json_write(staging / 'failure.json', dict(error=f'{type(exc).__name__}: {exc}', attempts=audit))
        raise


def observed_rate(timestamps):
    stamps = np.unique(timestamps)
    return None if len(stamps) < 2 else float((len(stamps)-1)/(stamps[-1]-stamps[0]))


def validate_dataset(root, *, decode_videos=True,use_video_cache=True):
    """Check rows/schema every time; reuse full video checks by content hash."""
    import av
    import pyarrow.parquet as pq
    root = Path(root)
    info = json.loads((root / 'meta/info.json').read_text())
    preprocessing_path=root/'meta/image_preprocessing.json'
    if preprocessing_path.exists():
        from .image_preprocessing import load_spec
        load_spec(preprocessing_path)
        width,height=384,384
    else:
        width,height=480,360
    expected = features(image_size_wh=(width,height))
    if info['codebase_version'] != 'v2.1' or info['fps'] != FPS or info['features'] != expected:
        raise ValueError('DOM schema mismatch')
    episodes = jsonl_read(root / 'meta/episodes.jsonl')
    tasks = jsonl_read(root / 'meta/tasks.jsonl')
    stats = jsonl_read(root / 'meta/episodes_stats.jsonl')
    splits = json.loads((root / 'meta/splits.json').read_text())
    ids = [ep for group in splits.values() for ep in group]
    if sorted(ids) != list(range(len(episodes))) or len(set(ids)) != len(ids):
        raise ValueError('Split overlap or missing episodes')
    global_index = 0;video_jobs=[]
    for ep, item in enumerate(episodes):
        if item['episode_index'] != ep or stats[ep]['episode_index'] != ep:
            raise ValueError('Episode metadata indices differ')
        rows = pq.read_table(root / DATA_PATH.format(episode_chunk=ep//1000, episode_index=ep)).to_pylist()
        if len(rows) != item['length']:
            raise ValueError('Parquet length mismatch')
        for i, row in enumerate(rows):
            if (row['frame_index'] != i or row['episode_index'] != ep or row['index'] != global_index+i
                    or not 0 <= row['task_index'] < len(tasks) or abs(row['timestamp']-i/FPS) > 1e-4):
                raise ValueError('Row index/timestamp mismatch')
            for key, length in [('action',7), ('observation.state',6), ('observation.environment_state',9)]:
                if len(row[key]) != length or not np.isfinite(row[key]).all():
                    raise ValueError('Invalid numeric feature')
            if not -1 <= row['action'][-1] <= 1:
                raise ValueError('Invalid DOM gripper range')
        for camera in CAMERAS:
            key = 'observation.images.' + camera
            path = root / VIDEO_PATH.format(episode_chunk=ep//1000, episode_index=ep, video_key=key)
            if decode_videos:
                video_jobs.append((path,width,height,len(rows),FPS))
                continue
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                if stream.width != width or stream.height != height or stream.codec_context.name not in ('av1','libdav1d'):
                    raise ValueError('Video dimensions/codec mismatch')
                if stream.frames and stream.frames!=len(rows):raise ValueError('Video header frame count mismatch')
        global_index += len(rows)
    if global_index != info['total_frames'] or len(episodes) != info['total_episodes']:
        raise ValueError('Dataset totals mismatch')
    # Parse exactly the JSON structure expected by DynamicVLA's instruction generator.
    for task in tasks:
        instruction = json.loads(task['task'])
        if instruction['task'] not in ('pick','place','long-horizon') or not instruction.get('objects'):
            raise ValueError('Task is not DOM instruction metadata')
    if video_jobs:
        from .video_validation import validate_videos,DEFAULT_CACHE
        validate_videos(video_jobs,cache_path=DEFAULT_CACHE if use_video_cache else None,verbose=len(video_jobs)>=30)
    return dict(episodes=len(episodes), frames=global_index, splits={k:len(v) for k,v in splits.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    export = sub.add_parser('export')
    export.add_argument('--sessions', type=Path, nargs='+', required=True)
    export.add_argument('--output', type=Path, required=True)
    export.add_argument('--image-preprocessing',type=Path,help='Saved manual front crop and 384px KRVLA resize/pad settings')
    export.add_argument('--max-camera-age', type=float, default=.25)
    export.add_argument('--max-robot-age', type=float, default=.12)
    export.add_argument('--max-object-age', type=float, default=.3)
    validate = sub.add_parser('validate')
    validate.add_argument('--dataset', type=Path, required=True)
    validate.add_argument('--force-video-decode',action='store_true',help='Ignore cached video checks and decode every frame')
    args = parser.parse_args(argv)
    if args.mode == 'export':
        limits = [args.max_camera_age, args.max_robot_age, args.max_object_age]
        if not np.isfinite(limits).all() or min(limits) <= 0 or max(limits) > 1:
            parser.error('Alignment ages must be finite, positive and at most one second')
        report = export_sessions(args.sessions, args.output, camera_age=limits[0], robot_age=limits[1], object_age=limits[2], image_preprocessing=args.image_preprocessing)
        print(json.dumps(dict(episodes=report['episodes'], frames=report['frames'], output=str(args.output))))
    else:
        print(json.dumps(validate_dataset(args.dataset,use_video_cache=not args.force_video_decode)))


if __name__ == '__main__':
    main()
