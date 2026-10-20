"""Run `python -m real_robot --help` from the repository root."""
import os
# Set before NumPy/OpenCV imports, independently of the training environments.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .geometry import fit_rigid_transform, transform_point, validate_transform
from .tracking import VelocityTracker
from .controller import HoverLimits, ShadowHoverController, hover_preview_target


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def new_run(path):
    root = Path(path)
    root.mkdir(parents=True, exist_ok=False)
    return root


def calibration_for(path, serial, arm=None):
    if not path:
        return None
    d = json.loads(Path(path).read_text())
    if arm is not None and d.get('arm') != arm:
        raise ValueError(f'Calibration must identify the selected {arm} arm; do not reuse another arm\'s transform')
    if (d.get('camera_serial') != serial or d.get('destination_frame') != 'robot_base'
            or d.get('static_camera') is not True or d.get('calibration_validated') is not True):
        raise ValueError('Require a validated static camera-to-base calibration for this serial')
    return validate_transform(d['T_base_camera'])


def provisional_calibration_for(path, serial, arm):
    """Fit operator landmarks for display only, never a validated robot frame."""
    import hashlib
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if (data.get('camera_serial') != serial or data.get('arm') != arm
            or data.get('static_camera') is not True):
        raise ValueError('Provisional landmarks must match the fixed camera and selected arm')
    transform, errors = fit_rigid_transform(data['camera_points_m'], data['base_points_m'], min_points=3)
    report = {'camera_serial': serial, 'arm': arm, 'static_camera': True,
              'destination_frame': f'{arm}_base_estimate', 'calibration_validated': False,
              'T_base_camera': transform.tolist(), 'fit_errors_m': errors.tolist(),
              'landmark_count': len(errors), 'source_sha256': hashlib.sha256(raw).hexdigest(),
              'coordinate_note': 'Operator-defined approximate tabletop axes near robot base; SO101 kinematic axes unverified.',
              'usage': 'Unvalidated visualization only; independent base calibration required for robot control.'}
    return transform, report


def candidate_calibration_for(path, serial, arm):
    """Load the ChArUco fit for display without promoting it to validated control."""
    import hashlib
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if (data.get('schema') != 'aloha_handeye_candidate_v1'
            or data.get('front_camera_serial') != serial or arm != 'right'
            or data.get('arm') != arm or data.get('destination_frame') != 'right_base_estimate'
            or data.get('calibration_validated') is not False):
        raise ValueError('Require a matching provisional right-arm ChArUco/front-camera candidate')
    transform = validate_transform(data['T_right_base_front_camera_candidate'])
    report = {**data, 'T_base_camera': transform.tolist(),
              'source_sha256': hashlib.sha256(raw).hexdigest(),
              'usage': 'Visualization only; camera and base must remain fixed since capture.'}
    return transform, report


def track(args):
    import cv2
    from .camera import RGBDCamera
    from .perception import detect_mask, depth_measurement, hsv_bounds
    from .visualization import TrackingOverlay
    cv2.setNumThreads(1)
    if not args.fixed_camera:
        raise ValueError('This milestone requires --fixed-camera. Moving wrist-camera compensation is not yet wired.')
    if not 0 < args.seconds <= 300 or not 0 < args.min_depth < args.max_depth:
        raise ValueError('Use 0 < seconds <= 300 and 0 < min-depth < max-depth')
    transform = calibration_for(args.calibration, args.serial, args.arm)
    provisional = None
    if args.provisional_landmarks:
        transform, provisional = provisional_calibration_for(args.provisional_landmarks, args.serial, args.arm)
        print(f"UNVALIDATED display frame: {provisional['destination_frame']}; "
              f"maximum landmark fit error {1000*max(provisional['fit_errors_m']):.1f} mm.", flush=True)
    if args.candidate_calibration:
        transform, provisional = candidate_calibration_for(args.candidate_calibration, args.serial, args.arm)
        print('UNVALIDATED ChArUco frame: right_base_estimate. Preview only.', flush=True)
    if args.hover_height is not None:
        # Validate parameters before opening a camera or creating an output run.
        hover_preview_target(None, args.hover_height, args.hover_lookahead)
        if transform is None:
            raise ValueError('Hover preview requires a base-frame calibration or explicit estimate')
    args.hsv_low, args.hsv_high = hsv_bounds(args.detector, args.hsv_low, args.hsv_high)
    frame = provisional['destination_frame'] if provisional else ('robot_base' if transform is not None else 'camera_optical')
    tracker = VelocityTracker(frame=frame, stationary_updates=not args.disable_stationary_update)
    overlay = TrackingOverlay()
    root = new_run(args.output)
    metadata = {**vars(args), 'schema': 'dynamicvla_tracking_probe_v1',
                'gpu_used': False, 'motion_enabled': False, 'expert_episode': False,
                'position_semantics': 'visible_surface_proxy',
                'output_frame': frame, 'base_calibration_validated': transform is not None and provisional is None,
                'stationary_update': tracker.stationary_settings,
                'calibration': None if transform is None else transform.tolist(),
                'clock_note': 'Filter uses relative sensor time. Host arrival is recorded separately; transit latency is unmeasured.',
                'controller_note': 'Camera-only capture has no measured end-effector state and emits no robot action.'}
    write_json(root / 'manifest.json', metadata)
    if provisional:
        write_json(root / 'frame_estimate.json', provisional)
    camera, last_view, complete = None, None, False
    accepted = ready = frames = 0
    hover_targets = 0
    process_times = []
    start = time.monotonic()
    timestamp_origin = last_timestamp = clock_domain = None
    try:
        camera = RGBDCamera(args.serial, args.width, args.height, args.fps)
        start = time.monotonic()
        with (root / 'tracking.jsonl').open('w') as log:
            while time.monotonic() - start < args.seconds:
                bgr, depth, timing, deproject = camera.read()
                begin = time.monotonic()
                timestamp = timing['sensor_timestamp_s']
                if clock_domain is not None and timing['timestamp_domain'] != clock_domain:
                    raise RuntimeError('Camera clock domain changed; restart a separate capture')
                clock_domain = timing['timestamp_domain']
                if last_timestamp is not None and timestamp <= last_timestamp:
                    raise RuntimeError('Camera timestamp repeated or moved backwards; refuse stale input')
                last_timestamp = timestamp
                if timestamp_origin is None:
                    timestamp_origin = timestamp
                mask = detect_mask(bgr, args.detector, args.marker_id, args.hsv_low, args.hsv_high, args.roi)
                point, quality = depth_measurement(mask, depth, deproject, args.min_depth, args.max_depth)
                if point is not None and transform is not None:
                    point = transform_point(transform, point)
                state = tracker.update(timestamp - timestamp_origin, point)
                hover = (None if args.hover_height is None else
                         hover_preview_target(state, args.hover_height, args.hover_lookahead))
                if hover is not None:
                    # Host arrival excludes unknown camera/USB transit latency.
                    if time.monotonic() - timing['host_arrival_monotonic_s'] > .12:
                        hover = {**hover, 'target_m': None, 'reason': 'host_processing_stale'}
                    hover_targets += int(hover['target_m'] is not None)
                processing = time.monotonic() - begin
                process_times.append(processing)
                frames += 1
                accepted += int(state is not None and state.accepted)
                ready += int(state is not None and state.ready)
                record = {'index': frames - 1, **timing, 'processing_s': processing,
                          'measurement_m': None if point is None else point.tolist(),
                          'quality': quality, 'track': None if state is None else state.as_dict(),
                          'action': None, 'motion_sent': False}
                if hover is not None:
                    record['hover_preview'] = hover
                if args.save_rgbd:
                    name = f'frame_{frames - 1:06d}.npz'
                    np.savez_compressed(root / name, bgr=bgr, depth_m=depth.astype(np.float32))
                    record['rgbd_file'] = name
                log.write(json.dumps(record, allow_nan=False) + '\n')
                last_view = overlay.draw(bgr, mask, quality, state, timing, processing,
                                         transform, args.roi, frame=frame, hover=hover)
                if args.live_frame and frames % max(1, args.fps) == 0:
                    # Atomic publication lets an external viewer inspect the latest image.
                    cv2.imwrite(str(root / 'live_frame.tmp.jpg'), last_view)
                    (root / 'live_frame.tmp.jpg').replace(root / 'live_frame.jpg')
                if args.preview:
                    cv2.imshow('CPU tracking only - no robot motion', last_view)
                    if cv2.waitKey(1) & 0xff == ord('q'):
                        break
        complete = True
    finally:
        if camera is not None:
            camera.close()
        if args.preview:
            cv2.destroyAllWindows()
        if last_view is not None:
            cv2.imwrite(str(root / 'last_frame.jpg'), last_view)
        summary = {'complete': complete, 'frames': frames, 'accepted_measurements': accepted,
                   'ready_frames': ready, 'elapsed_s': time.monotonic() - start,
                   'hover_preview_targets': hover_targets,
                   'processing_p50_ms': None if not process_times else float(1000 * np.median(process_times)),
                   'processing_p95_ms': None if not process_times else float(1000 * np.percentile(process_times, 95)),
                   'expert_episode': False, 'gpu_used': False, 'motion_sent': False}
        write_json(root / 'summary.json', summary)
        print(json.dumps(summary, indent=2))


def calibrate(args):
    data = json.loads(Path(args.landmarks).read_text())
    if data.get('arm') not in ('left', 'right'):
        raise ValueError('Landmarks must identify arm as left or right before fitting')
    t, errors = fit_rigid_transform(data['camera_points_m'], data['base_points_m'])
    if not np.isfinite(args.max_error) or args.max_error <= 0:
        raise ValueError('max-error must be finite and positive')
    validation_errors = []
    a, b = data.get('validation_camera_points_m', []), data.get('validation_base_points_m', [])
    if len(a) != len(b):
        raise ValueError('Validation landmark counts differ')
    for camera, base in zip(a, b):
        base = np.asarray(base, float)
        if base.shape != (3,) or not np.isfinite(base).all():
            raise ValueError('Invalid validation landmark')
        validation_errors.append(float(np.linalg.norm(transform_point(t, camera) - base)))
    validated = (len(validation_errors) >= 3 and max(validation_errors) <= args.max_error
                 and max(errors) <= args.max_error and data.get('static_camera') is True)
    result = {'camera_serial': data['camera_serial'], 'destination_frame': 'robot_base',
              'arm': data['arm'],
              'static_camera': data.get('static_camera') is True,
              'T_base_camera': t.tolist(), 'fit_errors_m': errors.tolist(),
              'heldout_errors_m': validation_errors, 'threshold_m': args.max_error,
              'calibration_validated': bool(validated),
              'note': 'Validity also depends on independent landmarks, unchanged mounting, and metric base coordinates.'}
    with Path(args.output).open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


def demo(args):
    root = new_run(args.output)
    rng = np.random.default_rng(42)
    tracker = VelocityTracker(frame='robot_base')
    controller = ShadowHoverController(HoverLimits([0, -.3, .01], [.5, .3, .4]))
    ee = np.array([.2, -.08, .12])
    errors, candidates = [], 0
    with (root / 'tracking.jsonl').open('w') as out:
        for i in range(240):
            t = i / 30
            truth = np.array([.2 + .03 * np.sin(t), .01 * t, .04])
            velocity = np.array([.03 * np.cos(t), .01, 0])
            z = truth + rng.normal(0, .002, 3)
            if 90 <= i < 104:
                z = None
            if i == 160:
                z = truth + [.5, .5, .5]
            state = tracker.update(t, z)
            command = controller.step(state, ee, t, 1 / 30)
            if command['status'] == 'candidate':
                ee = np.array(command['candidate_next_position_m'])
                candidates += 1
            if state is not None and state.ready:
                errors.append(np.linalg.norm(state.velocity_mps - velocity))
            out.write(json.dumps({'timestamp_s': t, 'synthetic': True,
                                 'track': None if state is None else state.as_dict(),
                                 'controller': command}, allow_nan=False) + '\n')
    summary = {'synthetic': True, 'frames': 240, 'candidate_steps': candidates,
               'velocity_rmse_mps_on_ready_frames': float(np.sqrt(np.mean(np.square(errors)))),
               'gpu_used': False, 'motion_sent': False, 'expert_episode': False}
    write_json(root / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description='CPU tracking and shadow-control prototype; no robot motion commands')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('devices', help='Enumerate RealSense devices without motor access')
    p = commands.add_parser('track', help='Record stationary-camera metric position and velocity')
    p.add_argument('--serial', required=True)
    p.add_argument('--arm', choices=['left', 'right'], default='right', help='Target arm for calibration matching; default: right')
    p.add_argument('--output', required=True, help='New directory, preferably under scratch/real_robot')
    p.add_argument('--fixed-camera', action='store_true')
    p.add_argument('--seconds', type=float, default=30)
    p.add_argument('--fps', type=int, default=15)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--detector', choices=['aruco', 'hsv', 'orange_ball', 'blue_tape'], default='aruco')
    p.add_argument('--marker-id', type=int, default=0)
    p.add_argument('--hsv-low', nargs=3, type=int, help='Default blue_tape: 90 60 45; orange_ball: 3 100 70; hsv: 0 120 80')
    p.add_argument('--hsv-high', nargs=3, type=int, help='Default blue_tape: 135 255 255; orange_ball: 25 255 255; hsv: 10 255 255')
    p.add_argument('--roi', nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'))
    p.add_argument('--min-depth', type=float, default=.1)
    p.add_argument('--max-depth', type=float, default=1.5)
    frame_options = p.add_mutually_exclusive_group()
    frame_options.add_argument('--calibration', help='Validated static camera-to-base JSON')
    frame_options.add_argument('--provisional-landmarks', help='Operator landmarks for an explicitly unvalidated base-frame display estimate; camera mounting must be unchanged')
    frame_options.add_argument('--candidate-calibration', help='Provisional right-arm ChArUco fit, visualization only')
    p.add_argument('--hover-height', type=float, help='Preview target this many meters above object along base +Z; no motor commands')
    p.add_argument('--hover-lookahead', type=float, default=.10, help='Causal target prediction horizon, seconds')
    p.add_argument('--live-frame', action='store_true', help='Publish live_frame.jpg once per second in output directory')
    p.add_argument('--preview', action='store_true')
    p.add_argument('--disable-stationary-update', action='store_true', help='Disable inferred zero-velocity updates for comparison')
    p.add_argument('--save-rgbd', action='store_true', help='Save aligned arrays for replay; adds CPU/disk overhead')
    p = commands.add_parser('calibrate', help='Fit a rigid camera-to-base transform from measured landmarks')
    p.add_argument('--landmarks', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--max-error', type=float, default=.005)
    p = commands.add_parser('collect-landmarks', help='Click fixed physical landmarks and enter independently measured base XYZ; no motor connection')
    p.add_argument('--serial', required=True)
    p.add_argument('--arm', choices=['left', 'right'], default='right')
    p.add_argument('--output', required=True, help='New calibration session directory')
    p.add_argument('--resume-from', help='Copy an existing landmarks.json into a new session; camera mounting and reference axes must be unchanged')
    p.add_argument('--fixed-camera', action='store_true', required=True)
    p = commands.add_parser('demo', help='Synthetic tracker/controller test; not robot data')
    p.add_argument('--output', required=True)
    p = commands.add_parser('marker', help='Generate an ArUco marker to attach to the test object')
    p.add_argument('--output', required=True)
    p.add_argument('--id', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'devices':
        from .camera import enumerate_cameras
        print(json.dumps(enumerate_cameras(), indent=2))
    elif args.command == 'track':
        track(args)
    elif args.command == 'calibrate':
        calibrate(args)
    elif args.command == 'collect-landmarks':
        from .calibration_capture import collect_landmarks
        collect_landmarks(args)
    elif args.command == 'demo':
        demo(args)
    elif args.command == 'marker':
        import cv2
        if not 0 <= args.id < 50:
            raise ValueError('Marker id must be 0..49')
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        image = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), args.id, 400)
        image = cv2.copyMakeBorder(image, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)
        if not cv2.imwrite(args.output, image):
            raise OSError('Could not save marker image')


if __name__ == '__main__':
    main()
