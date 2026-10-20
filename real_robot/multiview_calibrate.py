"""Camera-only front/side ChArUco capture and held-out relative calibration."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .multiview import CameraPair, FRONT_SERIAL, SIDE_SERIAL, charuco_detector, estimate_board, fit_relative, pose_difference
from .geometry import validate_transform


def steady(history):
    if len(history) < 8 or history[-1][0]-history[0][0] < .6: return False
    for role in ('front', 'side'):
        common = set(history[0][1][role]['ids'])
        for _, row in history: common &= set(row[role]['ids'])
        if len(common) < 12: return False
        arrays = []
        for _, row in history:
            r = row[role]; points = dict(zip(r['ids'], r['pixels']))
            arrays.append([points[i] for i in sorted(common)])
        arrays = np.array(arrays)
        # Bound motion over the complete window, not just adjacent frames.
        rms = np.sqrt(np.mean(np.sum((arrays-arrays.mean(0))**2, axis=2), axis=1))
        if max(rms) > 1.2: return False
    return True


def novel(row, observations):
    for old in observations:
        distance, angle = pose_difference(row['front']['T_camera_board'], old['front']['T_camera_board'])
        if distance < .03 and angle < 10: return False
    return True


def write(path, data): path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def build_rig(root, observations, candidate=None, excluded_indices=(), exclusion_reason=None):
    excluded = list(excluded_indices)
    if (len(set(excluded)) != len(excluded) or any(i < 0 or i >= len(observations) for i in excluded)
            or (excluded and not exclusion_reason)):
        raise ValueError('Excluded pose indices must be unique, in range, and have a documented reason')
    eligible = [i for i in range(len(observations)) if i not in excluded]
    if len(eligible) < 7: raise ValueError('Need at least seven retained distinct board poses')
    heldout_count = min(max(2, len(eligible)//3), len(eligible)-5)
    fit, heldout = eligible[:-heldout_count], eligible[-heldout_count:]
    result = fit_relative(observations, fit, heldout)
    rig = {**result, 'schema': 'front_side_rig_v1', 'front_serial': FRONT_SERIAL, 'side_serial': SIDE_SERIAL,
           'fixed_cameras': True, 'output_frame': 'front_camera_optical', 'base_calibration_validated': False,
           'calibration_validated': False, 'motion_enabled': False,
           'board': {'squares': [6, 6], 'square_m': .05, 'marker_m_provisional': .0375,
                     'dictionary': 'DICT_4X4_50'},
           'position_semantics': 'visible_surface_proxy; not a volumetric object centroid',
           'timestamp_note': 'Global-time timestamp pairing; exposures are not hardware synchronized.',
           'sources': [f'pose_{i:03d}.json' for i in range(len(observations))]}
    rig.update(excluded_indices=excluded, exclusion_reason=exclusion_reason)
    rig['camera_profiles'] = {role: {key: observations[eligible[0]][role]['timing'][key]
                                   for key in ('camera_serial', 'pyrealsense2_version', 'intrinsics', 'distortion_model')}
                              for role in ('front', 'side')}
    if candidate:
        raw = Path(candidate).read_bytes(); c = json.loads(raw)
        if (c.get('arm') != 'right' or c.get('front_camera_serial') != FRONT_SERIAL
                or c.get('calibration_validated') is not False):
            raise ValueError('Base reference must match the provisional right-front ChArUco candidate')
        base = validate_transform(c['T_right_base_front_camera_candidate'])
        rig.update(T_base_front_candidate=base.tolist(),
                   T_base_side_candidate=(base@np.array(result['T_front_side'])).tolist(),
                   output_frame='right_base_estimate', base_reference_sha256=hashlib.sha256(raw).hexdigest(),
                   operator_confirmed_front_and_base_unchanged=True)
    write(root/'rig.json', rig)
    return rig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--poses', type=int, default=9)
    parser.add_argument('--seconds', type=float, default=900)
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--base-candidate', help='Use only after operator confirms fixed front/base unchanged')
    parser.add_argument('--fit-only', action='store_true', help='Fit existing pose files without opening cameras')
    parser.add_argument('--exclude-poses', type=int, nargs='*', default=[], help='Zero-based pose indices with independently documented capture problems')
    parser.add_argument('--exclusion-reason')
    args = parser.parse_args(); root = Path(args.output)
    if args.fit_only:
        rows = [json.loads(p.read_text()) for p in sorted(root.glob('pose_*.json'))]
        print(json.dumps(build_rig(root, rows, args.base_candidate, args.exclude_poses, args.exclusion_reason), indent=2)); return
    if args.poses < 7 or not 0 < args.seconds <= 1800: raise ValueError('Require >=7 poses and <=1800 seconds')
    root.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 'front_side_charuco_capture_v1', 'front_serial': FRONT_SERIAL, 'side_serial': SIDE_SERIAL,
                'requested_poses': args.poses, 'auto': args.auto, 'motion_sent': False,
                'board': {'squares': [6, 6], 'square_m': .05, 'marker_m_provisional': .0375},
                'pair_span_limit_s': .05, 'stationary_board_capture_only': True}
    write(root/'manifest.json', manifest)
    board, detector = charuco_detector(); observations = []; history = deque()
    cameras = CameraPair(max_skew_s=.05).start(); start = time.monotonic(); pair_count = 0
    last_rows = {}; message = 'Show printed board face in BOTH views; hold still for one second'
    complete = False; error = None
    cv2.setNumThreads(1)
    try:
        with (root/'pair-quality.jsonl').open('w', buffering=1) as log:
            while time.monotonic()-start < args.seconds and len(observations) < args.poses:
                pair = cameras.poll(); can_save = False; row = None
                if pair:
                    front, side, timing = pair; pair_count += 1
                    row = {'front': estimate_board(front, board, detector),
                           'side': estimate_board(side, board, detector), 'pair': timing}
                    last_rows = row
                    log.write(json.dumps({'pair': timing, 'front_corners': row['front']['corner_count'],
                                          'side_corners': row['side']['corner_count'],
                                          'both_valid': row['front']['valid'] and row['side']['valid']})+'\n')
                    if row['front']['valid'] and row['side']['valid']:
                        history.append((time.monotonic(), row))
                        while history and time.monotonic()-history[0][0] > 1.:
                            history.popleft()
                        can_save = steady(history) and novel(row, observations)
                    else: history.clear()
                views = cameras.views(); panels = []
                for role in ('front', 'side'):
                    image = views[role]['bgr'].copy() if role in views else np.zeros((480, 640, 3), np.uint8)
                    detection = last_rows.get(role, {})
                    if detection.get('ids'):
                        cv2.aruco.drawDetectedCornersCharuco(image, np.array(detection['pixels'], np.float32).reshape(-1, 1, 2),
                                                            np.array(detection['ids'], np.int32).reshape(-1, 1))
                    cv2.putText(image, f'{role.upper()} | corners {detection.get("corner_count", 0)}/25',
                                (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                    panels.append(image)
                footer = np.zeros((110, 1280, 3), np.uint8)
                labels = [f'CAMERA ONLY | saved {len(observations)}/{args.poses} | '+('STEADY / NEW POSE' if can_save else 'waiting for steady new pose'),
                          'Hold still 1s; after a save move board 5-10 cm or tilt 10-20 degrees; keep printed face visible in both views',
                          message, 'AUTO capture enabled | S manual save | Q finish | no robot commands' if args.auto else 'S save | Q finish | no robot commands']
                for i, label in enumerate(labels):
                    cv2.putText(footer, label, (8, 23+25*i), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
                view = np.vstack([np.hstack(panels), footer])
                cv2.imshow('Front + side ChArUco - camera only', view)
                key = cv2.waitKey(1) & 0xff
                if key == ord('q'): break
                if can_save and row is not None and (args.auto or key == ord('s')):
                    i = len(observations)
                    for role, frame in [('front', front), ('side', side)]:
                        name = f'pose_{i:03d}_{role}'
                        cv2.imwrite(str(root/(name+'.jpg')), frame['bgr'])
                        np.savez_compressed(root/(name+'.npz'), bgr=frame['bgr'], depth_m=frame['depth'].astype(np.float32))
                        row[role]['rgbd_file'] = name+'.npz'
                    write(root/f'pose_{i:03d}.json', row); observations.append(row); history.clear()
                    message = f'Saved pose {i+1}; move/tilt board to a different pose, then hold still'
                    print(message, flush=True)
                if not views and time.monotonic()-start > 30: raise RuntimeError('No camera frames after 30 seconds')
                status = {'saved_poses': len(observations), 'pairs': pair_count,
                          'front_corners': last_rows.get('front', {}).get('corner_count', 0),
                          'side_corners': last_rows.get('side', {}).get('corner_count', 0),
                          'message': message, 'motion_sent': False, 'dropped_frames': cameras.pairer.dropped}
                write(root/'status.tmp.json', status); (root/'status.tmp.json').replace(root/'status.json')
                if pair_count % 15 == 0 or pair is None:
                    cv2.imwrite(str(root/'live_frame.tmp.jpg'), view); (root/'live_frame.tmp.jpg').replace(root/'live_frame.jpg')
                if pair is None: time.sleep(.01)
        complete = len(observations) >= args.poses
    except BaseException as exc:
        error = str(exc); raise
    finally:
        cameras.close(); cv2.destroyAllWindows()
        write(root/'capture-summary.json', {'complete': complete, 'saved_poses': len(observations),
                                            'pairs': pair_count, 'error': error, 'motion_sent': False})
    if len(observations) >= 7:
        print('Fitting relative camera geometry with held-out board poses...', flush=True)
        rig = build_rig(root, observations, args.base_candidate, args.exclude_poses, args.exclusion_reason)
        print(json.dumps(rig, indent=2), flush=True)


if __name__ == '__main__': main()
