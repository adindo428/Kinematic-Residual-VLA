"""Interactive measured-landmark capture. Never connects to a motor bus."""
import json
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np


def clicked_point(depth, pixel, deproject):
    """Use a small continuous surface patch, deprojecting the clicked ray."""
    x, y = map(int, pixel)
    if not (2 <= x < depth.shape[1] - 2 and 2 <= y < depth.shape[0] - 2):
        raise ValueError('Click at least two pixels inside the image')
    patch = depth[y-2:y+3, x-2:x+3]
    valid = patch[np.isfinite(patch) & (patch >= .1) & (patch <= 1.5)]
    if len(valid) < 20 or not np.isfinite(depth[y, x]) or not .1 <= depth[y, x] <= 1.5:
        raise ValueError('Insufficient depth at the clicked point')
    # A hole or object boundary must not masquerade as a precise landmark.
    if np.ptp(valid) > .01:
        raise ValueError('Depth discontinuity: choose a continuous surface away from edges')
    point = np.asarray(deproject([x, y], float(np.median(valid))), float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('Invalid deprojected point')
    return point


def landmark_document(serial, records, arm='right'):
    if arm not in ('left', 'right'):
        raise ValueError('Arm must be left or right')
    result = {'camera_serial': serial, 'static_camera': True,
              'robot': 'so101_follower', 'arm': arm,
              'calibration_validated': False, 'motion_sent': False,
              'coordinate_note': 'Base XYZ entered by operator in meters, independently measured for the same physical point.',
              'camera_points_m': [], 'base_points_m': [],
              'validation_camera_points_m': [], 'validation_base_points_m': [],
              'observations': records}
    for record in records:
        prefix = 'validation_' if record['partition'] == 'validation' else ''
        result[prefix + 'camera_points_m'].append(record['camera_point_m'])
        result[prefix + 'base_points_m'].append(record['base_point_m'])
    return result


def prepare_landmark_session(output, serial, arm, resume_from=None):
    """Copy a session to a new directory, leaving the source evidence intact."""
    root = Path(output)
    records, provenance, source = [], {}, None
    if resume_from:
        source = Path(resume_from)
        raw = source.read_bytes()
        data = json.loads(raw)
        if (data.get('camera_serial') != serial or data.get('arm') != arm
                or data.get('static_camera') is not True):
            raise ValueError('Resume source must match the fixed camera and selected arm')
        records = data['observations']
        reconstructed = landmark_document(serial, records, arm)
        for key in ['camera_points_m', 'base_points_m', 'validation_camera_points_m', 'validation_base_points_m']:
            if data.get(key) != reconstructed[key]:
                raise ValueError('Resume landmark arrays disagree with recorded observations')
        for record in records:
            if record['partition'] not in ('fit', 'validation'):
                raise ValueError('Invalid landmark partition')
            for key in ['camera_point_m', 'base_point_m']:
                point = np.asarray(record[key], float)
                if point.shape != (3,) or not np.isfinite(point).all():
                    raise ValueError('Invalid saved landmark point')
            filename = record['rgbd_file']
            if Path(filename).name != filename or not (source.parent/filename).is_file():
                raise ValueError('Resume observation must reference an existing local RGB-D file')
        provenance = {'resumed_from_sha256': hashlib.sha256(raw).hexdigest(),
                      'resumed_observations': len(records)}
    root.mkdir(parents=True, exist_ok=False)
    if source:
        (root/'source_landmarks.json').write_bytes(raw)
        for record in records:
            src, dst = source.parent/record['rgbd_file'], root/record['rgbd_file']
            shutil.copy2(src, dst)
            if hashlib.sha256(src.read_bytes()).digest() != hashlib.sha256(dst.read_bytes()).digest():
                raise OSError('Resumed RGB-D copy checksum mismatch')
    return records, provenance


def atomic_json(path, document):
    path = Path(path)
    with path.with_suffix('.tmp').open('w') as stream:
        stream.write(json.dumps(document, indent=2, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    path.with_suffix('.tmp').replace(path)


def collect_landmarks(args):
    import cv2
    from .camera import RGBDCamera
    cv2.setNumThreads(1)
    root = Path(args.output)
    records, provenance = prepare_landmark_session(root, args.serial, args.arm, args.resume_from)
    recovery = Path(tempfile.mkdtemp(prefix='dynamicvla-landmark-recovery-'))
    clicks = []
    partition, frozen, camera = 'fit', None, None
    title = f'{args.arm.capitalize()} arm camera calibration - no robot motion'

    def save():
        document = {**landmark_document(args.serial, records, args.arm), **provenance}
        atomic_json(recovery/'landmarks.json', document)
        atomic_json(root/'landmarks.json', document)

    def mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and frozen is not None:
            clicks.append((x, y))

    save()
    print(f'Resumed {len(records)} points. Internal temporary metadata backup: {recovery}', flush=True)
    print(f'Selected arm: {args.arm}. Keep camera/base and landmarks fixed. F freezes a frame; click a known surface point, then enter its base XYZ in METERS in this terminal.', flush=True)
    print('V switches fit/held-out validation; U undoes last point; Q ends. Collect >=4 spread-out fit points and >=3 distinct held-out points.', flush=True)
    print('Do not guess base coordinates or use the tape-hole/table point as a tape-surface landmark.', flush=True)
    try:
        camera = RGBDCamera(args.serial)
        cv2.namedWindow(title)
        cv2.setMouseCallback(title, mouse)
        while True:
            current = camera.read() if frozen is None else frozen
            bgr, depth, timing, deproject = current
            view = bgr.copy()
            n_fit = sum(r['partition'] == 'fit' for r in records)
            status = 'FROZEN: click point' if frozen is not None else 'LIVE: F to freeze'
            cv2.putText(view, f'{status} | {partition} | fit {n_fit}, held-out {len(records)-n_fit}',
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
            cv2.imshow(title, view)
            key = cv2.waitKey(20) & 0xff
            if key == ord('q'):
                break
            if key == ord('f'):
                frozen = current if frozen is None else None
                clicks.clear()
            if key == ord('v'):
                partition = 'validation' if partition == 'fit' else 'fit'
            if key == ord('u') and records:
                records.pop(); save()
            if clicks and frozen is not None:
                pixel = clicks.pop(0)
                clicks.clear()
                try:
                    point = clicked_point(depth, pixel, deproject)
                    raw = input(f'{partition} camera XYZ {point.round(4).tolist()}; corresponding {args.arm.upper()} BASE x y z (meters), blank to skip: ').strip()
                    if not raw:
                        continue
                    base = np.array([float(s) for s in raw.split()])
                    if base.shape != (3,) or not np.isfinite(base).all():
                        raise ValueError('Enter exactly three finite coordinates in meters')
                    # Each observation has its frozen image/depth and timing evidence.
                    from uuid import uuid4
                    filename = f'landmark_{uuid4().hex}.npz'
                    record = {'partition': partition, 'pixel': list(pixel),
                              'camera_point_m': point.tolist(), 'base_point_m': base.tolist(),
                              'capture': timing, 'rgbd_file': filename}
                    # Retain exact coordinates even if the external SSD disconnects
                    # before its image file can be committed.
                    atomic_json(recovery/'pending_landmark.json', {**record, 'rgbd_saved': False})
                    np.savez_compressed(root / filename, bgr=bgr, depth_m=depth.astype(np.float32))
                    records.append(record)
                    save()
                    (recovery/'pending_landmark.json').unlink()
                    print(f'Saved {len(records)} landmarks to {root / "landmarks.json"}', flush=True)
                    frozen = None
                except ValueError as exc:
                    print(f'Point not recorded: {exc}', flush=True)
    except (KeyboardInterrupt, EOFError):
        print('\nCapture ended; recorded landmarks retained.', flush=True)
    finally:
        if camera is not None:
            camera.close()
        cv2.destroyAllWindows()
        try:
            save()
        except OSError as exc:
            print(f'Could not save to SSD: {exc}. Metadata recovery copy: {recovery}', flush=True)
    print(f'Landmarks saved to {root / "landmarks.json"}. Transform has NOT been fitted or validated.', flush=True)
