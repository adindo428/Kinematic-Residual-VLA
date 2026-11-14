"""Camera-only exposed/covered chessboard depth comparison; no calibration writes."""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .base_calibration import pose_difference, write
from .calibration_board import Chessboard, ChessboardDetector
from .multiview import estimate_board

WINDOW = 'Camera texture diagnostic - no robot control'


def perimeter_pixels(detection, board):
    cols, rows = board.pattern_size
    ids = [r * cols + c for r in range(rows) for c in range(cols)
           if r in (0, rows - 1) or c in (0, cols - 1)]
    return np.float32([detection['pixels'][detection['ids'].index(i)] for i in ids])


def check_perimeter(original, current, pixels):
    """Image consistency, not proof of unchanged 3-D pose or calibration accuracy."""
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    now = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    seed = np.asarray(pixels, np.float32).reshape(-1, 1, 2)
    moved, status, _ = cv2.calcOpticalFlowPyrLK(gray, now, seed, None, winSize=(11, 11), maxLevel=1)
    if moved is None or not np.all(status):
        raise ValueError('Outer corners obscured; clear hands and leave the outermost intersections visible')
    back, status, _ = cv2.calcOpticalFlowPyrLK(now, gray, moved, None, winSize=(11, 11), maxLevel=1)
    if back is None or not np.all(status) or np.max(np.linalg.norm(back-seed, axis=2)) > .2:
        raise ValueError('Outer-corner tracking is inconsistent; clear hands and shadows')
    displacement = float(np.max(np.linalg.norm(moved-seed, axis=2)))
    if displacement > .7:
        raise ValueError(f'Board image moved {displacement:.2f} px; restore its position or restart')
    for a, b in zip(seed.reshape(-1, 2), moved.reshape(-1, 2)):
        first = cv2.getRectSubPix(gray, (7, 7), tuple(a)).astype(float).ravel()
        second = cv2.getRectSubPix(now, (7, 7), tuple(b)).astype(float).ravel()
        first -= first.mean(); second -= second.mean()
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        if denominator < 1 or float(first @ second / denominator) < .90:
            raise ValueError('Outer-corner appearance changed; keep paper central and clear hands/shadows')
    return dict(max_displacement_px=displacement, checked_corners=len(seed))


def plane_sample(frame, pose, pixel, radius=3):
    """Compare depth Z with a frozen RGB plane at each sampled pixel's own ray."""
    pose = np.asarray(pose, float)
    normal = pose[:3, 2]
    plane_distance = float(normal @ pose[:3, 3])
    x, y = np.rint(pixel).astype(int)
    height, width = frame['depth'].shape
    if x-radius < 0 or y-radius < 0 or x+radius >= width or y+radius >= height:
        raise ValueError('Center sample falls outside the depth image')
    depths, predicted, residuals = [], [], []
    for v in range(y-radius, y+radius+1):
        for u in range(x-radius, x+radius+1):
            z = float(frame['depth'][v, u])
            if not np.isfinite(z) or z <= 0:
                continue
            ray = np.asarray(frame['deproject']([float(u), float(v)], 1.), float)
            divisor = float(normal @ ray)
            if not np.isfinite(divisor) or abs(divisor) < 1e-8:
                raise ValueError('Invalid board/ray geometry')
            expected_z = plane_distance / divisor * ray[2]
            if not np.isfinite(expected_z) or expected_z <= 0:
                raise ValueError('Board plane is behind the camera')
            depths.append(z); predicted.append(expected_z); residuals.append(z-expected_z)
    if len(depths) < .8 * (2*radius+1)**2:
        raise ValueError('Insufficient valid center depth; keep hands clear and target flat')
    return dict(pixel=list(map(float, pixel)), valid_pixels=len(depths),
                median_depth_z_mm=float(np.median(depths)*1000),
                median_rgb_plane_z_mm=float(np.median(predicted)*1000),
                median_signed_z_error_mm=float(np.median(residuals)*1000),
                p90_abs_z_error_mm=float(np.percentile(np.abs(residuals), 90)*1000))


def summarize(records):
    result = {stage: dict(frames=len(records[stage]), median_signed_z_error_mm=float(np.median([
        r['sample']['median_signed_z_error_mm'] for r in records[stage]])))
        for stage in ('exposed', 'covered')}
    result['covered_minus_exposed_z_error_mm'] = (result['covered']['median_signed_z_error_mm']
                                                - result['exposed']['median_signed_z_error_mm'])
    result['note'] = ('Diagnostic only. Image consistency cannot prove a fixed 3-D plane; paper must lie flat '
                      'without moving the board. This does not validate a rig or establish a global depth correction.')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=12)
    args = parser.parse_args(argv)
    if not 3 <= args.frames <= 30:
        parser.error('frames must be 3..30')
    manifest = json.loads((args.session/'manifest.json').read_text())
    if manifest.get('board_type') != 'chessboard' or any(n % 2 for n in manifest['board_squares']):
        parser.error('Requires a plain chessboard with even square counts and a central intersection')
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    from .camera import RGBDCamera
    from .recording_lease import LeaseOwner
    cv2.setNumThreads(1)
    board = Chessboard(manifest['board_squares'], manifest['square_mm']/1000)
    detector = ChessboardDetector(board)
    camera = lease = None
    records = dict(exposed=[], covered=[])
    report = dict(schema='krvla_camera_texture_check_v1', source_session=str(args.session.resolve()),
                  robot_ports_opened=False, calibration_parameters_changed=False,
                  board_squares=manifest['board_squares'], square_mm=manifest['square_mm'],
                  complete=False, validated=False)
    stage = 'reference'; reference = None; error = None
    last_detection = None; stable_count = 0; capture_covered = False
    try:
        lease = LeaseOwner(Path(__file__).resolve().parents[1]/'scratch/real_robot/active-collection.json')
        camera = RGBDCamera(manifest['rig']['front_serial'], fps=15)
        print('No robot control. Full board clear and stationary: B locks a reference and samples exposed depth.', flush=True)
        print('Then lay a ~10 x 10 cm piece of thin matte paper with irregular text/marks flat over the center.', flush=True)
        print('Keep outer intersections visible, board fixed, hands clear. S samples covered depth. Q quits.', flush=True)
        while True:
            bgr, depth, timing, deproject = camera.read()
            if not 0 <= time.monotonic()-timing['host_arrival_monotonic_s'] <= .2:
                continue
            frame = dict(bgr=bgr, depth=depth, timing=timing, deproject=deproject)
            detection = None; ready = False; sample = None; perimeter = None
            message = 'Show complete board, clear hands; building stable reference'
            try:
                if stage in ('reference', 'exposed'):
                    detection = estimate_board(frame, board, detector)
                    if not detection.get('valid') or detection['corner_count'] != np.prod(board.pattern_size):
                        stable_count = 0
                        raise ValueError('Full board required; clear hands and paper')
                    if stage == 'reference':
                        distance, angle = ((1., 180.) if last_detection is None else pose_difference(
                            last_detection['T_camera_board'], detection['T_camera_board']))
                        stable_count = stable_count+1 if distance <= .002 and angle <= 1 else 0
                        last_detection = detection
                        ready = stable_count >= 8
                        message = 'BOARD READY: B locks reference' if ready else message
                    else:
                        distance, angle = pose_difference(reference['pose'], detection['T_camera_board'])
                        if distance > .002 or angle > 1:
                            raise ValueError('Board pose changed during exposed sampling; restart')
                if reference is not None:
                    perimeter = check_perimeter(reference['bgr'], bgr, reference['perimeter_pixels'])
                    sample = plane_sample(frame, reference['pose'], reference['pixel'])
                    message = (f'{stage}: center depth Z error {sample["median_signed_z_error_mm"]:+.1f} mm; '
                               f'{len(records[stage])}/{args.frames} frames')
                    if stage == 'covered' and not capture_covered:
                        message = 'Lay marked paper flat over center; clear hands; S samples, Q quits'
                    if stage == 'exposed' or capture_covered:
                        records[stage].append(dict(sample=sample, perimeter=perimeter, timing=timing))
                        if len(records[stage]) == 1:
                            cv2.imwrite(str(root/f'{stage}.jpg'), bgr)
                            np.savez(root/f'{stage}.npz', bgr=bgr, depth_m=depth)
                        if len(records[stage]) == args.frames:
                            if stage == 'exposed':
                                stage = 'covered'
                                print('Exposed samples saved. Place central paper without moving board, then press S with hands clear.', flush=True)
                            else:
                                report['summary'] = summarize(records); report['complete'] = True
                                print(json.dumps(report['summary'], indent=2), flush=True)
                                break
            except ValueError as exc:
                message = str(exc)
                ready = False
                if stage == 'reference':
                    last_detection = None
                elif stage == 'exposed' or capture_covered:
                    records[stage].clear()  # Require one uninterrupted block of valid samples.
            preview = bgr.copy()
            if reference is not None:
                cv2.circle(preview, tuple(np.rint(reference['pixel']).astype(int)), 8, (0, 0, 255), 2)
                for p in reference['perimeter_pixels']:
                    cv2.circle(preview, tuple(np.rint(p).astype(int)), 2, (0, 255, 0), -1)
            cv2.putText(preview, message[:100], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 0, 255), 1)
            cv2.imshow(WINDOW, preview)
            key = cv2.waitKey(1) & 255
            if key in (ord('q'), 27):
                report['cancelled'] = True
                break
            if key == ord('b') and stage == 'reference' and ready:
                cols, rows = board.pattern_size
                center_id = (rows//2)*cols+cols//2
                reference = dict(pose=detection['T_camera_board'], bgr=bgr.copy(),
                                 pixel=detection['pixels'][detection['ids'].index(center_id)],
                                 perimeter_pixels=perimeter_pixels(detection, board))
                report['reference'] = dict(detection=detection, center_corner_id=center_id,
                                           center_pixel=reference['pixel'])
                cv2.imwrite(str(root/'reference.jpg'), bgr)
                stage = 'exposed'
            if key == ord('s') and stage == 'covered' and sample is not None and perimeter is not None:
                report['operator_asserted_paper_flat_and_board_unchanged'] = True
                capture_covered = True
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report.update(records=records, error=error)
        write(root/'texture_check.json', report)
        try:
            if camera is not None:
                camera.close()
        finally:
            if lease is not None:
                lease.close()
            cv2.destroyAllWindows()
        print(f'Texture diagnostic saved: {root/"texture_check.json"}', flush=True)


if __name__ == '__main__':
    main()
