"""Timestamp-gated front/side RGB-D position-proxy fusion; no robot commands."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .geometry import validate_transform, transform_point
from .multiview import CameraPair, FRONT_SERIAL, SIDE_SERIAL, ProxyAssociation
from .perception import detect_mask, depth_measurement
from .tracking import VelocityTracker
from .velocity_report import velocity_report
from .visible_extent import visible_top
from .rgbd_archive import RGBDArchiveWriter


def load_rig(path):
    raw = Path(path).read_bytes(); rig = json.loads(raw)
    if (rig.get('schema') != 'front_side_rig_v1' or rig.get('front_serial') != FRONT_SERIAL
            or rig.get('side_serial') != SIDE_SERIAL or rig.get('fixed_cameras') is not True
            or rig.get('relative_calibration_validated') is not True):
        raise ValueError('Require a matching fixed front/side rig that passed held-out relative calibration checks')
    fs = validate_transform(rig['T_front_side'])
    if not .05 < np.linalg.norm(fs[:3, 3]) < 3:
        raise ValueError('Unexpected front/side baseline; inspect relative calibration')
    base = None
    if 'T_base_front_candidate' in rig:
        if (rig.get('output_frame') != 'right_base_estimate'
                or rig.get('operator_confirmed_front_and_base_unchanged') is not True
                or rig.get('base_calibration_validated') is not False):
            raise ValueError('Base candidate must remain explicitly provisional and mounting-confirmed')
        base = validate_transform(rig['T_base_front_candidate'])
        if not np.allclose(base@fs, validate_transform(rig['T_base_side_candidate']), atol=1e-7):
            raise ValueError('Front/side/base transform composition disagrees')
    elif rig.get('output_frame') != 'front_camera_optical':
        raise ValueError('Rig has no transform into its named output frame')
    return rig, fs, base, hashlib.sha256(raw).hexdigest()


def live_observation(fusion, views, last_pair_host_s, now):
    """Expire the published coordinates even when no further pairs arrive."""
    stale = (set(views) != {'front', 'side'} or last_pair_host_s is None
             or now-last_pair_host_s > .5
             or any(now-f['timing']['host_arrival_monotonic_s'] > .5 for f in views.values()))
    if stale:
        return {**fusion, 'status': 'waiting_or_stale', 'position_output_m': None,
                'position_front_m': None, 'covariance_front_m2': None,
                'relative_observation_covariance_m2': None, 'stale': True}
    return {**fusion, 'stale': False}


def check_camera_profile(rig, role, timing):
    profile = rig.get('camera_profiles', {}).get(role)
    if profile is None: return
    changed = any(timing.get(key) != value for key, value in profile.items() if key != 'intrinsics')
    expected, actual = profile['intrinsics'], timing.get('intrinsics', {})
    for key, value in expected.items():
        if key in ('fx', 'fy'):
            # Native per-frame rays account for small reported focal changes.
            # Bound accumulated drift to 0.5%: ~1.6 px at the 640px edge,
            # <7 mm lateral at the front's 1.5 m depth limit. This remains
            # below the heuristic 15 mm floor; it is not an accuracy claim.
            changed |= not np.isclose(actual.get(key, np.nan), value, rtol=.005, atol=0)
        else:
            changed |= actual.get(key) != value
    if changed:
        raise ValueError(f'{role} camera profile differs from the relative calibration: {timing}; recalibrate before fusion')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rig', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=float, default=90)
    parser.add_argument('--max-skew', type=float, default=.025,
                        help='Maximum timestamp span over BOTH cameras and RGB/depth streams')
    parser.add_argument('--max-disagreement', type=float, default=.06)
    parser.add_argument('--front-depth-range', type=float, nargs=2, default=[.10, 1.50], metavar=('MIN_M', 'MAX_M'))
    parser.add_argument('--side-depth-range', type=float, nargs=2, default=[.07, .50], metavar=('MIN_M', 'MAX_M'))
    parser.add_argument('--front-hsv-low', type=int, nargs=3, default=[90, 60, 45])
    parser.add_argument('--side-hsv-low', type=int, nargs=3, default=[90, 60, 45])
    parser.add_argument('--front-roi', type=int, nargs=4)
    parser.add_argument('--side-roi', type=int, nargs=4)
    parser.add_argument('--save-every', type=int, default=15, help='Save one paired aligned RGB-D sample every N accepted pairs; 0 disables')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--mask-tracker', choices=['hsv', 'efficienttam'], default='hsv')
    parser.add_argument('--checkpoint', default='third_party/EfficientTAM/checkpoints/efficienttam_ti_512x512.pt')
    parser.add_argument('--model-config', default='configs/efficienttam/efficienttam_ti_512x512.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--front-seed-box', type=int, nargs=4)
    parser.add_argument('--side-seed-box', type=int, nargs=4)
    parser.add_argument('--seed-session', help='Reuse this session\'s manual target images and boxes, retaining its provenance')
    parser.add_argument('--velocity-max-speed', type=float, default=1.2, help='Perception reporting limit only; does not set any robot speed')
    parser.add_argument('--observation-max-age',type=float,default=.20,help='Maximum host arrival age for exported arm observations')
    args = parser.parse_args(argv)
    for role in ('front', 'side'):
        bounds = getattr(args, role+'_depth_range')
        if not np.isfinite(bounds).all() or not .01 <= bounds[0] < bounds[1] <= 3:
            parser.error(f'{role} depth range must satisfy 0.01 <= min < max <= 3 meters')
    if not .05<=args.observation_max_age<=.25:raise ValueError('Invalid exported observation age')
    if (not np.isfinite([args.seconds, args.max_skew, args.max_disagreement]).all()
            or not 0 < args.seconds <= 1800 or not 0 < args.max_skew <= .05
            or not 0 < args.max_disagreement <= .1 or args.save_every < 0):
        raise ValueError('Invalid capture/pair/fusion limits')
    rig, fs, base, digest = load_rig(args.rig)
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    if not .1 <= args.velocity_max_speed <= 2: raise ValueError('Invalid perception velocity limit')
    mask_model = None
    if args.mask_tracker == 'efficienttam':
        if not args.preview and args.seed_session is None and (args.front_seed_box is None or args.side_seed_box is None):
            raise ValueError('EfficientTAM requires preview for manual boxes or both explicit seed boxes')
        from .efficienttam_tracker import EfficientTAMTracker
        mask_model = EfficientTAMTracker(args.checkpoint, root, args.model_config, args.device)
        if args.seed_session:
            previous = Path(args.seed_session)
            for role in ('front', 'side'):
                prompt = json.loads((previous/(role+'-prompt.json')).read_text())
                if prompt['timing']['camera_serial'] != rig[role+'_serial']:
                    raise ValueError('Manual prompt belongs to a different camera')
                image = cv2.imread(str(previous/('prompt_'+role)/'00000.jpg'))
                if image is None: raise ValueError('Missing manual prompt image')
                mask_model.seed(role, image, prompt['timing']['sensor_timestamp_s'], prompt['box_xywh'])
                prompt['reused_from'] = str(previous.resolve())
                (root/(role+'-prompt.json')).write_text(json.dumps(prompt, indent=2)+'\n')
    output_frame = rig['output_frame']
    manifest = {**vars(args), 'schema': 'front_side_fusion_probe_v1', 'rig_sha256': digest,
                'rig': rig, 'output_frame': output_frame, 'base_calibration_validated': False,
                'position_semantics': 'fused_visible_surface_proxy', 'centroid_validated': False,
                'motion_sent': False, 'motion_enabled': False, 'expert_episode': False,
                'timestamp_note': 'Native global capture timestamps; complete RGB/depth span gated. No hardware exposure sync or transit-latency calibration.',
                'filter_note': 'Causal constant-velocity filter with 15 mm measurement std. Reset on contributing-view changes to avoid interpreting surface-proxy bias as motion.',
                'covariance_note': 'Reported observation covariance includes a heuristic 15 mm floor and view spread, excludes absolute base-calibration uncertainty.'}
    manifest['profile_note'] = 'SDK, resolution, principal point and distortion must match calibration; focal length may vary at most 0.5%, using native per-frame rays. Every live profile is recorded.'
    manifest['association_note'] = 'Acquire from a unique 3D match among blue candidates. Single-view continuity gate: 60 mm plus 0.5 m/s times elapsed capture time. After 0.5 s without an associated observation, require both views to reacquire.'
    manifest['rgbd_storage'] = ('background uncompressed NPZ; bounded queue, atomic file publication'
                                if args.save_every == 1 else 'sampled compressed NPZ')
    if mask_model is not None: manifest['mask_model'] = mask_model.provenance
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    cameras = CameraPair(max_skew_s=args.max_skew).start()
    association = ProxyAssociation(fs, max_disagreement_m=args.max_disagreement)
    tracker = VelocityTracker(frame=output_frame, measurement_std_m=.015, stationary_updates=False, max_speed_mps=args.velocity_max_speed)
    signature = None; origin = last_stamp = None
    start = time.monotonic(); count = fused_count = single_count = 0
    statuses = {}; skews = []; processing = []; last_view = None
    complete = False; error = None; last_publish_s = 0.; last_pair_host_s = None
    paired_views = {}; paired_masks = {}; paired_qualities = {}
    latest_rgbd = None
    latest_fusion = {'status': 'waiting_for_paired_frames', 'position_output_m': None}
    latest_track = None; latest_timing = None; last_observation_write = 0.
    cv2.setNumThreads(1)
    archive = None
    try:
        if args.save_every == 1:
            archive = RGBDArchiveWriter(root)
        with (root/'fusion.jsonl').open('w', buffering=1) as log:
            while time.monotonic()-start < args.seconds:
                if archive is not None: archive.check()
                pair = cameras.poll()
                views = cameras.views(); masks = {}; qualities = {}; track = None
                if pair is not None:
                    before = time.monotonic(); front, side, timing = pair
                    stamp = timing['timestamp_s']
                    if last_stamp is not None and stamp <= last_stamp:
                        raise RuntimeError('Paired timestamps repeated or moved backwards')
                    if origin is None: origin = stamp
                    last_stamp = stamp; count += 1; skews.append(timing['span_s'])
                    candidates = {}; per_view = {}; model_tracking = {}
                    for role, frame in [('front', front), ('side', side)]:
                        check_camera_profile(rig, role, frame['timing'])
                        low = getattr(args, role+'_hsv_low'); roi = getattr(args, role+'_roi')
                        if mask_model is None:
                            detected = detect_mask(frame['bgr'], 'blue_tape', hsv_low=low, roi=roi, all_candidates=True)
                        else:
                            if role not in mask_model.states:
                                box = getattr(args, role+'_seed_box')
                                if box is None:
                                    window = 'Select tape in '+role.upper()+' view, then press ENTER'
                                    print(window, flush=True)
                                    box = cv2.selectROI(window, frame['bgr'], showCrosshair=True, fromCenter=False)
                                    cv2.destroyWindow(window)
                                mask, info = mask_model.seed(role, frame['bgr'], frame['timing']['sensor_timestamp_s'], box)
                                (root/(role+'-prompt.json')).write_text(json.dumps({'box_xywh': list(map(int, box)), 'timing': frame['timing'], 'model': info}, indent=2)+'\n')
                            else:
                                mask, info = mask_model.track(role, frame['bgr'], frame['timing']['sensor_timestamp_s'])
                            model_tracking[role] = info
                            if mask is not None and roi is not None:
                                x, y, w, h = roi; outside = mask.copy(); outside[y:y+h, x:x+w] = 0
                                if np.any(outside): mask = None; info['mask_reason'] = 'model_mask_outside_operator_roi'
                            detected = [] if mask is None else [mask]
                        candidates[role] = []
                        for candidate_index, mask in enumerate(detected):
                            point, quality = depth_measurement(mask, frame['depth'], frame['deproject'],
                                                               *getattr(args, role+'_depth_range'))
                            candidates[role].append({'point': point, 'quality': quality, 'mask': mask, 'candidate_index': candidate_index})
                    if mask_model is not None:
                        mask_model.recover_pair({'front': front, 'side': side}, candidates, model_tracking, fs)
                    valid = {r: [c for c in cc if c['point'] is not None] for r, cc in candidates.items()}
                    if mask_model is not None and count == 1:
                        start = time.monotonic()  # Manual selection is not part of the live test duration.
                    fusion = association.update(stamp, {r: [c['point'] for c in cc] for r, cc in valid.items()})
                    for role, frame in [('front', front), ('side', side)]:
                        index = fusion['selected_candidate_indices'][role]
                        selected = None if index is None else valid[role][index]
                        masks[role] = None if selected is None else selected['mask']
                        quality = {'reason': fusion['status']} if selected is None else selected['quality']
                        if mask_model is not None and selected is None and candidates[role]:
                            # Show visual tracking even when metric depth or
                            # cross-view geometry rejects the 3D observation.
                            masks[role] = candidates[role][0]['mask']
                            if candidates[role][0]['point'] is None:
                                quality = candidates[role][0]['quality']
                        elif mask_model is not None and not candidates[role] and model_tracking[role].get('mask_reason'):
                            quality = {'reason': model_tracking[role]['mask_reason']}
                        qualities[role] = quality
                        per_view[role] = {'timing': frame['timing'], 'quality': quality,
                            'model_tracking': model_tracking.get(role),
                            'point_camera_m': None if selected is None else selected['point'].tolist(),
                            'selected_candidate_index': None if selected is None else selected['candidate_index'],
                            'valid_candidate_indices': [c['candidate_index'] for c in valid[role]],
                            'candidates': [{'candidate_index': c['candidate_index'],
                                            'point_camera_m': None if c['point'] is None else c['point'].tolist(),
                                            'quality': c['quality']} for c in candidates[role]]}
                    status = fusion['status']; statuses[status] = statuses.get(status, 0)+1
                    fused_count += int(status == 'fused'); single_count += int(status == 'single_view')
                    point = fusion['position_front_m']
                    output = None if point is None else (np.asarray(point) if base is None else transform_point(base, point))
                    current_signature = tuple(fusion['sources']) if point is not None else None
                    if current_signature is not None and current_signature != signature:
                        tracker = VelocityTracker(frame=output_frame, measurement_std_m=.015, stationary_updates=False, max_speed_mps=args.velocity_max_speed)
                        signature = current_signature
                    track = tracker.update(stamp-origin, output)
                    latest_track = track
                    latest_timing = {'native_timestamp_s': stamp,
                        'oldest_host_arrival_monotonic_s': min(front['timing']['host_arrival_monotonic_s'], side['timing']['host_arrival_monotonic_s'])}
                    covariance = fusion['covariance_front_m2']
                    if covariance is not None and base is not None:
                        covariance = (base[:3, :3]@np.array(covariance)@base[:3, :3].T).tolist()
                    latest_fusion = {**fusion, 'output_frame': output_frame,
                                     'position_output_m': None if output is None else output.tolist(),
                                     'relative_observation_covariance_m2': covariance}
                    tops={}
                    if base is not None:
                        for role,frame in [('front',front),('side',side)]:
                            if role not in fusion['sources']:continue
                            top=visible_top(masks.get(role),frame['depth'],frame['deproject'],
                                base if role=='front' else base@fs,qualities[role])
                            if top is not None:tops[role]=top
                    latest_fusion['visible_top_z_base_m']=max(tops.values()) if tops else None
                    latest_fusion['visible_top_per_view_m']=tops
                    duration = time.monotonic()-before; processing.append(duration)
                    record = {'index': count-1, 'pair': timing, 'views': per_view,
                              'fusion': latest_fusion, 'track': None if track is None else track.as_dict(),
                              'velocity_report': velocity_report(track),
                              'processing_s': duration, 'host_processed_monotonic_s': time.monotonic(),
                              'action': None, 'motion_sent': False}
                    if args.save_every and (count-1) % args.save_every == 0:
                        record['rgbd_files'] = {role: f'pair_{count-1:06d}_{role}.npz' for role in ('front', 'side')}
                        frames_to_save = {'front': front, 'side': side}
                        if archive is not None:
                            archive.submit(record['rgbd_files'], frames_to_save)
                        else:
                            for role, name in record['rgbd_files'].items():
                                frame = frames_to_save[role]
                                np.savez_compressed(root/name, bgr=frame['bgr'], depth_m=frame['depth'].astype(np.float32))
                    latest_rgbd = {'pair_index': count-1, 'pair': timing,
                        'views': {role: {'timing': frame['timing'], 'quality': qualities[role]}
                                  for role, frame in [('front', front), ('side', side)]},
                        'files': record.get('rgbd_files', {}), 'save_every': args.save_every}
                    log.write(json.dumps(record, allow_nan=False)+'\n')
                    views = {'front': front, 'side': side}
                    last_pair_host_s = time.monotonic()
                    paired_views, paired_masks, paired_qualities = views, masks, qualities
                # Do not display stale coordinates if pairing stops.
                published = live_observation(latest_fusion, views, last_pair_host_s, time.monotonic())
                stale = published['stale']
                observation_stale = (stale or latest_timing is None or
                    time.monotonic()-latest_timing['oldest_host_arrival_monotonic_s'] > args.observation_max_age)
                speed_report = velocity_report(latest_track, observation_stale)
                if pair is not None or time.monotonic()-last_observation_write > .05:
                    observation = {'schema': 'front_side_observation_v1', 'running': True,
                        'rig_sha256': digest, 'output_frame': output_frame, 'timing': latest_timing,
                        'rgbd': latest_rgbd,
                        'host_published_monotonic_s': time.monotonic(), 'stale': observation_stale,
                        'fusion': published, 'track': None if latest_track is None or observation_stale else latest_track.as_dict(),
                        'velocity_report': speed_report, 'motion_sent': False}
                    (root/'observation.tmp.json').write_text(json.dumps(observation, allow_nan=False)+'\n')
                    (root/'observation.tmp.json').replace(root/'observation.json'); last_observation_write = time.monotonic()
                if not stale:
                    views, masks, qualities = paired_views, paired_masks, paired_qualities
                panels = []
                for role in ('front', 'side'):
                    view = views[role]['bgr'].copy() if role in views else np.zeros((480, 640, 3), np.uint8)
                    mask = masks.get(role)
                    if mask is not None:
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(view, contours, -1, (0, 255, 0), 2)
                    reason = qualities.get(role, {}).get('reason')
                    readable = {'insufficient_depth': 'depth unavailable', 'depth_spread': 'depth uncertain',
                        'model_object_not_visible': 'object hidden/lost', 'model_mask_too_small': 'mask too small',
                        'association_lost_or_disagreeing': '3D waiting', 'awaiting_two_view_reacquisition': '3D waiting',
                        'association_ambiguous': '3D ambiguous'}
                    label = role.upper()+(' | EfficientTAM' if mask_model is not None else '')+(' | '+readable.get(reason, str(reason)) if reason else '')
                    cv2.putText(view, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                    panels.append(view)
                footer = np.full((167, 1280, 3), 24, np.uint8)
                point = published.get('position_output_m')
                labels = ['CAMERA-ONLY 3D FUSION | '+('waiting/stale' if stale else latest_fusion['status'])+' | '+output_frame,
                          'XYZ (m): '+(', '.join(f'{v:+.3f}' for v in point) if point is not None else 'unavailable'),
                          f'Pairs {count} | fused {fused_count} | single view {single_count} | maximum allowed RGB/depth span {1000*args.max_skew:g} ms',
                          'Velocity (m/s): '+('unavailable / '+speed_report['status'] if speed_report['velocity_mps'] is None else ', '.join(f'{v:+.3f}' for v in speed_report['velocity_mps'])),
                          'Speed: '+('unavailable' if speed_report['speed_mps'] is None else f'{speed_report["speed_mps"]:.3f} m/s ({speed_report["speed_cm_s"]:.1f} cm/s) | '+speed_report['range_assessment'])+' | DynamicVLA real: 0.2-0.8 m/s',
                          'Surface proxy | provisional base | Q closes perception; external arm controller holds on stale data']
                for i, label in enumerate(labels):
                    cv2.putText(footer, label, (8, 23+26*i), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
                last_view = np.vstack([np.hstack(panels), footer])
                if args.preview:
                    cv2.imshow('Front + side fused 3D - camera only', last_view)
                    if cv2.waitKey(1) & 0xff == ord('q'): break
                if time.monotonic()-last_publish_s >= .5:
                    cv2.imwrite(str(root/'live_frame.tmp.jpg'), last_view); (root/'live_frame.tmp.jpg').replace(root/'live_frame.jpg')
                    (root/'status.tmp.json').write_text(json.dumps({'pairs': count, 'fused_pairs': fused_count,
                        'single_view_pairs': single_count, 'latest_fusion': published,
                        'velocity_report': speed_report,
                        'dropped_frames': cameras.pairer.dropped, 'running': True, 'motion_sent': False}, indent=2)+'\n')
                    (root/'status.tmp.json').replace(root/'status.json')
                    last_publish_s = time.monotonic()
                if not views and time.monotonic()-start > 30: raise RuntimeError('No camera frames after 30 seconds')
                if pair is None: time.sleep(.01)
        complete = True
    except BaseException as exc:
        error = str(exc); raise
    finally:
        cleanup_error = None
        try:
            cameras.close()
        except BaseException as exc:
            cleanup_error = exc
        if archive is not None:
            try:
                archive.close()
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
        if cleanup_error is not None:
            complete = False
            error = error or str(cleanup_error)
        if args.preview: cv2.destroyAllWindows()
        if last_view is not None: cv2.imwrite(str(root/'last_frame.jpg'), last_view)
        summary = {'complete': complete, 'error': error, 'pairs': count, 'statuses': statuses,
                   'paired_data_available': count > 0, 'two_view_fusion_observed': fused_count > 0,
                   'dropped_frames': cameras.pairer.dropped, 'reader_overruns': cameras.overruns,
                   'pair_span_p50_ms': None if not skews else float(1000*np.median(skews)),
                   'pair_span_max_ms': None if not skews else float(1000*max(skews)),
                   'processing_p95_ms': None if not processing else float(1000*np.percentile(processing, 95)),
                   'motion_sent': False, 'centroid_validated': False, 'expert_episode': False}
        status = {'pairs': count, 'fused_pairs': fused_count, 'single_view_pairs': single_count,
                  'running': False, 'complete': complete, 'error': error, 'motion_sent': False,
                  'latest_fusion': live_observation(latest_fusion, {}, None, time.monotonic()),
                  'dropped_frames': cameras.pairer.dropped}
        (root/'status.tmp.json').write_text(json.dumps(status, indent=2)+'\n')
        (root/'status.tmp.json').replace(root/'status.json')
        (root/'observation.tmp.json').write_text(json.dumps({'schema': 'front_side_observation_v1',
            'running': False, 'stale': True, 'track': None, 'rig_sha256': digest,
            'velocity_report': velocity_report(None, True), 'motion_sent': False})+'\n')
        (root/'observation.tmp.json').replace(root/'observation.json')
        (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n'); print(json.dumps(summary, indent=2))
        if cleanup_error is not None: raise cleanup_error


if __name__ == '__main__': main()
