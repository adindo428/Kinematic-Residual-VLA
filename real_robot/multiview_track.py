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
from .multiview import CameraPair, FRONT_SERIAL, SIDE_SERIAL, ProxyAssociation, fuse_proxy
from .perception import detect_mask, depth_measurement
from .tracking import VelocityTracker
from .velocity_report import velocity_report
from .visible_extent import visible_top
from .rgbd_archive import RGBDArchiveWriter
from .recording_lease import recording_active
from .object_profiles import load_prompt, save_prompt, has_prompt
from .mask_prompt import seed_live


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
                or type(rig.get('base_calibration_validated')) is not bool):
            raise ValueError('Base transform needs an explicit validation status and fixed mounting confirmation')
        if rig['base_calibration_validated']:
            from .base_calibration import check_evidence
            check_evidence(rig)
        base = validate_transform(rig['T_base_front_candidate'])
        if not np.allclose(base@fs, validate_transform(rig['T_base_side_candidate']), atol=1e-7):
            raise ValueError('Front/side/base transform composition disagrees')
    elif rig.get('output_frame') != 'front_camera_optical':
        raise ValueError('Rig has no transform into its named output frame')
    return rig, fs, base, hashlib.sha256(raw).hexdigest()


def live_observation(fusion, views, last_pair_host_s, now, required_views=('front','side')):
    """Expire the published coordinates even when no further pairs arrive."""
    stale = (set(views) != set(required_views) or last_pair_host_s is None
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
    changed = any(timing.get(key) != value for key, value in profile.items()
                  if key not in ('intrinsics', 'requested_fps'))
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
    parser.add_argument('--surface-calibration',help='RGB sphere/plane measurement for camera-only speed studies; never a robot-control input')
    parser.add_argument('--min-mask-pixels',type=int,default=80,help='Minimum eroded EfficientTAM area; metric depth still needs >=20 valid pixels and 50% coverage')
    parser.add_argument('--front-camera-only',action='store_true',help='Open only front RGB-D; side camera is disabled (for camera-only speed studies)')
    parser.add_argument('--front-mask-only', action='store_true',
                        help='Track a front mask only; side camera remains a raw preview')
    parser.add_argument('--position-source', choices=['fused', 'front'], default='fused',
                        help='front: use only the prompted front RGB-D target for metric tracking')
    parser.add_argument('--max-disagreement', type=float, default=.06)
    parser.add_argument('--front-depth-range', type=float, nargs=2, default=[.10, 1.50], metavar=('MIN_M', 'MAX_M'))
    parser.add_argument('--side-depth-range', type=float, nargs=2, default=[.07, .50], metavar=('MIN_M', 'MAX_M'))
    parser.add_argument('--front-hsv-low', type=int, nargs=3, default=[90, 60, 45])
    parser.add_argument('--side-hsv-low', type=int, nargs=3, default=[90, 60, 45])
    parser.add_argument('--front-roi', type=int, nargs=4)
    parser.add_argument('--side-roi', type=int, nargs=4)
    parser.add_argument('--save-every', type=int, default=15, help='Save one paired aligned RGB-D sample every N accepted pairs; 0 disables')
    parser.add_argument('--recording-lease', help='Archive only while the same-boot recorder lease is active')
    parser.add_argument('--compress-rgbd', action='store_true')
    parser.add_argument('--camera-fps', type=int, choices=(15,30), default=15)
    parser.add_argument('--front-color-format',choices=['bgr8','yuyv'],default='bgr8')
    parser.add_argument('--object-profile', help='Reusable per-object reference image/mask directory')
    parser.add_argument('--auto-mask',action='store_true',help='Propose a fresh mask from saved reference appearance; fall back to manual if ambiguous')
    parser.add_argument('--new-mask', action='store_true', help='Replace the saved object prompts interactively')
    parser.add_argument('--interactive-reseed',action='store_true',help='Allow manual reselection and automatically re-prompt failed evaluation startup')
    parser.add_argument('--collection-mode', action='store_true', help='Use newest frame pairs and object-general recovery')
    parser.add_argument('--policy-feed', action='store_true', help='Atomically publish the original front RGB and matching tracking metadata for learned-policy evaluation')
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
    if args.auto_mask:args.new_mask=True
    if args.interactive_reseed and (args.mask_tracker!='efficienttam' or not args.preview):
        parser.error('--interactive-reseed requires --mask-tracker efficienttam --preview')
    if args.surface_calibration and (not args.front_camera_only or args.position_source!='front' or args.save_every or args.recording_lease):
        parser.error('Surface mode is restricted to front-camera-only speed studies with no robot recording lease or RGB-D archive')
    if args.min_mask_pixels<20:parser.error('--min-mask-pixels must be >=20')
    if args.front_camera_only:args.front_mask_only=True
    active_roles=('front',) if args.front_camera_only else ('front','side')
    if args.front_mask_only and args.position_source != 'front':
        parser.error('--front-mask-only requires --position-source front')
    if args.position_source == 'front' and args.mask_tracker != 'efficienttam':
        parser.error('Front-only acquisition requires an explicitly prompted EfficientTAM target')
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
    surface_model=None
    if args.surface_calibration:
        from .surface_tracking import SurfaceModel
        surface_model=SurfaceModel.load(args.surface_calibration)
        if surface_model.spec['rig_sha256']!=digest:raise ValueError('Rig changed after surface calibration')
    def report_velocity(track, stale=False):
        report=velocity_report(track,stale)
        if surface_model is not None:
            report.update(semantics='RGB mask-centroid sphere center constrained to calibrated support planes; assumes rolling contact, no stereo depth',
                          measurement_method='rgb_support_plane',
                          uncertainty_note='Filter uncertainty excludes board scale/plane calibration, mask-centroid bias and airborne motion')
        return report
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    if not .1 <= args.velocity_max_speed <= 2: raise ValueError('Invalid perception velocity limit')
    mask_model = None
    if args.mask_tracker == 'efficienttam':
        roles = ('front',) if args.front_mask_only else ('front', 'side')
        ready_prompts = (not args.new_mask and args.object_profile and
                         all(has_prompt(args.object_profile, role) for role in roles))
        if not args.preview and not ready_prompts and args.seed_session is None and any(getattr(args, role+'_seed_box') is None for role in roles):
            raise ValueError('EfficientTAM requires preview for manual boxes or both explicit seed boxes')
        from .efficienttam_tracker import EfficientTAMTracker
        mask_model = EfficientTAMTracker(args.checkpoint, root, args.model_config, args.device,
                                        generic_recovery=args.collection_mode,min_mask_pixels=args.min_mask_pixels)
        mask_model.recovery_depth_ranges={role:getattr(args,role+'_depth_range') for role in ('front','side')}
        for role in ('front','side'):
            if args.front_mask_only and role == 'side': continue
            previous = (Path(args.object_profile) if args.object_profile and
                        has_prompt(args.object_profile,role) and not args.new_mask else
                        Path(args.seed_session) if args.seed_session and not args.new_mask else None)
            if previous is not None:
                image, saved_mask, prompt = load_prompt(previous,role,rig[role+'_serial'])
                # A stored reference is not a live measurement and may be from a
                # different camera clock epoch. Start its internal seed at zero.
                try:
                    mask_model.seed(role,image,0.,prompt['box_xywh'],seed_mask=saved_mask)
                except ValueError:
                    if not args.interactive_reseed:raise
                    mask_model.states.pop(role,None)
                    print('Saved '+role+' prompt failed; select a fresh object on the live frame.',flush=True)
                    continue
                prompt['reused_from']=str(previous.resolve())
                (root/(role+'-prompt.json')).write_text(json.dumps(prompt,indent=2)+'\n')
                if args.object_profile:
                    save_prompt(root,args.object_profile,role)
                print(f'Reused {role} object mask: {previous}',flush=True)
    output_frame = rig['output_frame']
    manifest = {**vars(args), 'schema': 'front_side_fusion_probe_v1', 'rig_sha256': digest,
                'rig': rig, 'output_frame': output_frame,
                'base_calibration_validated': rig.get('base_calibration_validated',False),
                'position_semantics': 'fused_visible_surface_proxy', 'centroid_validated': False,
                'motion_sent': False, 'motion_enabled': False, 'expert_episode': False,
                'timestamp_note': 'Native global capture timestamps; complete RGB/depth span gated. No hardware exposure sync or transit-latency calibration.',
                'filter_note': 'Causal constant-velocity filter with 15 mm measurement std. Reset on contributing-view changes to avoid interpreting surface-proxy bias as motion.',
                'covariance_note': 'Reported observation covariance includes a heuristic 15 mm floor and view spread, excludes absolute base-calibration uncertainty.'}
    if args.position_source == 'front':
        manifest['association_note'] = 'Front prompted target only; acquire from a unique valid front depth candidate. Side measurements do not contribute. No cross-view identity validation.'
    manifest['profile_note'] = 'SDK, resolution, principal point and distortion must match calibration; focal length may vary at most 0.5%, using native per-frame rays. Every live profile is recorded.'
    if args.position_source != 'front':
        manifest['association_note'] = 'Acquire from a unique 3D match among blue candidates. Single-view continuity gate: 60 mm plus 0.5 m/s times elapsed capture time. After 0.5 s without an associated observation, require both views to reacquire.'
    manifest['rgbd_storage'] = ('background '+('compressed' if args.compress_rgbd else 'uncompressed')+' NPZ; bounded queue, atomic file publication'
                                if args.save_every == 1 else 'sampled compressed NPZ')
    if args.compress_rgbd and args.save_every==1:
        manifest['rgbd_compression']='lossless NPZ ZIP_DEFLATED level 1'
        manifest['rgbd_archive_workers']=4 if args.collection_mode else 1
    if mask_model is not None: manifest['mask_model'] = mask_model.provenance
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if surface_model is not None:
        manifest['position_semantics']='estimated_sphere_center_constrained_to_support_plane'
        manifest['surface_calibration_sha256']=hashlib.sha256(Path(args.surface_calibration).read_bytes()).hexdigest()
        (root/'surface_calibration.json').write_text(json.dumps(surface_model.spec,indent=2)+'\n')
    manifest['camera_reconnect_attempts']=3 if args.collection_mode else 0
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    formats={'front':args.front_color_format} if args.front_color_format!='bgr8' else {}
    cameras = CameraPair(fps=args.camera_fps, max_skew_s=args.max_skew,reconnect_attempts=3 if args.collection_mode else 0,front_only=args.front_camera_only,depth_corrections=rig.get('depth_corrections'),**({'color_formats':formats} if formats else {})).start()
    association = ProxyAssociation(fs, max_disagreement_m=args.max_disagreement)
    tracker = VelocityTracker(frame=output_frame, measurement_std_m=.015, stationary_updates=False, max_speed_mps=args.velocity_max_speed)
    signature = None; origin = last_stamp = None
    start = time.monotonic(); count = fused_count = single_count = 0
    statuses = {}; skews = []; processing = []; last_view = None
    complete = False; error = None; last_publish_s = 0.; last_pair_host_s = None
    startup_invalid_since=None
    paired_views = {}; paired_masks = {}; paired_qualities = {}
    latest_rgbd = None
    latest_views = {}
    latest_fusion = {'status': 'waiting_for_paired_frames', 'position_output_m': None}
    latest_track = None; latest_timing = None; last_observation_write = 0.
    cv2.setNumThreads(1)
    archive = None
    try:
        if args.save_every == 1:
            archive = RGBDArchiveWriter(root, compressed=args.compress_rgbd,
                                       workers=4 if args.collection_mode and args.compress_rgbd else 1)
        with (root/'fusion.jsonl').open('w', buffering=1) as log:
            while time.monotonic()-start < args.seconds:
                if archive is not None: archive.check()
                pair = cameras.poll_latest() if args.collection_mode else cameras.poll()
                for event in cameras.take_events():
                    with (root/'camera_events.jsonl').open('a') as events:events.write(json.dumps(event)+'\n')
                views = cameras.views(); masks = {}; qualities = {}; track = None
                if pair is not None:
                    before = time.monotonic(); front, side, timing = pair
                    frames={'front':front} if args.front_camera_only else {'front':front,'side':side}
                    stamp = timing['timestamp_s']
                    if last_stamp is not None and stamp <= last_stamp:
                        raise RuntimeError('Paired timestamps repeated or moved backwards')
                    if origin is None: origin = stamp
                    last_stamp = stamp; count += 1; skews.append(timing['span_s'])
                    if args.interactive_reseed and (root/'RESEED').exists():
                        for role in list(mask_model.states):mask_model.states.pop(role,None)
                        signature=None;startup_invalid_since=None
                        (root/'RESEED').unlink()
                    candidates = {}; per_view = {}; model_tracking = {}
                    for role, frame in frames.items():
                        check_camera_profile(rig, role, frame['timing'])
                        low = getattr(args, role+'_hsv_low'); roi = getattr(args, role+'_roi')
                        if args.front_mask_only and role == 'side':
                            detected = []
                            model_tracking[role] = {'tracker': None, 'object_probability': 0.,
                                                    'mask_reason': 'preview_only'}
                        elif mask_model is None:
                            detected = detect_mask(frame['bgr'], 'blue_tape', hsv_low=low, roi=roi, all_candidates=True)
                        else:
                            if role not in mask_model.states:
                                box = getattr(args, role+'_seed_box')
                                mask,info=seed_live(mask_model,role,frame,root,args.object_profile,box,
                                                    retry=args.interactive_reseed,automatic=args.auto_mask)
                            else:
                                mask, info = mask_model.track(role, frame['bgr'], frame['timing']['sensor_timestamp_s'])
                            model_tracking[role] = info
                            if mask is not None and roi is not None:
                                x, y, w, h = roi; outside = mask.copy(); outside[y:y+h, x:x+w] = 0
                                if np.any(outside): mask = None; info['mask_reason'] = 'model_mask_outside_operator_roi'
                            detected = [] if mask is None else [mask]
                        candidates[role] = []
                        for candidate_index, mask in enumerate(detected):
                            if surface_model is not None:
                                surface_model.check_camera(frame['timing'])
                                point,quality=surface_model.measure(mask,frame['deproject'])
                            else:
                                point, quality = depth_measurement(mask, frame['depth'], frame['deproject'],
                                                                   *getattr(args, role+'_depth_range'))
                                correction = frame['timing'].get('depth_correction')
                                if correction and quality.get('depth_m') is not None:
                                    corrected_z = quality['depth_m']
                                    raw_z = 1/(1/corrected_z-correction['inverse_depth_offset_per_m'])
                                    lo, hi = correction['raw_depth_domain_m']
                                    quality.update(depth_correction_model_sha256=correction['model_sha256'],
                                                   raw_depth_m=raw_z,
                                                   depth_correction_extrapolated=not lo <= raw_z <= hi)
                            candidates[role].append({'point': point, 'quality': quality, 'mask': mask, 'candidate_index': candidate_index})
                    if mask_model is not None and not args.front_mask_only:
                        mask_model.recover_pair({'front': front, 'side': side}, candidates, model_tracking, fs)
                    valid = {r: [c for c in cc if c['point'] is not None] for r, cc in candidates.items()}
                    if mask_model is not None and count == 1:
                        start = time.monotonic()  # Manual selection is not part of the live test duration.
                    if args.position_source == 'front':
                        selected = 0 if len(valid['front']) == 1 else None
                        fusion = fuse_proxy(None if selected is None else valid['front'][selected]['point'], None, fs)
                        fusion['selected_candidate_indices'] = {'front': selected, 'side': None}
                    else:
                        fusion = association.update(stamp, {r: [c['point'] for c in cc] for r, cc in valid.items()})
                    for role, frame in frames.items():
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
                    if surface_model is not None:
                        fusion['position_semantics']='estimated_sphere_center_constrained_to_support_plane'
                        fusion['measurement_method']='rgb_support_plane'
                        fusion['support_surface']=(qualities.get('front') or {}).get('surface')
                    current_signature = (tuple(fusion['sources'])+((fusion.get('support_surface'),) if surface_model is not None else ())) if point is not None else None
                    if current_signature is not None and current_signature != signature:
                        tracker = VelocityTracker(frame=output_frame, measurement_std_m=.015, stationary_updates=False, max_speed_mps=args.velocity_max_speed)
                        signature = current_signature
                    track = tracker.update(stamp-origin, output)
                    latest_track = track
                    if args.interactive_reseed and not (root/'EVAL_STARTED').exists():
                        if track is not None and track.ready:startup_invalid_since=None
                        else:
                            now=time.monotonic()
                            if startup_invalid_since is None:startup_invalid_since=now
                            elif now-startup_invalid_since>=4:
                                print('No usable startup geometry: select a fresh mask. The selected area also needs valid depth.',flush=True)
                                (root/'RESEED').touch();startup_invalid_since=None
                    latest_timing = {'native_timestamp_s': stamp,
                        'oldest_host_arrival_monotonic_s': min(frame['timing']['host_arrival_monotonic_s'] for frame in frames.values())}
                    covariance = fusion['covariance_front_m2']
                    if covariance is not None and base is not None:
                        covariance = (base[:3, :3]@np.array(covariance)@base[:3, :3].T).tolist()
                    latest_fusion = {**fusion, 'output_frame': output_frame,
                                     'position_output_m': None if output is None else output.tolist(),
                                     'relative_observation_covariance_m2': covariance}
                    tops={}
                    if base is not None and surface_model is None:
                        for role,frame in frames.items():
                            if role not in fusion['sources']:continue
                            top=visible_top(masks.get(role),frame['depth'],frame['deproject'],
                                base if role=='front' else base@fs,qualities[role])
                            if top is not None:tops[role]=top
                    latest_fusion['visible_top_z_base_m']=max(tops.values()) if tops else None
                    latest_fusion['visible_top_per_view_m']=tops
                    duration = time.monotonic()-before; processing.append(duration)
                    latest_views = per_view
                    record = {'index': count-1, 'pair': timing, 'views': per_view,
                              'fusion': latest_fusion, 'track': None if track is None else track.as_dict(),
                              'velocity_report': report_velocity(track),
                              'processing_s': duration, 'host_processed_monotonic_s': time.monotonic(),
                              'action': None, 'motion_sent': False}
                    if (args.save_every and (count-1) % args.save_every == 0
                            and recording_active(args.recording_lease, time.monotonic())):
                        record['rgbd_files'] = {role: f'pair_{count-1:06d}_{role}.npz' for role in active_roles}
                        frames_to_save = frames
                        if archive is not None:
                            archive.submit(record['rgbd_files'], frames_to_save)
                        else:
                            for role, name in record['rgbd_files'].items():
                                frame = frames_to_save[role]
                                np.savez_compressed(root/name, bgr=frame['bgr'], depth_m=frame['depth'].astype(np.float32))
                    latest_rgbd = {'pair_index': count-1, 'pair': timing,
                        'views': {role: {'timing': frame['timing'], 'quality': qualities[role]}
                                  for role, frame in frames.items()},
                        'files': record.get('rgbd_files', {}), 'save_every': args.save_every}
                    log.write(json.dumps(record, allow_nan=False)+'\n')
                    views = frames
                    last_pair_host_s = time.monotonic()
                    paired_views, paired_masks, paired_qualities = views, masks, qualities
                # Do not display stale coordinates if pairing stops.
                published = live_observation(latest_fusion, views, last_pair_host_s, time.monotonic(),required_views=active_roles)
                stale = published['stale']
                observation_stale = (stale or latest_timing is None or
                    time.monotonic()-latest_timing['oldest_host_arrival_monotonic_s'] > args.observation_max_age)
                speed_report = report_velocity(latest_track, observation_stale)
                if pair is not None or time.monotonic()-last_observation_write > .05:
                    camera_health=cameras.health()
                    cameras_ready=(camera_health['ready'] and latest_timing is not None and
                                   time.monotonic()-latest_timing['oldest_host_arrival_monotonic_s']<=.75)
                    observation = {'camera_health':camera_health,'fixed_cameras_ready':cameras_ready,'schema': 'front_side_observation_v1', 'running': True,
                        'rig_sha256': digest, 'output_frame': output_frame, 'timing': latest_timing,
                        'recording_lease': args.recording_lease,
                        'camera_fps': args.camera_fps,
                        'archive':None if archive is None else archive.stats(),
                        'rgbd': latest_rgbd, 'views': latest_views,
                        'host_published_monotonic_s': time.monotonic(), 'stale': observation_stale,
                        'fusion': published, 'track': None if latest_track is None or observation_stale else latest_track.as_dict(),
                        'velocity_report': speed_report, 'motion_sent': False}
                    (root/'observation.tmp.json').write_text(json.dumps(observation, allow_nan=False)+'\n')
                    (root/'observation.tmp.json').replace(root/'observation.json'); last_observation_write = time.monotonic()
                    if args.policy_feed and pair is not None and 'front' in frames:
                        with (root/'policy_front.tmp').open('wb') as feed:
                            np.savez(feed, bgr=frames['front']['bgr'], metadata=json.dumps(observation,allow_nan=False))
                        (root/'policy_front.tmp').replace(root/'policy_front.npz')
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
                    label = role.upper()+(' | DISABLED' if args.front_camera_only and role=='side' else ' | preview only' if args.front_mask_only and role == 'side' else ' | EfficientTAM' if mask_model is not None else '')+(' | '+readable.get(reason, str(reason)) if reason else '')
                    cv2.putText(view, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                    if args.front_camera_only and role=='front':
                        info=(latest_views.get(role) or {}).get('model_tracking') or {}
                        depth_quality=qualities.get(role) or {}
                        fraction=depth_quality.get('valid_fraction')
                        detail=(f'Mask: {info.get("mask_pixels",0)} px | RGB plane: {depth_quality.get("surface","unavailable")}' if surface_model is not None else
                                f'Mask: {info.get("mask_pixels",0)} px | valid depth: '+('unavailable' if fraction is None else f'{fraction:.0%}'))
                        cv2.putText(view,detail,(8,50),cv2.FONT_HERSHEY_SIMPLEX,.55,
                                    (0,255,0) if (surface_model is not None and depth_quality.get('reason') is None) or (fraction is not None and fraction>=.5) else (0,100,255),1)
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
                   'archive':None if archive is None else archive.stats(),
                   'camera_health':cameras.health(),
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
            'running': False, 'stale': True, 'track': None, 'rig_sha256': digest,'error':error,
            'velocity_report': report_velocity(None, True), 'motion_sent': False})+'\n')
        (root/'observation.tmp.json').replace(root/'observation.json')
        (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n'); print(json.dumps(summary, indent=2))
        if cleanup_error is not None: raise cleanup_error


if __name__ == '__main__': main()
