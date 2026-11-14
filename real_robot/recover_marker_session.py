"""Offline green-marker recovery from archived, already corrected RGB-D frames.

Never opens hardware or edits the source session. This detector is specific to
the fixed front view and green marker in the November 8 collection. Occlusion,
ambiguous masks and rejected depth remain missing; no predicted positions are
used as measurements and no grasp phases are inferred.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np
import pyrealsense2 as rs

from .dom_dataset import json_write, sha256
from .perception import depth_measurement
from .tracking import VelocityTracker
from .velocity_report import velocity_report


def marker_mask(bgr):
    if bgr.shape != (480, 640, 3):
        raise ValueError('Detector requires the reviewed native front view')
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (65, 85, 30), (95, 255, 255))
    # Reviewed tabletop/box image region; exclude background objects and cables.
    region = np.zeros(green.shape, np.uint8)
    cv2.fillConvexPoly(region, np.array([[200,150],[425,150],[535,479],[120,479]]), 255)
    green &= region
    # Link the green cap and barrel across the white printing/collar, retaining
    # only original green pixels for depth (the closing never invents samples).
    joined = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((21,21), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
    candidates = []
    for i in range(1, count):
        x,y,w,h,area = stats[i]
        mask = (labels == i) & (green > 0)
        pixels = int(mask.sum())
        if pixels < 45 or not 12 <= max(w,h) <= 85 or min(w,h) > 40:
            continue
        candidates.append((pixels, mask))
    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        return None, 'green_marker_not_visible'
    if len(candidates) > 1 and candidates[1][0] >= .3*candidates[0][0]:
        return None, 'ambiguous_green_components'
    return candidates[0][1], None


def deprojector(timing):
    intr = rs.intrinsics()
    for key, value in timing['intrinsics'].items():
        setattr(intr, key, value)
    name = timing['distortion_model'].split('.')[-1]
    intr.model = getattr(rs.distortion, name)
    return lambda pixel, z: rs.rs2_deproject_pixel_to_point(intr, pixel, z)


def measure_marker(mask, depth, deproject):
    # Glossy green plastic leaves many depth holes. Use only measured pixels,
    # requiring more samples and tighter spread than the live generic detector.
    # A low coverage fraction is recorded explicitly, never filled with table Z.
    if mask is not None:
        good = mask & np.isfinite(depth) & (depth >= .1) & (depth <= 1.5)
        if good.sum() < 40:
            return None, dict(reason='fewer_than_40_measured_depth_pixels', valid_depth_pixels=int(good.sum()))
    point, quality = depth_measurement(mask, depth, deproject,
                                       min_valid_fraction=.15, max_depth_mad_m=.01)
    quality['sparse_depth_recovery'] = quality.get('valid_fraction', 0) < .5
    return point, quality


def recover(source, output):
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise ValueError('Output exists; use a new path')
    audit = json.loads((source.parent/'resume_stream_audit.json').read_text())
    ids = {r['episode_id'] for r in audit['attempts'] if r['missing'] == ['object_positions']}
    manifest = json.loads((source/'manifest.json').read_text())
    if manifest['task']['id'] != 'marker':
        raise ValueError('Only the green marker collection is supported')
    transform = np.asarray(manifest['rig']['T_base_front_candidate'])
    output.mkdir(parents=True)
    for name in ['fixed', 'wrist']:
        shutil.copytree(source/name, output/name, copy_function=os.link)
    for name in ['robot.jsonl', 'summary.json', 'attempts.jsonl']:
        shutil.copy2(source/name, output/name)
    (output/'episodes').mkdir()
    episodes = []
    for i in sorted(ids):
        path = source/f'episodes/episode_{i:06d}.json'
        shutil.copy2(path, output/'episodes'/path.name)
        episodes.append(json.loads(path.read_text()))
    spec = dict(method='green_marker_hsv_depth_v1', source=str(source),
                source_observations_sha256=sha256(source/'observations.jsonl'),
                script_sha256=sha256(Path(__file__)), recovered_episode_ids=sorted(ids),
                depth_processing='Saved corrected depth used unchanged; no second correction',
                phase_annotations='None; no automatic grasp confirmation',
                depth_gate=dict(min_measured_pixels=40,min_valid_fraction=.15,max_mad_m=.01),
                position_semantics='green visible-surface proxy; not object centroid')
    manifest['offline_tracking_recovery'] = spec
    json_write(output/'manifest.json', manifest)
    counters = Counter(); per_episode = {str(i):Counter() for i in ids}
    review = output/'review'; review.mkdir()
    trackers = {}; origins = {}; last = {}; sample_counts = Counter()
    with (source/'observations.jsonl').open() as src, (output/'observations.jsonl').open('w') as dest:
        for line in src:
            row = json.loads(line); host = row['host_recorded_monotonic_s']
            ep = next((e for e in episodes if e['start_monotonic_s'] <= host < e['end_monotonic_s']), None)
            if ep is None:
                continue
            eid = ep['episode_id']; key = str(eid)
            frame_path = source/row['rgbd']['files']['front']
            with np.load(frame_path, allow_pickle=False) as data:
                bgr, depth = data['bgr'], data['depth_m']
            timing = row['rgbd']['views']['front']['timing']
            correction = timing.get('depth_correction')
            if not correction:
                raise ValueError('Expected already depth-corrected archive')
            mask, reason = marker_mask(bgr)
            point, quality = measure_marker(mask, depth, deprojector(timing))
            if reason: quality['reason'] = reason
            stamp = timing['color_timestamp_s']
            base = None if point is None else transform[:3,:3]@point+transform[:3,3]
            if base is not None and eid in last:
                old_t, old_p = last[eid]
                if stamp-old_t < .3 and np.linalg.norm(base-old_p) > .015+1.2*(stamp-old_t):
                    point = base = None; quality['reason'] = 'implausible_position_jump'
            if base is not None:
                last[eid] = (stamp, base)
                raw_z = 1/(1/point[2]-correction['inverse_depth_offset_per_m'])
                lo,hi = correction['raw_depth_domain_m']
                quality.update(depth_correction_model_sha256=correction['model_sha256'],
                               depth_correction_extrapolated=not lo<=raw_z<=hi)
            if eid not in trackers:
                trackers[eid] = VelocityTracker(frame='right_base_estimate', measurement_std_m=.015,
                                                stationary_updates=False, max_speed_mps=1.2)
                origins[eid] = stamp
            track = trackers[eid].update(stamp-origins[eid], base)
            row['stale'] = False
            row['fusion'] = dict(position_front_m=None if point is None else point.tolist(),
                                 position_output_m=None if base is None else base.tolist(),
                                 sources=[] if point is None else ['front'], stale=False,
                                 output_frame='right_base_estimate', status='lost' if point is None else 'single_view',
                                 position_semantics='green_visible_surface_proxy', motion_sent=False)
            row['views']['front'].update(quality=quality, point_camera_m=None if point is None else point.tolist(),
                                        model_tracking={'tracker':spec['method'], 'mask_pixels':0 if mask is None else int(mask.sum())})
            row['track'] = None if track is None else track.as_dict()
            row['velocity_report'] = velocity_report(track)
            row['offline_tracking_recovery'] = spec['method']
            # Preserve every original robot/camera timestamp and image reference.
            dest.write(json.dumps(row, allow_nan=False)+'\n')
            label = quality.get('reason') or 'valid_position'
            counters[label] += 1; per_episode[key][label] += 1
            per_episode[key]['frames'] += 1
            per_episode[key]['valid_velocity'] += row['velocity_report']['velocity_mps'] is not None
            # Spread visual QA across each attempt, retaining missing detections.
            fraction = (host-ep['start_monotonic_s'])/(ep['end_monotonic_s']-ep['start_monotonic_s'])
            if sample_counts[eid] < 5 and fraction >= [.05,.25,.5,.7,.9][sample_counts[eid]]:
                view = bgr.copy()
                if mask is not None: view[mask] = (0,0,255)
                cv2.putText(view,f'{eid} {fraction:.2f} {label}',(8,25),0,.55,(0,0,255),1)
                cv2.imwrite(str(review/f'{eid:03d}-{sample_counts[eid]}.jpg'),view)
                sample_counts[eid] += 1
            if sum(counters.values()) % 1000 == 0:
                print(f'Recovered {sum(counters.values())} frames; {counters["valid_position"]} valid positions',flush=True)
    report = {**spec, 'complete':True, 'counts':dict(counters),
              'episodes':{k:dict(v) for k,v in per_episode.items()}}
    json_write(output/'recovery_report.json', report)
    print(json.dumps({'counts':report['counts'],'episodes':len(episodes)}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    recover(args.source,args.output)
