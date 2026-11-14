"""Reproject saved surface points and replay causal velocity filtering offline.

Uses complete perception logs, then aligns to the original exported anchors.
No video inference, hardware access, Hub writes, or changes to robot/action data.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq

from .calibration_acceptance import accepted_transform, dataset_acceptance
from .dataset_quality import repair_tracking
from .depth_calibration import check_profile, digest
from .dom_dataset import DATA_PATH, json_write, jsonl_read, jsonl_write, numeric_stats, sha256, write_parquet
from .tracking import VelocityTracker


def replay_session(session, model):
    session = Path(session)
    perception = session.parent/'perception'
    needed = [session/'observations.jsonl', perception/'fusion.jsonl', perception/'manifest.json']
    if not all(p.exists() for p in needed):
        return None, dict(status='missing_raw_tracking', missing=[str(p) for p in needed if not p.exists()])
    manifest = json.loads(needed[2].read_text())
    rig = manifest['rig']
    if manifest.get('position_source') != 'front' or manifest.get('surface_calibration'):
        raise ValueError(f'Expected front depth-derived points in {session}')
    if not accepted_transform(rig):
        raise ValueError('Source transform does not match accepted transform')
    if rig.get('depth_corrections'):
        raise ValueError('Source already has depth corrections; refuse double correction')
    transform = np.asarray(rig['T_base_front_candidate'])
    b = model['inverse_depth_offset_per_m']
    lo, hi = model['raw_depth_domain_m']
    app_lo, app_hi = model.get('application_depth_range_m', [lo, hi])
    tracker = VelocityTracker(frame='right_base_estimate', measurement_std_m=.015,
                              stationary_updates=False, max_speed_mps=manifest.get('velocity_max_speed', 1.2))
    records = {}; origin = None; previous = None; profiles = set(); counts = Counter()
    with needed[1].open() as stream:
        for line in stream:
            row = json.loads(line); stamp = row['pair']['timestamp_s']
            if previous is not None and stamp <= previous:
                raise ValueError('Perception timestamps are not strictly increasing')
            previous = stamp
            if origin is None: origin = stamp
            fusion = row['fusion']; point = None; extrapolated = False; old_point = fusion.get('position_output_m')
            if old_point is not None:
                if fusion['sources'] != ['front']:
                    raise ValueError('Cannot correct a fused or side-only point using the front depth model')
                front = row['views']['front']; timing = front['timing']
                profile = json.dumps({k:timing.get(k) for k in ('camera_serial','intrinsics','distortion_model','depth_correction')}, sort_keys=True)
                if profile not in profiles:
                    check_profile(model, timing); profiles.add(profile)
                raw = np.asarray(front['point_camera_m'], float)
                if not np.allclose(transform[:3,:3]@raw+transform[:3,3], old_point, atol=1e-7, rtol=0):
                    raise ValueError('Recorded camera point does not reproduce its base-frame point')
                if app_lo <= raw[2] <= app_hi:
                    point = transform[:3,:3]@(raw/(1+b*raw[2]))+transform[:3,3]
                    extrapolated = not lo <= raw[2] <= hi
                    counts['corrected_measurements'] += 1
                    counts['extrapolated_measurements'] += extrapolated
                else:
                    counts['outside_application_range'] += 1
            track = tracker.update(stamp-origin, point)
            velocity = track.velocity_mps.tolist() if track is not None and track.accepted and track.ready else None
            records[stamp] = dict(position=None if point is None else point.tolist(),
                                  raw_position=old_point, velocity=velocity, extrapolated=extrapolated)
            counts['perception_updates'] += 1
    observations = []; positions = []
    with needed[0].open() as stream:
        for line in stream:
            row = json.loads(line)
            timing = row.get('timing') or {}
            stamp = timing.get('native_timestamp_s')
            record = records.get(stamp)
            host = row['host_recorded_monotonic_s']
            measurement = timing.get('oldest_host_arrival_monotonic_s', host)
            fusion = row.get('fusion') or {}
            valid = (fusion.get('position_output_m') is not None and not row.get('stale', True)
                     and not fusion.get('stale', True))
            item = dict(host=host, measurement=measurement, record=record, original_valid=valid,
                        raw_position=fusion.get('position_output_m'))
            observations.append(item)
            if valid:
                if record is None:
                    raise ValueError('Export observation is missing from complete perception log')
                positions.append(item)
    observations.sort(key=lambda r:r['host']); positions.sort(key=lambda r:r['host'])
    result = dict(observations=observations, positions=positions,
                  observation_times=[r['host'] for r in observations], position_times=[r['host'] for r in positions])
    return result, dict(status='replayed', counts=dict(counts),
                        source_files={str(p):sha256(p) for p in needed},
                        velocity_method='Replay causal VelocityTracker on corrected points at original native capture timestamps',
                        position_method='Reproject stored front visible-surface point on its recorded ray; segmentation/ray selection retained')


def past(items, times, now):
    i = bisect_right(times, now)-1
    return items[i] if i >= 0 else None


def correct_episode(rows, original_quality, phase_quality, replay, model):
    rows = copy.deepcopy(rows); quality = copy.deepcopy(original_quality)
    if not len(rows) == len(quality) == len(phase_quality):
        raise ValueError('Frame and quality lengths differ')
    model_hash = digest(model); required = []
    for row, q, phase in zip(rows, quality, phase_quality):
        now = q['anchor_monotonic_s']; env = np.array(row['observation.environment_state'], float)
        required.append(bool(phase['velocity_required']))
        q.update(manipulation_phase=phase['manipulation_phase'], phase_confirmed=phase['phase_confirmed'],
                 original_position_current=q['position_current'],
                 original_velocity_estimated=q['velocity_estimated'], original_velocity_zero_filled=q['velocity_zero_filled'],
                 depth_correction_model_sha256=model_hash, depth_position_corrected=False,
                 depth_correction_extrapolated=False, depth_velocity_replayed=False)
        p = o = None
        if replay is not None:
            p = past(replay['positions'], replay['position_times'], now)
            o = past(replay['observations'], replay['observation_times'], now)
        if p is not None:
            if not np.allclose(env[:3], p['raw_position'], atol=1e-6, rtol=0):
                raise ValueError('Original anchor position does not match the causal raw observation')
            if abs(now-p['measurement']-q['position_age_s']) > 1e-6:
                raise ValueError('Original position measurement timestamp differs')
            if p['record']['position'] is not None:
                env[:3] = p['record']['position']
                q.update(depth_position_corrected=True,
                         depth_correction_extrapolated=p['record']['extrapolated'])
        q['position_current'] = bool(q['original_position_current'] and q['depth_position_corrected'])
        velocity = None
        if o is not None and o['original_valid'] and o['record'] is not None and 0 <= now-o['measurement'] <= .3:
            velocity = o['record']['velocity']
        env[6:9] = [0., 0., 0.] if velocity is None else velocity
        q.update(velocity_estimated=velocity is not None, velocity_zero_filled=velocity is None,
                 depth_velocity_replayed=velocity is not None,
                 depth_correction_status='corrected' if q['depth_position_corrected'] else 'unavailable')
        row['observation.environment_state'] = env.astype(np.float32).tolist()
    extrapolation_by_time = {q['anchor_monotonic_s']-q['position_age_s']:q['depth_correction_extrapolated']
                             for q in quality if q['position_current']}
    rows, quality = repair_tracking(rows, quality, max_gap_s=.2, required=required)
    for q in quality:
        if q['position_repaired']:
            q.update(depth_position_corrected=True, depth_correction_status='interpolated_corrected_positions')
        if q.get('repair_measurement_bracket_s'):
            q['depth_correction_extrapolated'] |= any(extrapolation_by_time[t] for t in q['repair_measurement_bracket_s'])
        if replay is None:
            # Preserve the demonstration for baseline training, but no geometry
            # anchors or environment statistics can claim corrected coordinates.
            q.update(tracking_usable=False, depth_correction_status='missing_raw_tracking')
    return rows, quality


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, help='Quality-processed dataset supplying reviewed phases/splits')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.dataset.resolve(); output = args.output.resolve()
    model = json.loads(args.model.read_text())
    if (root/'depth_reprocessing_report.json').exists():
        raise ValueError('Input already reprocessed; use the original quality version')
    original = Path(json.loads((root/'export_report.json').read_text())['source_dataset'])
    sources = jsonl_read(root/'meta/sources.jsonl')
    cameras = jsonl_read(root/'meta/camera.jsonl')
    episode_stats = jsonl_read(root/'meta/episodes_stats.jsonl')
    policy = json.loads((root/'meta/quality_policy.json').read_text())
    old_audit = json.loads((root/'quality_audit.json').read_text())
    splits = json.loads((root/'meta/splits.json').read_text())
    groups = defaultdict(list)
    for source in sources: groups[source['raw_session']].append(source['episode_index'])
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(root/'meta', output/'meta')
    (output/'videos').symlink_to((root/'videos').resolve(), target_is_directory=True)
    json_write(output/'export_report.json', dict(complete=False, source_dataset=str(root)))
    total = Counter(); episode_reports = {}; session_reports = {}; usable = {}
    for session, ids in groups.items():
        print(f'Replaying {Path(session).parent.name}: {len(ids)} episodes', flush=True)
        replay, session_report = replay_session(session, model)
        session_reports[session] = session_report
        for ep in ids:
            relative = DATA_PATH.format(episode_chunk=ep//1000, episode_index=ep)
            before = pq.read_table(original/relative).to_pylist()
            qname = f'quality/episode_{ep:06d}.jsonl'
            rows, quality = correct_episode(before, jsonl_read(original/qname), jsonl_read(root/qname), replay, model)
            # Enforce preservation as part of writing the derived artifact.
            for a, b in zip(before, rows):
                for key in a:
                    if key != 'observation.environment_state' and a[key] != b[key]:
                        raise ValueError(f'Unexpected robot/action/time change in episode {ep}: {key}')
                if a['observation.environment_state'][3:6] != b['observation.environment_state'][3:6]:
                    raise ValueError('Assumed object orientation changed')
                if not np.isfinite(b['observation.environment_state']).all():
                    raise ValueError('Nonfinite environment state')
            counts = Counter(frames=len(rows))
            for q in quality:
                for key in ('depth_position_corrected','depth_velocity_replayed','depth_correction_extrapolated',
                            'position_repaired','velocity_repaired','tracking_usable'):
                    counts[key] += bool(q[key])
            total.update(counts)
            usable[str(ep)] = [i for i,q in enumerate(quality) if q['tracking_usable']]
            write_parquet(rows, output/relative); jsonl_write(output/qname, quality)
            # Geometry statistics exclude unavailable tracking and uncorrected
            # placeholder positions. Robot/image statistics retain all frames.
            geometry_rows = [r['observation.environment_state'] for r,q in zip(rows,quality)
                             if q['tracking_usable'] and q['depth_position_corrected']]
            episode_stats[ep]['stats'].pop('observation.environment_state', None)
            if geometry_rows:
                episode_stats[ep]['stats']['observation.environment_state'] = numeric_stats(geometry_rows)
            for camera in cameras[ep]['cameras']:
                if camera['source_role'] != 'wrist':
                    camera['original_calibration_sha256'] = digest(camera['calibration'])
                    if replay is not None:
                        camera['calibration']['depth_corrections'] = {'front':model}
                camera['label_depth_reprocessed'] = replay is not None
            sources[ep]['depth_reprocessing'] = dict(model_sha256=digest(model),
                                                    status=session_report['status'])
            episode_reports[str(ep)] = dict(episode_index=ep, counts=dict(counts),
                                            geometry_statistics_frames=len(geometry_rows),
                                            status=session_report['status'])
        del replay
    jsonl_write(output/'meta/camera.jsonl', cameras)
    jsonl_write(output/'meta/sources.jsonl', sources)
    jsonl_write(output/'meta/episodes_stats.jsonl', episode_stats)
    from lerobot.common.datasets.compute_stats import aggregate_stats
    train_stats = aggregate_stats([{k:{n:np.asarray(v) for n,v in values.items()}
                                    for k,values in episode_stats[ep]['stats'].items()} for ep in splits['train']])
    json_write(output/'meta/train_stats.json', {k:{n:v.tolist() for n,v in values.items()} for k,values in train_stats.items()})
    acceptance = dataset_acceptance(output, cameras)
    if not acceptance:
        raise ValueError('Accepted transform provenance lost during reprocessing')
    policy.update(usable_frames=usable, base_calibration_acceptance=acceptance,
                  depth_correction_model_sha256=digest(model), depth_extrapolation_included=True,
                  geometry_statistics='Only usable, depth-corrected environment rows; other features unchanged')
    json_write(output/'meta/quality_policy.json', policy)
    report = dict(schema='krvla_depth_reprocessing_v1', complete=True, source_dataset=str(root),
                  original_unrepaired_dataset=str(original), model=model, model_sha256=digest(model),
                  totals=dict(total), episodes=episode_reports, sessions=session_reports,
                  excluded_geometry_episodes=[ep for ep in range(len(sources)) if not usable[str(ep)]],
                  preserved='Actions, robot states, object orientations, frame indices, timestamps, splits and RGB videos',
                  limitation='Stored segmentation rays retained; no mask inference rerun. Extrapolation beyond board range is flagged, not independently validated.',
                  source_quality_policy_sha256=sha256(root/'meta/quality_policy.json'))
    json_write(output/'depth_reprocessing_report.json', report)
    old_audit.update(dataset=str(output), base_calibration_acceptance=acceptance, depth_reprocessing='depth_reprocessing_report.json',
                     depth_reprocessing_totals=dict(total))
    for ep_report in old_audit['episode_reports']:
        counts = episode_reports[str(ep_report['episode_index'])]['counts']
        for key in ('position_repaired','velocity_repaired'): ep_report[key] = counts[key]
    for key in ('position_repaired','velocity_repaired'): old_audit['totals'][key] = total[key]
    json_write(output/'quality_audit.json', old_audit)
    json_write(output/'export_report.json', dict(complete=True, episodes=len(sources), frames=total['frames'],
               source_dataset=str(root), depth_reprocessing='depth_reprocessing_report.json',
               quality_processing='quality_audit.json'))
    print(json.dumps(dict(output=str(output), totals=dict(total), excluded_geometry_episodes=report['excluded_geometry_episodes']), indent=2), flush=True)


if __name__ == '__main__':
    main()
