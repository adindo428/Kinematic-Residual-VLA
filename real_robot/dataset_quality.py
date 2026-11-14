"""Audit and conservatively repair DOM tracking without changing raw recordings.

Run ``python -m real_robot.dataset_quality --help``. Outputs are new local
versions; no robot connections or Hub writes are made.
"""
import argparse
from collections import Counter, defaultdict
import copy
import itertools
import json
from pathlib import Path
import shutil
from bisect import bisect_right

import numpy as np
import pyarrow.parquet as pq

from .dom_dataset import DATA_PATH, json_write, jsonl_read, jsonl_write, numeric_stats, sha256, write_parquet
from .calibration_acceptance import dataset_acceptance
from .static_object import quality_valid as static_quality_valid,SCHEMA as STATIC_SCHEMA,AUTOMATIC_PHASE_METHOD


def phases(rows, quality, events=None, held_intervals=()):
    """Telemetry candidates are review aids, never proof of object attachment."""
    n = len(rows)
    labels = ['unknown'] * n
    opened = False
    close_z = None
    for i, row in enumerate(rows):
        grip, z = row['action'][-1], row['observation.state'][2]
        if grip > .25:
            opened, close_z = True, None
            labels[i] = 'open_candidate'
        elif opened and grip <= 0:
            if close_z is None:
                close_z = z
            labels[i] = 'lift_candidate' if z - close_z >= .03 else 'closing_candidate'
        elif close_z is not None:
            labels[i] = 'closing_candidate'
    required = np.ones(n, dtype=bool)
    for i,q in enumerate(quality):
        if (q.get('static_collection_schema')==STATIC_SCHEMA and q.get('static_automatic_held') is True
                and q.get('static_automatic_method')==AUTOMATIC_PHASE_METHOD and q.get('episode_success') is True):
            labels[i]='held_success_inferred';required[i]=False
    # Markers narrow video review; a lift marker alone does not prove holding.
    events = events or {}
    if events.get('lift') is not None:
        start = next((i for i, q in enumerate(quality) if q['anchor_monotonic_s'] >= events['lift']), n)
        end = next((i for i, q in enumerate(quality) if q['anchor_monotonic_s'] >= events.get('release', float('inf'))), n)
        labels[start:end] = ['marked_lift_candidate'] * max(0, end-start)
    for start, end in held_intervals:
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end <= n:
            raise ValueError('Held intervals must be [start_frame, exclusive_end_frame] within the episode')
        labels[start:end] = ['held_confirmed'] * (end-start)
        required[start:end] = False
    # Unlike the legacy L candidate marker, static-mode L is an explicit
    # operator confirmation of visible attachment and lift, saved separately.
    confirmed=events.get('operator_visible_held')
    if confirmed is not None:
        release=events.get('release',float('inf'))
        if (not np.isfinite(confirmed) or release<=confirmed
                or not all(q.get('static_collection_schema')==STATIC_SCHEMA for q in quality)):
            raise ValueError('Invalid static operator holding confirmation')
        for i,q in enumerate(quality):
            if confirmed<=q['anchor_monotonic_s']<release:
                labels[i]='held_confirmed';required[i]=False
    return labels, required


def repair_tracking(rows, quality, *, max_gap_s=.20, max_speed_mps=2., required=None):
    """Interpolate short bracketed gaps using *measurement* times, not held rows.

    Uses future measurements offline. Never extrapolates or crosses a confirmed
    phase boundary. Original flags survive; repair provenance is separate.
    """
    rows, quality = copy.deepcopy(rows), copy.deepcopy(quality)
    n = len(rows)
    if n != len(quality) or any(q['frame_index'] != i for i, q in enumerate(quality)):
        raise ValueError('Quality/frame alignment mismatch')
    t = np.array([q['anchor_monotonic_s'] for q in quality], float)
    if not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError('Anchor times must be finite and strictly increasing')
    required = np.ones(n, bool) if required is None else np.asarray(required, bool)
    env = np.array([r['observation.environment_state'] for r in rows], float)
    # One accepted observation may be held across several export anchors.
    measurements = []
    for i, q in enumerate(quality):
        mt = t[i] - q['position_age_s']
        if q['position_current'] and (not measurements or mt > measurements[-1][0] + 1e-6):
            measurements.append((mt, i, env[i, :3].copy()))
    mt = np.array([x[0] for x in measurements])
    for i, q in enumerate(quality):
        pv = bool(q['position_current']) or static_quality_valid(q)
        vv = bool(q['velocity_estimated'] and not q['velocity_zero_filled']) or static_quality_valid(q)
        q.update(position_repaired=False, velocity_repaired=False,
                 velocity_required=bool(required[i]), tracking_repair_uses_future=False)
        if not required[i]:
            # Neutral model input during labeled holding; not physical zero velocity.
            env[i, 6:9] = 0
            q['velocity_input_source'] = ('neutral_after_success_supported_grasp' if q.get('static_automatic_held')
                                          else 'neutral_after_confirmed_lift')
        elif not (pv and vv) and len(mt) >= 2 and q.get('static_collection_schema')!=STATIC_SCHEMA:
            right = int(np.searchsorted(mt, t[i], side='right'))
            if 0 < right < len(mt):
                ta, ia, pa = measurements[right-1]
                tb, ib, pb = measurements[right]
                same_phase = required[min(ia, i):max(ib, i)+1].all()
                # Do not bridge a gripper transition, even without phase labels.
                grips = np.array([r['action'][-1] for r in rows[min(ia,i):max(ib,i)+1]])
                same_grip = not ((grips > .25).any() and (grips <= 0).any())
                velocity = (pb-pa)/(tb-ta)
                if tb-ta <= max_gap_s + 1e-8 and same_phase and same_grip and np.linalg.norm(velocity) <= max_speed_mps:
                    if not pv:
                        env[i, :3] = pa + (t[i]-ta)*velocity
                        q['position_repaired'] = pv = True
                    if not vv:
                        env[i, 6:9] = velocity
                        q['velocity_repaired'] = vv = True
                    q.update(tracking_repair_uses_future=True,
                             repair_measurement_bracket_s=[float(ta), float(tb)])
                    if any(quality[k].get('depth_correction_extrapolated',False) for k in (ia,ib)):
                        q['depth_correction_extrapolated'] = True
        q['tracking_usable'] = bool((pv and vv) or not required[i])
        rows[i]['observation.environment_state'] = env[i].astype(np.float32).tolist()
    return rows, quality


def session_splits(sources, episodes):
    """Deterministic task-stratified whole-session split; fail if impossible."""
    by_task = defaultdict(lambda: defaultdict(list))
    for s, e in zip(sources, episodes):
        by_task[tuple(e['tasks'])][s['raw_session']].append(e['episode_index'])
    splits = {k: [] for k in ('train', 'validation', 'test')}
    assignment = {}
    for task, sessions in sorted(by_task.items()):
        if len(sessions) < 3:
            raise ValueError(f'Task needs at least three sessions for disjoint train/validation/test: {task}')
        total = sum(map(len, sessions.values()))
        names = sorted(sessions)
        # Choose held-out sessions closest to 10% each, retaining >=1 train session.
        val, test = min(itertools.permutations(names, 2),
                        key=lambda pair: (abs(len(sessions[pair[0]])/total-.1) + abs(len(sessions[pair[1]])/total-.1), pair))
        for session, ids in sessions.items():
            split = 'validation' if session == val else 'test' if session == test else 'train'
            if session in assignment and assignment[session] != split:
                raise ValueError('Multi-task session would cross splits; use an explicit split file')
            assignment[session] = split
            splits[split].extend(ids)
    return {k: sorted(v) for k, v in splits.items()}


def prepare_review(dataset, output, episode_ids=None):
    """Wrist/front contact sheets at sustained measured-close + EE-lift candidates.

    Visual object retention must still be reviewed. Missing robot records never
    become positive grasp labels. JPEGs use exported video time, not raw indices.
    """
    import cv2
    root, output = Path(dataset), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources = jsonl_read(root/'meta/sources.jsonl')
    if episode_ids is not None:
        sources = [s for s in sources if s['episode_index'] in set(episode_ids)]
    cache, candidates = {}, []
    for source in sources:
        ep, session = source['episode_index'], Path(source['raw_session'])
        if session not in cache:
            times, measured = [], []
            if (session/'robot.jsonl').exists():
                manifest = json.loads((session/'manifest.json').read_text())
                closed, opened = manifest['gripper_closed_percent'], manifest['gripper_open_percent']
                with (session/'robot.jsonl').open() as stream:
                    for line in stream:
                        r = json.loads(line)
                        if r.get('follower_calibrated') is not None:
                            times.append(r['host_read_end_monotonic_s'])
                            measured.append(2*(r['follower_calibrated'][-1]-closed)/(opened-closed)-1)
            cache[session] = (times, measured)
        times, measured = cache[session]
        rows = pq.read_table(root/DATA_PATH.format(episode_chunk=ep//1000,episode_index=ep)).to_pylist()
        quality = jsonl_read(root/f'quality/episode_{ep:06d}.jsonl')
        close_start, close_z, seen_open = None, None, False
        selected = []
        for i, (row,q) in enumerate(zip(rows,quality)):
            now = q['anchor_monotonic_s']
            j = bisect_right(times, now)-1
            grip = measured[j] if j>=0 and 0<=now-times[j]<=.12 else None
            if grip is not None and grip > .25:
                seen_open, close_start, close_z = True, None, None
            elif grip is not None and grip<=0 and seen_open:
                if close_start is None:
                    close_start, close_z = i, row['observation.state'][2]
                if i-close_start>=5 and row['observation.state'][2]-close_z>=.03:
                    selected.append(dict(frame=i,close_frame=close_start,measured_gripper=float(grip),
                                         ee_lift_m=row['observation.state'][2]-close_z))
                    seen_open = False  # one review strip per close/lift, reset on opening
            else:
                close_start, close_z = None, None
        # Include the worst stale interval even if the concurrence test found no lift.
        worst = max(range(len(quality)),key=lambda i:quality[i]['position_age_s'])
        centers = sorted(set([v['frame'] for v in selected]+[worst]))
        frames = sorted(set(max(0,min(len(rows)-1,c+d)) for c in centers for d in [-10,0,10,25]))
        panels=[]
        for role,key in [('wrist','wrist_cam'),('front','opst_cam')]:
            video=root/f'videos/chunk-{ep//1000:03d}/observation.images.{key}/episode_{ep:06d}.mp4'
            cap=cv2.VideoCapture(str(video));tiles=[]
            for i in frames:
                cap.set(cv2.CAP_PROP_POS_MSEC, rows[i]['timestamp']*1000)
                ok,frame=cap.read()
                if not ok:
                    raise ValueError(f'Cannot decode review frame {ep}:{i}:{role}')
                tile=cv2.resize(frame,(240,240))
                q=quality[i]
                cv2.rectangle(tile,(0,0),(240,42),(0,0,0),-1)
                cv2.putText(tile,f'{role} ep{ep} f{i} {rows[i]["timestamp"]:.2f}s',(4,15),0,.36,(255,255,255),1)
                cv2.putText(tile,f'z={rows[i]["observation.state"][2]:.3f} stale={not q["position_current"]} vmiss={q["velocity_zero_filled"]}',(4,32),0,.32,(255,255,255),1)
                tiles.append(tile)
            cap.release()
            # Rows of at most eight frames keep contact sheets inspectable.
            for start in range(0,len(tiles),8):
                chunk=tiles[start:start+8]
                chunk += [np.zeros_like(chunk[0]) for _ in range(8-len(chunk))]
                panels.append(np.concatenate(chunk,axis=1))
        image_name=f'episode_{ep:06d}.jpg'
        cv2.imwrite(str(output/image_name),np.concatenate(panels,axis=0))
        candidates.append(dict(episode_index=ep, source_episode=source['source_episode'],
                               lift_candidates=selected, worst_stale_frame=worst, contact_sheet=image_name,
                               confirmed=False))
    json_write(output/'review_candidates.json',candidates)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Grasp review</title><style>img{max-width:100%}body{font-family:sans-serif}</style><h1>Wrist + measured gripper + EE lift review</h1><p>Candidates are unconfirmed. Inspect visible object retention during lift before recording held intervals.</p>'+''.join(f'<h2>Episode {c["episode_index"]}</h2><p>{c["lift_candidates"]}</p><img loading="lazy" src="{c["contact_sheet"]}">' for c in candidates))
    return candidates


def process(dataset, output, *, repair=False, max_gap_s=.20, annotations=None, make_splits=False, split_override=None):
    root, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists() or root == output or root in output.parents:
        raise ValueError('Choose a new output directory outside the input dataset')
    info = json.loads((root/'meta/info.json').read_text())
    sources = jsonl_read(root/'meta/sources.jsonl')
    episodes = jsonl_read(root/'meta/episodes.jsonl')
    cameras = jsonl_read(root/'meta/camera.jsonl')
    if not len(sources) == len(episodes) == len(cameras) == info['total_episodes']:
        raise ValueError('Source/camera/episode metadata lengths differ')
    annotations = annotations or {}
    split = split_override or (session_splits(sources, episodes) if make_splits else json.loads((root/'meta/splits.json').read_text()))
    if sorted(ep for ids in split.values() for ep in ids) != list(range(len(episodes))):
        raise ValueError('Splits must cover every episode exactly once')
    if set(split) != {'train','validation','test'} or not any(split.values()):
        raise ValueError('Provide train/validation/test lists and at least one episode')
    session_membership = defaultdict(set)
    for name, ids in split.items():
        for ep in ids:
            session_membership[sources[ep]['raw_session']].add(name)
    shared_sessions = {k:sorted(v) for k,v in session_membership.items() if len(v)>1}
    original_stats = jsonl_read(root/'meta/episodes_stats.jsonl')
    output.mkdir(parents=True)
    counts, summaries, usable = Counter(), [], {}
    stats = []
    calibration_ok = True
    for source, episode, camera in zip(sources, episodes, cameras):
        ep = episode['episode_index']
        if source['episode_index'] != ep or camera['episode_index'] != ep:
            raise ValueError('Source/camera indices differ from episode metadata')
        path = root/DATA_PATH.format(episode_chunk=ep//1000, episode_index=ep)
        rows = pq.read_table(path).to_pylist()
        qpath = root/f'quality/episode_{ep:06d}.jsonl'
        quality = jsonl_read(qpath)
        if len(rows) != len(quality):
            raise ValueError(f'Quality length mismatch in episode {ep}')
        raw_episode = Path(source['source_episode'])
        events = json.loads(raw_episode.read_text()).get('events', {}) if raw_episode.exists() else {}
        ann = annotations.get(str(ep), {})
        if ann and not ann.get('review_note'):
            raise ValueError('Reviewed annotations need a review_note')
        labels, required = phases(rows, quality, events, ann.get('held_intervals', ()))
        before = Counter()
        for label, q in zip(labels, quality):
            before[f'{label}:frames'] += 1
            before[f'{label}:stale_position'] += not q['position_current']
            before[f'{label}:missing_velocity'] += q['velocity_zero_filled']
        angle_crossings = int((np.abs(np.array([r['action'][:6] for r in rows])[:,3:6] -
                                     np.array([r['observation.state'] for r in rows])[:,3:6]) > np.pi).any(axis=1).sum())
        if repair:
            rows, quality = repair_tracking(rows, quality, max_gap_s=max_gap_s, required=required)
        for label, q in zip(labels, quality):
            q['manipulation_phase'] = label
            q['phase_confirmed'] = label == 'held_confirmed'
        phase_counts = dict(before)
        per_ep = dict(episode_index=ep, task=episode['tasks'], source_episode=source['source_episode'],
                      frames=len(rows), phase_counts=phase_counts, angle_crossing_frames=angle_crossings,
                      max_position_age_s=max(q['position_age_s'] for q in quality),
                      position_repaired=sum(q.get('position_repaired',False) for q in quality),
                      velocity_repaired=sum(q.get('velocity_repaired',False) for q in quality))
        summaries.append(per_ep)
        counts.update(before)
        counts['frames'] += len(rows)
        counts['angle_crossing_frames'] += angle_crossings
        counts['position_repaired'] += per_ep['position_repaired']
        counts['velocity_repaired'] += per_ep['velocity_repaired']
        calibration_ok &= all(c['calibration'].get('base_calibration_validated', False)
                              for c in camera['cameras'] if c['source_role'] != 'wrist')
        if repair:
            usable[str(ep)] = [i for i,q in enumerate(quality) if q['tracking_usable']]
            write_parquet(rows, output/path.relative_to(root))
            jsonl_write(output/qpath.relative_to(root), quality)
            stats.append(dict(episode_index=ep, stats={**original_stats[ep]['stats'],
                         **{k:numeric_stats([r[k] for r in rows]) for k in rows[0]}}))
    acceptance = dataset_acceptance(root, cameras)
    report = dict(schema='krvla_quality_audit_v1', dataset=str(root), input_info_sha256=sha256(root/'meta/info.json'),
                  episodes=len(summaries), totals=dict(counts), base_calibration_validated=bool(calibration_ok),
                  base_calibration_acceptance=acceptance,
                  phase_note='Legacy markers are candidates. Static mode supports separately labeled automatic success+measured-closure/lift intervals; explicit operator confirmations and reviewed intervals remain distinct.',
                  repair_uses_future=repair, max_interpolation_gap_s=max_gap_s, episode_reports=summaries,
                  shared_split_sessions=shared_sessions, splits=split)
    report['reviewed_phase_annotations'] = annotations
    json_write(output/'quality_audit.json', report)
    if make_splits or split_override:
        json_write(output/'splits.json', split)
    if repair:
        shutil.copytree(root/'meta', output/'meta')
        # A separate version references original videos without modifying them.
        (output/'videos').symlink_to(root/'videos', target_is_directory=True)
        jsonl_write(output/'meta/episodes_stats.jsonl', stats)
        json_write(output/'meta/splits.json', split)
        split_for = {ep:k for k,ids in split.items() for ep in ids}
        for s in sources:
            s['split'] = split_for[s['episode_index']]
        jsonl_write(output/'meta/sources.jsonl', sources)
        # Lists may be noncontiguous; never advertise a false contiguous range.
        info['splits'] = split
        json_write(output/'meta/info.json', info)
        from lerobot.common.datasets.compute_stats import aggregate_stats
        train = [{k:{n:np.asarray(v) for n,v in values.items()} for k,values in stats[ep]['stats'].items()} for ep in split['train']]
        if train:
            aggregate = aggregate_stats(train)
            json_write(output/'meta/train_stats.json', {k:{n:v.tolist() for n,v in values.items()} for k,values in aggregate.items()})
        elif (output/'meta/train_stats.json').exists():
            (output/'meta/train_stats.json').unlink()
        json_write(output/'meta/quality_policy.json', dict(schema='krvla_quality_policy_v1',
                   usable_frames=usable, base_calibration_validated=bool(calibration_ok),
                   base_calibration_acceptance=acceptance,
                   angle_delta='wrapped_xyz_v1', uses_future_interpolation=True,
                   static_object_assumptions='krvla_static_object_v1: per-episode measured position seed; declared zero approach velocity; invalidation and phase boundaries retained',
                   held_velocity='neutral input after confirmed lift; not physical stationary velocity'))
        json_write(output/'export_report.json', dict(complete=True, episodes=len(episodes),
                   frames=info['total_frames'], source_dataset=str(root), quality_processing='quality_audit.json'))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repair', action='store_true')
    parser.add_argument('--make-session-splits', action='store_true')
    parser.add_argument('--split-file', type=Path, help='Explicit split lists; sharing sessions is reported')
    parser.add_argument('--max-gap-s', type=float, default=.20)
    parser.add_argument('--phase-annotations', type=Path,
                        help='JSON mapping episode IDs to {held_intervals:[[start,end]], review_note:...}')
    parser.add_argument('--prepare-review', action='store_true', help='Create wrist/front contact sheets with measured-gripper + lift candidates')
    parser.add_argument('--review-episodes', type=int, nargs='+', help='Optional episode IDs for review sheets')
    args = parser.parse_args(argv)
    if args.prepare_review:
        candidates=prepare_review(args.dataset,args.output,args.review_episodes)
        print(f'Prepared {len(candidates)} episode review sheets: {args.output/"index.html"}')
        return
    if not np.isfinite(args.max_gap_s) or not 0 < args.max_gap_s <= .5:
        parser.error('max-gap-s must be in (0, 0.5]')
    report = process(args.dataset, args.output, repair=args.repair, max_gap_s=args.max_gap_s,
                     annotations=json.loads(args.phase_annotations.read_text()) if args.phase_annotations else None,
                     make_splits=args.make_session_splits,
                     split_override=json.loads(args.split_file.read_text()) if args.split_file else None)
    print(json.dumps({k:v for k,v in report.items() if k != 'episode_reports'}, indent=2))


if __name__ == '__main__':
    main()
