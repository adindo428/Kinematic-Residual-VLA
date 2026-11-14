"""Build and publish the processed KRVLA training branch after raw append uploads."""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from collections import Counter, defaultdict
import copy
import fcntl
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq

from .calibration_acceptance import accepted_transform, dataset_acceptance
from .dataset_quality import phases, repair_tracking
from .depth_calibration import digest
from .dom_dataset import DATA_PATH, json_write, jsonl_read, jsonl_write, numeric_stats, sha256, validate_dataset, write_parquet

PROJECT = Path(__file__).resolve().parents[1]
REPO = 'ailab-Clemson/KRVLA-dataset'
BRANCH = 'training'
STATE = PROJECT/'scratch/real_robot/training_releases/latest.json'
BASELINE = PROJECT/'scratch/real_robot/KRVLA-quality-319-v3-depth'
MODEL = PROJECT/'real_robot/calibration/depth_correction.json'


def validate_capture_rig(rig):
    if not accepted_transform(rig):
        raise ValueError('Collection transform is not the accepted camera-to-base transform')
    expected = json.loads(MODEL.read_text())
    if rig.get('depth_corrections', {}).get('front') != expected:
        raise ValueError('KRVLA collection requires the current depth correction; use real_robot/config_collection.yaml')


def raw_hashes(raw, count):
    result = {}
    for ep in range(count):
        for name in (DATA_PATH.format(episode_chunk=ep//1000, episode_index=ep), f'quality/episode_{ep:06d}.jsonl'):
            result[name] = sha256(raw/name)
    return result


def prepare(raw, baseline, output, raw_revision):
    raw, baseline, output = map(lambda p:Path(p).resolve(), (raw, baseline, output))
    if output.exists():
        report = json.loads((output/'training_release.json').read_text())
        if not report.get('complete') or report['raw_revision'] != raw_revision:
            raise ValueError('Existing release directory is incomplete or belongs to another revision')
        return report
    info = json.loads((raw/'meta/info.json').read_text())
    old_info = json.loads((baseline/'meta/info.json').read_text())
    total, old_total = info['total_episodes'], old_info['total_episodes']
    if total < old_total:
        raise ValueError('Raw dataset predates the processed baseline')
    for key in ('fps','features','robot_type'):
        if info[key] != old_info[key]: raise ValueError('Dataset schema changed: '+key)
    sources, cameras, episodes, stats = [jsonl_read(raw/f'meta/{name}.jsonl')
                                        for name in ('sources','camera','episodes','episodes_stats')]
    old_sources, old_cameras, old_stats = [jsonl_read(baseline/f'meta/{name}.jsonl')
                                         for name in ('sources','camera','episodes_stats')]
    for i in range(old_total):
        if any(sources[i][key] != old_sources[i][key] for key in ('episode_index','source_episode','raw_session')):
            raise ValueError('Raw source prefix differs from processed baseline')
    hashes = raw_hashes(raw, total)
    if (baseline/'training_release.json').exists():
        previous = json.loads((baseline/'training_release.json').read_text())
        expected_hashes = previous['raw_episode_hashes']
    else:
        previous = json.loads((baseline/'depth_reprocessing_report.json').read_text())
        expected_hashes = raw_hashes(Path(previous['original_unrepaired_dataset']), old_total)
    if any(hashes.get(k) != v for k,v in expected_hashes.items()):
        raise ValueError('Raw episode prefix changed; refusing to reuse processed labels')
    splits = json.loads((baseline/'meta/splits.json').read_text())
    for ep in range(old_total, total):
        splits[sources[ep]['split']].append(ep)
    # Replace provisional within-session holdouts once the operator has actually
    # collected dedicated validation AND test sessions for that task.
    tasks = defaultdict(list)
    for ep, episode in enumerate(episodes): tasks[tuple(episode['tasks'])].append(ep)
    finalized_tasks = []
    for task, ids in tasks.items():
        if {'train','validation','test'} <= {sources[ep]['split'] for ep in ids}:
            membership = defaultdict(set)
            for ep in ids: membership[sources[ep]['raw_session']].add(sources[ep]['split'])
            if all(len(v) == 1 for v in membership.values()):
                selected = set(ids)
                for name in splits: splits[name] = [ep for ep in splits[name] if ep not in selected]
                for ep in ids: splits[sources[ep]['split']].append(ep)
                finalized_tasks.append(list(task))
    splits = {name:sorted(ids) for name,ids in splits.items()}
    if sorted(ep for ids in splits.values() for ep in ids) != list(range(total)) or any(not ids for ids in splits.values()):
        raise ValueError('Need complete, nonempty train/validation/test splits')
    model = json.loads(MODEL.read_text())
    model_hash = digest(model)
    policy = json.loads((baseline/'meta/quality_policy.json').read_text())
    if policy.get('depth_correction_model_sha256') != model_hash:
        raise ValueError('Baseline depth model differs; reprocess before combining calibrations')
    staging = output.with_name(output.name+'.partial')
    staging.mkdir(parents=True, exist_ok=False)
    shutil.copytree(raw/'meta', staging/'meta')
    (staging/'videos').symlink_to((raw/'videos').resolve(), target_is_directory=True)
    usable = {}; episode_reports = []; counts = Counter(); new_counts = Counter()
    for ep in range(total):
        name = DATA_PATH.format(episode_chunk=ep//1000, episode_index=ep)
        qname = f'quality/episode_{ep:06d}.jsonl'
        if ep < old_total:
            rows = pq.read_table(baseline/name).to_pylist()
            quality = jsonl_read(baseline/qname)
            destination = staging/name; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(baseline/name, destination)
            (staging/'quality').mkdir(exist_ok=True); shutil.copy2(baseline/qname, staging/qname)
            sources[ep], cameras[ep], stats[ep] = copy.deepcopy(old_sources[ep]), copy.deepcopy(old_cameras[ep]), copy.deepcopy(old_stats[ep])
        else:
            fixed = [c for c in cameras[ep]['cameras'] if c['source_role'] != 'wrist']
            if not fixed: raise ValueError('Missing fixed-camera provenance')
            for camera in fixed: validate_capture_rig(camera['calibration'])
            rows = pq.read_table(raw/name).to_pylist(); quality = jsonl_read(raw/qname)
            raw_rows = copy.deepcopy(rows)
            if len(rows) != len(quality): raise ValueError('Quality sidecar length mismatch')
            source_episode = Path(sources[ep]['source_episode'])
            events = json.loads(source_episode.read_text()).get('events', {}) if source_episode.exists() else {}
            # Preserve separately identified static automatic success-supported
            # telemetry intervals and explicit operator holding confirmations.
            labels, required = phases(rows, quality, events)
            transform = np.asarray(fixed[0]['calibration']['T_base_front_candidate'])
            for r, q, label in zip(rows, quality, labels):
                camera = transform[:3,:3].T@(np.asarray(r['observation.environment_state'][:3])-transform[:3,3])
                raw_z = camera[2]/(1-model['inverse_depth_offset_per_m']*camera[2])
                lo, hi = model['raw_depth_domain_m']
                q.update(manipulation_phase=label, phase_confirmed=label=='held_confirmed',
                         depth_correction_model_sha256=model_hash, depth_position_corrected=True,
                         depth_correction_extrapolated=not lo <= raw_z <= hi,
                         depth_processing='applied_during_capture; no second depth correction')
            rows, quality = repair_tracking(rows, quality, max_gap_s=.2, required=required)
            for before, after in zip(raw_rows, rows):
                if any(before[k] != after[k] for k in before if k != 'observation.environment_state'):
                    raise ValueError('Quality repair changed robot/actions/timing')
            write_parquet(rows, staging/name); jsonl_write(staging/qname, quality)
            stats[ep]['stats'].pop('observation.environment_state', None)
            geometry = [r['observation.environment_state'] for r,q in zip(rows,quality) if q['tracking_usable']]
            if geometry: stats[ep]['stats']['observation.environment_state'] = numeric_stats(geometry)
            sources[ep]['depth_reprocessing'] = dict(status='corrected_during_capture', model_sha256=model_hash)
        usable[str(ep)] = [i for i,q in enumerate(quality) if q['tracking_usable']]
        summary = Counter(frames=len(rows), usable_geometry_frames=len(usable[str(ep)]))
        for q in quality:
            for key in ('position_repaired','velocity_repaired','depth_correction_extrapolated','phase_confirmed'):
                summary[key] += bool(q.get(key,False))
        counts.update(summary)
        if ep >= old_total: new_counts.update(summary)
        episode_reports.append(dict(episode_index=ep, **dict(summary)))
    split_for = {ep:name for name, ids in splits.items() for ep in ids}
    for source in sources: source['split'] = split_for[source['episode_index']]
    info['splits'] = splits
    json_write(staging/'meta/info.json', info); json_write(staging/'meta/splits.json', splits)
    for name, values in [('sources',sources),('camera',cameras),('episodes_stats',stats)]:
        jsonl_write(staging/f'meta/{name}.jsonl', values)
    from lerobot.common.datasets.compute_stats import aggregate_stats
    aggregate = aggregate_stats([{k:{n:np.asarray(v) for n,v in values.items()} for k,values in stats[ep]['stats'].items()}
                                 for ep in splits['train']])
    json_write(staging/'meta/train_stats.json', {k:{n:v.tolist() for n,v in values.items()} for k,values in aggregate.items()})
    acceptance = dataset_acceptance(staging, cameras)
    if not acceptance: raise ValueError('Dataset calibration acceptance does not match')
    policy.update(usable_frames=usable, base_calibration_acceptance=acceptance)
    json_write(staging/'meta/quality_policy.json', policy)
    memberships = defaultdict(set)
    for source in sources: memberships[source['raw_session']].add(source['split'])
    shared = {s:sorted(names) for s,names in memberships.items() if len(names)>1}
    report = dict(schema='krvla_training_release_v1', complete=True, raw_revision=raw_revision,
                  raw_dataset=str(raw), processed_baseline=str(baseline), baseline_episodes=old_total,
                  episodes=total, new_episodes=total-old_total, frames=info['total_frames'],
                  totals=dict(counts), new_episode_totals=dict(new_counts), model_sha256=model_hash,
                  raw_episode_hashes=hashes, split_counts={k:len(v) for k,v in splits.items()},
                  shared_split_sessions=shared, excluded_geometry_episodes=[int(k) for k,v in usable.items() if not v],
                  tasks_with_dedicated_holdouts=finalized_tasks,
                  calibration_status='operator_accepted_existing_transform',
                  phase_note='Only baseline reviewed held intervals are confirmed; new intervals remain unreviewed',
                  euler_delta_convention='wrapped_xyz_v1',
                  depth_note='Empirical RGB-referenced correction; extrapolated depths flagged; no second correction of new captures')
    json_write(staging/'training_release.json', report)
    json_write(staging/'quality_audit.json', dict(schema='krvla_quality_audit_v1', totals=dict(counts),
               episodes=total, episode_reports=episode_reports, shared_split_sessions=shared,
               base_calibration_validated=False, base_calibration_acceptance=acceptance))
    json_write(staging/'export_report.json', dict(complete=True, episodes=total, frames=info['total_frames'],
               training_release='training_release.json', source_dataset=str(raw), raw_revision=raw_revision))
    (staging/'README.md').write_text(
        '---\ntask_categories:\n- robotics\ntags:\n- lerobot\n---\n\n# KRVLA processed training dataset\n\n'
        f'{total} episodes, {info["total_frames"]} frames. Use revision `training` (pin its commit for reproducibility). '
        'The `main` branch preserves raw append-only collection exports.\n\n'
        f'Splits: {report["split_counts"]}. Splits are provisional while dedicated held-out sessions are collected. '
        'Some older orange/marker splits share sessions; see `training_release.json`.\n\n'
        'Includes corrected depth-derived positions/velocities, quality masks, bounded gap interpolation, explicit splits and train-only statistics. '
        'Use the updated repository loader with `euler_delta_convention="wrapped_xyz_v1"` and quality-aware KRV/DOMV policies. '
        'The existing camera-to-base transform is explicitly operator-accepted, not independently validated. '
        'Object positions are visible-surface estimates and object orientation is assumed. '
        'Depth extrapolation is flagged. Tracking loss is never treated as stationary.\n\n'
        'Reviewed/explicitly confirmed holding and separately identified static success-supported measured closure/lift intervals disable velocity requirements; automatic intervals are not manual visual confirmation. '
        'Inspect `quality_audit.json`, `meta/quality_policy.json` and `training_release.json`. '
        'Raw RGB-D and logs remain local.\n')
    validation = validate_dataset(staging, decode_videos=False)
    json_write(staging/'dataset_validation.json', {**validation, 'video_check':'headers; existing videos reused'})
    staging.rename(output)
    return report


def publish(api, repo, dataset, raw_revision):
    from huggingface_hub import CommitOperationAdd, CommitOperationCopy, hf_hub_download
    from .hub_upload import matches_remote
    dataset = Path(dataset)
    if api.repo_info(repo_id=repo, repo_type='dataset').sha != raw_revision:
        raise ValueError('Raw main changed; rerun training_release to include the latest append')
    api.create_branch(repo_id=repo, repo_type='dataset', branch=BRANCH, revision=raw_revision, exist_ok=True)
    head = api.repo_info(repo_id=repo, repo_type='dataset', revision=BRANCH).sha
    remote = {f.path:f for f in api.list_repo_tree(repo_id=repo,repo_type='dataset',revision=head,recursive=True) if hasattr(f,'blob_id')}
    raw_files = {f.path:f for f in api.list_repo_tree(repo_id=repo,repo_type='dataset',revision=raw_revision,recursive=True) if hasattr(f,'blob_id')}
    operations = []; expected = {}
    # Symlinked videos are already in raw main. Copy their LFS pointers between
    # branches server-side; never silently omit them or re-upload/re-encode video.
    for name, item in raw_files.items():
        if name.startswith('videos/'):
            if name not in remote or remote[name].blob_id != item.blob_id:
                operations.append(CommitOperationCopy(src_path_in_repo=name,path_in_repo=name,src_revision=raw_revision))
            expected[name] = item.blob_id
    for path in sorted(dataset.rglob('*')):
        if not path.is_file(): continue
        name = str(path.relative_to(dataset))
        if name.startswith('videos/') or name == '.gitattributes': continue
        if name not in remote or not matches_remote(path,remote[name]):
            operations.append(CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(path)))
    if operations:
        report = json.loads((dataset/'training_release.json').read_text())
        commit = api.create_commit(repo_id=repo,repo_type='dataset',revision=BRANCH,parent_commit=head,
            operations=operations,commit_message=f'Prepare {report["episodes"]} quality-aware KRVLA training episodes')
        head = commit.oid
    after = {f.path:f for f in api.list_repo_tree(repo_id=repo,repo_type='dataset',revision=head,recursive=True) if hasattr(f,'blob_id')}
    if any(name not in after or after[name].blob_id != oid for name,oid in expected.items()):
        raise RuntimeError('Training branch video pointers differ from raw upload')
    for path in dataset.rglob('*'):
        if not path.is_file(): continue
        name = str(path.relative_to(dataset))
        if name.startswith('videos/'): continue
        if name not in after or not matches_remote(path,after[name]):
            raise RuntimeError('Training branch verification failed: '+name)
    return dict(repo_id=repo, revision=BRANCH, commit=head, raw_revision=raw_revision,
                dataset_path=str(dataset), url=f'https://huggingface.co/datasets/{repo}/tree/{BRANCH}',
                remote_files_verified=len(after), complete=True)


def sync(repo=REPO, raw_dataset=None, raw_revision=None):
    if repo != REPO: raise ValueError('Training release configuration belongs to '+REPO)
    from .hub_upload import authenticated_repo, find_published_base
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with STATE.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        api, repo = authenticated_repo(repo)
        info = api.repo_info(repo_id=repo,repo_type='dataset')
        if raw_revision != info.sha or raw_dataset is None:
            raw_revision = info.sha; raw_dataset = find_published_base(api,repo,info)
        previous = json.loads(STATE.read_text()) if STATE.exists() else {}
        baseline = Path(previous.get('dataset_path', str(BASELINE)))
        if not (baseline/'meta/quality_policy.json').exists():
            from huggingface_hub import snapshot_download
            training_head = api.repo_info(repo_id=repo,repo_type='dataset',revision=BRANCH).sha
            baseline = Path(snapshot_download(repo,repo_type='dataset',revision=training_head,
                local_dir=str(STATE.parent/f'baseline-{training_head[:12]}'),
                allow_patterns=['meta/*','data/*','quality/*','training_release.json']))
        output = STATE.parent/f'KRVLA-{raw_revision[:12]}'
        report = prepare(raw_dataset,baseline,output,raw_revision)
        print(f'Processed training release: {report["episodes"]} episodes; {report["new_episodes"]} new; {report["split_counts"]}', flush=True)
        result = publish(api,repo,output,raw_revision)
        json_write(STATE,result); json_write(output.parent/f'{output.name}-upload.json',result)
        print('Training-ready upload: '+result['url']+' @ '+result['commit'], flush=True)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default=REPO)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--raw-dataset',type=Path)
    parser.add_argument('--raw-revision')
    parser.add_argument('--baseline',type=Path,default=BASELINE)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args(argv)
    if args.prepare_only:
        if not all((args.raw_dataset,args.raw_revision,args.output)):
            parser.error('--prepare-only requires --raw-dataset, --raw-revision and --output')
        report = prepare(args.raw_dataset,args.baseline,args.output,args.raw_revision)
        print(json.dumps({k:report[k] for k in ('episodes','new_episodes','frames','totals','new_episode_totals','split_counts')},indent=2))
    else:
        sync(args.repo,args.raw_dataset,args.raw_revision)


if __name__ == '__main__': main()
