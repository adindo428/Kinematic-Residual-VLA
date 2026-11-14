#!/usr/bin/env python3
"""Attach epoch-one validation, verified DOM scores and example videos to W&B."""
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'evaluation/official_dom/scripts'))
from report import read_records, verify_folder


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def write(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def snapshot(request, maximum_videos):
    root = Path(request['results'])
    run = Path(request['family_root']) / 'runs/krv100'
    completed = read(run / 'training_complete.json')
    if (completed.get('epoch') != 1 or completed.get('pending_validation')
            or completed.get('fingerprint') != request['experiment_fingerprint']):
        raise ValueError('Completed, validated epoch-one training is required')
    checkpoint = Path(completed['checkpoint'])
    config = read(checkpoint / 'config.json')
    if config['wandb_id'] != '2qh0ijb1':
        raise ValueError('Refusing to publish to an unexpected training run')
    validation_path = run / 'validation-epoch-001.json'
    validation = read(validation_path)
    if not validation or validation['h20/action_mse'] != completed['best']['value']:
        raise ValueError('Held-out validation differs from the selected checkpoint')
    protocol = read(root / 'protocol.json')
    baseline = read(Path(request['baseline_results']) / 'summary.json')
    status = read(root / 'suite_status.json')
    verified = {}
    folders = []
    if protocol:
        if protocol['trained_checkpoint'] != str(checkpoint):
            raise ValueError('Evaluation belongs to another checkpoint')
        candidates = [('pilot', root / 'pilot', root / 'test_lists/pilot.txt', 1)]
        candidates += [(dimension, root / 'full' / f'{i:02d}_{dimension}',
                        root / 'test_lists' / f'{i:02d}_{dimension}.txt', 20)
                       for i, dimension in enumerate(protocol['dimension_order'])]
        for dimension, folder, listing, repeats in candidates:
            if not (folder / 'verification.json').is_file():
                continue
            result = verify_folder(folder, root, listing, repeats)
            verified[dimension] = result
            folders.append((dimension, folder, read_records(folder)))
    summary = read(root / 'summary.json')
    if summary and (not summary.get('complete') or summary.get('verified_trials') != 1800
                    or any(d not in verified for d in protocol['dimension_order'])):
        raise ValueError('Full Table 1 results must contain 1800 verified trials')
    partial = read(root / 'partial_summary.json')
    partial_stop = status.get('state') == 'stopped_after_LS' and status.get('intentional_stop')
    if partial_stop and (not partial.get('intentional_stop') or partial.get('verified_trials') != 600
                         or partial.get('checkpoint') != str(checkpoint)
                         or any(d not in verified for d in ('CR', 'DA', 'LS'))):
        raise ValueError('Intentional LS stop requires 600 verified trials from this checkpoint')
    # Match each finished episode to its rendered MP4 by configuration and trial
    # order, checking the outcome encoded in the upstream filename as well.
    episodes = []
    for dimension, folder, records in folders:
        by_environment = {}
        for row in records:
            by_environment.setdefault(row['env_name'], []).append(row)
        for environment, rows in sorted(by_environment.items()):
            videos = sorted((folder / 'videos').rglob(environment + '-*.mp4'))
            if len(videos) != len(rows):
                continue
            for row, video in zip(sorted(rows, key=lambda r: r['trial']), videos):
                outcome = 'SUCCESS' if row['success'] else 'FAIL'
                if not video.name.endswith('-' + outcome + '.mp4') or not video.stat().st_size:
                    continue
                episodes.append({'dimension': dimension, **row, 'video': str(video)})
    # First available success and failure, then different configurations. This
    # deterministic illustration rule is disclosed; these are not extra trials.
    selected = []
    for success in (True, False):
        match = next((r for r in episodes if r['success'] == success), None)
        if match and len(selected) < maximum_videos:
            selected.append(match)
    for row in episodes:
        if len(selected) >= maximum_videos:
            break
        if row['env_name'] not in {r['env_name'] for r in selected}:
            selected.append(row)
    result = {'completed': completed, 'validation': validation, 'verified': verified,
              'summary': summary, 'baseline': baseline, 'protocol': protocol,
              'status': status, 'episodes': selected, 'checkpoint': str(checkpoint),
              'partial_summary': partial if partial_stop else {},
              'evaluation_finished': bool(summary) or partial_stop}
    # Stable progress signature: do not republish merely because polling time or
    # verification-file timestamps changed.
    signature = hashlib.sha256(json.dumps({k: result[k] for k in
        ('validation', 'verified', 'summary', 'status', 'episodes', 'partial_summary')}, sort_keys=True).encode()).hexdigest()
    return result, signature


def publish(request, data, signature, state_path):
    import wandb
    root = Path(request['results'])
    run_root = Path(request['family_root']) / 'runs/krv100'
    upload_root = state_path.parent
    validation = data['validation']
    validation_table = wandb.Table(columns=['epoch', 'scope', 'metric', 'value'])
    for metric, value in sorted(validation.items()):
        scope, name = metric.rsplit('/', 1)
        validation_table.add_data(1, scope, name, value)
    logs = {'Validation/epoch_001_table': validation_table}
    logs.update({'Validation/epoch_001/' + k: v for k, v in validation.items()})
    observed = data['summary'].get('observed', {})
    table = wandb.Table(columns=['metric', 'unit', 'published_DOM', 'local_DOM_baseline',
                                'KRVLA_velocity_epoch1', 'verified_KRVLA_trials', 'status'])
    protocol = data['protocol'] or data['baseline']['protocol']
    for metric in [*protocol['dimension_order'], 'Success', 'Time']:
        dimension = data['verified'].get(metric, {})
        value = observed.get(metric, dimension.get('success_percent'))
        count = 1800 if metric in observed else dimension.get('trials', 0)
        cell_status = ('verified' if value is not None else
                       'not_evaluated_user_stop' if data['partial_summary'] else 'pending')
        table.add_data(metric, 'seconds' if metric == 'Time' else 'percent',
                       protocol['paper_reference'][metric], data['baseline']['observed'][metric],
                       value, count, cell_status)
    logs['ClosedLoop/Table1_comparison'] = table
    logs['ClosedLoop/verified_trials'] = sum(v['trials'] for k, v in data['verified'].items() if k != 'pilot')
    logs['ClosedLoop/complete'] = bool(data['summary'])
    logs['ClosedLoop/finished'] = data['evaluation_finished']
    for dimension, result in data['verified'].items():
        prefix = 'Pilot' if dimension == 'pilot' else 'ClosedLoop/' + dimension
        logs[prefix + '/success_percent'] = result['success_percent']
        logs[prefix + '/mean_task_time_s'] = result['mean_task_time_s']
        logs[prefix + '/trials'] = result['trials']
    episode_table = wandb.Table(columns=['epoch', 'dimension', 'configuration', 'trial',
                                         'success', 'task_time_s', 'rollout'])
    for row in data['episodes']:
        caption = (f"KRVLA + Velocity, epoch 1 | {row['dimension']} | {row['env_name']} | "
                   f"trial {row['trial']} | {'success' if row['success'] else 'failure'} | "
                   f"{row['task_time_s']:.2f} simulated seconds")
        video = wandb.Video(row['video'], format='mp4', caption=caption)
        episode_table.add_data(1, row['dimension'], row['env_name'], row['trial'],
                               row['success'], row['task_time_s'], video)
    if data['episodes']:
        logs['ClosedLoop/example_episodes'] = episode_table
    report = {
        'wandb_run': 'aparame-clemson-university/Kinematic-Residual-VLA/2qh0ijb1',
        'epoch': 1, 'checkpoint': data['checkpoint'], 'fingerprint': request['experiment_fingerprint'],
        'held_out_validation': validation, 'verified_closed_loop': data['verified'],
        'full_suite_complete': bool(data['summary']), 'evaluation_state': data['status'].get('state'),
        'evaluation_finished': data['evaluation_finished'], 'partial_summary': data['partial_summary'],
        'selected_episodes': data['episodes'],
        'selection_rule': 'First available success and failure, then different configurations; up to six videos. Only verified, finished folders are eligible.',
        'notes': ['Held-out action prediction metrics are separate from closed-loop manipulation success.',
                  'Pilot results are separate from the 1800-trial Table 1 suite.',
                  'KRVLA additionally receives current simulator object pose and velocity.',
                  'Video examples illustrate outcomes and are not an unbiased success estimate.'],
        'signature': signature,
    }
    write(upload_root / 'report.json', report)
    report_files = [upload_root / 'report.json', run_root / 'validation-epoch-001.json',
                    run_root / 'training_complete.json', root / 'protocol.json',
                    root / 'summary.json', root / 'table1_comparison.csv',
                    root / 'table1_comparison.md', root / 'table1_baseline_comparison.csv',
                    root / 'table1_baseline_comparison.md', root / 'partial_summary.json',
                    root / 'partial_table1_comparison.md']
    with wandb.init(id='2qh0ijb1', entity='aparame-clemson-university',
                    project='Kinematic-Residual-VLA', resume='must',
                    dir=str(upload_root), settings=wandb.Settings(init_timeout=120)) as run:
        if not run.resumed or run.id != '2qh0ijb1':
            raise RuntimeError('The existing training run must be resumed')
        run.log(logs)  # Append at the SDK's next history step; preserve training steps.
        run.summary['Evaluation/epoch_001_checkpoint'] = data['checkpoint']
        run.summary['Evaluation/epoch_001_state'] = data['status'].get('state', 'unknown')
        run.summary['Evaluation/epoch_001_complete'] = bool(data['summary'])
        run.summary['Evaluation/epoch_001_finished'] = data['evaluation_finished']
        run.summary['Evaluation/example_video_count'] = len(data['episodes'])
        artifact = wandb.Artifact('krv100-epoch001-validation', type='evaluation',
                                  metadata={'epoch': 1, 'fingerprint': request['experiment_fingerprint'],
                                            'full_suite_complete': bool(data['summary']), 'signature': signature})
        for path in report_files:
            if path.is_file():
                artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact).wait()
        url = run.url
    # Publish receipt only after W&B finished syncing all queued files/media.
    receipt = {'uploaded_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'signature': signature, 'run_url': url,
               'validation_metrics': len(validation), 'example_videos': len(data['episodes']),
               'verified_closed_loop_trials': logs['ClosedLoop/verified_trials'],
               'full_suite_complete': bool(data['summary']),
               'evaluation_finished': data['evaluation_finished'],
               'evaluation_state': data['status'].get('state')}
    write(state_path, receipt)
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--max-videos', type=int, default=6)
    args = parser.parse_args()
    if not 0 <= args.max_videos <= 6:
        parser.error('--max-videos must be between zero and six')
    request = read(args.request)
    if not request.get('user_authorized'):
        raise ValueError('Authorized epoch-one evaluation request is required')
    upload_root = Path(request['family_root']).parent / 'wandb-evaluation'
    upload_root.mkdir(exist_ok=True)
    state_path = upload_root / 'upload_status.json'
    lock = (upload_root / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        try:
            data, signature = snapshot(request, args.max_videos)
            if read(state_path).get('signature') != signature:
                publish(request, data, signature, state_path)
            if not args.watch or data['evaluation_finished']:
                return
        except Exception as error:
            print(json.dumps({'upload_error': str(error), 'retrying': args.watch}), flush=True)
            if not args.watch:
                raise
        time.sleep(30)


if __name__ == '__main__':
    main()
