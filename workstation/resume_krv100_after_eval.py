#!/usr/bin/env python3
"""Resume the same run after full evaluation or an authorized LS stop decision."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'evaluation/official_dom/scripts'))
from report import read_records, verify_records


def verify_folder(folder, root, listing, repeats):
    """Use frozen trial checks without rewriting the uploader's receipt file."""
    protocol = read(root / 'protocol.json')
    tests = {row['name']: row for row in protocol['tests']}
    names = [Path(line).stem for line in listing.read_text().splitlines() if line.strip()]
    result = verify_records(read_records(folder), names, repeats,
                            {name: tests[name]['horizon_seconds'] for name in names})
    log = (folder / 'client.log').read_text()
    if (f'Evaluation completed. Total tests run: {len(names) * repeats}' not in log
            or 'Test suite done with success rates:' not in log):
        raise ValueError('Client did not acknowledge the completed dimension')
    return result


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def atomic(path, value):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def service_idle(name):
    result = subprocess.run(['systemctl', '--user', 'show', name, '-p', 'MainPID',
                             '-p', 'ActiveState', '-p', 'SubState'], capture_output=True, text=True)
    fields = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if not fields:
        raise RuntimeError('Cannot inspect dependency service: ' + name)
    return fields.get('MainPID', '0') == '0' and fields.get('ActiveState') not in ('active', 'activating', 'deactivating')


def validate(request):
    if (not request.get('user_authorized') or request['variant'] != 'krv100'
            or request['wandb_id'] != '2qh0ijb1' or request['resume_from_epoch'] != 1
            or request['stop_after_completed_epochs'] != 100):
        raise ValueError('Authorized continuation of the existing krv100 run is required')
    for name, expected in request['files_sha256'].items():
        if sha(Path(name)) != expected:
            raise ValueError('Queued continuation dependency changed: ' + name)
    rule = request.get('evaluation_stop_rule')
    if rule and (not rule.get('user_authorized') or rule.get('after_dimension') != 'LS'
                 or rule.get('comparison') != 'less_than_or_equal'
                 or not 0 <= rule.get('success_threshold_percent', -1) <= 100):
        raise ValueError('Invalid user-authorized LS decision rule')


def maybe_stop_after_ls(request):
    rule = request.get('evaluation_stop_rule')
    if not rule:
        return
    root = Path(request['evaluation_results'])
    status = read(root / 'suite_status.json')
    if status.get('state') in ('complete', 'stopped_after_LS'):
        return
    folder = root / 'full/02_LS'
    if not (folder / 'verification.json').is_file():
        return
    result = read(folder / 'verification.json')
    if not result.get('verified'):
        return
    if result['trials'] != 200 or result['success_percent'] > rule['success_threshold_percent']:
        return
    # The evaluator may already be initializing the next dimension. Stop its
    # service and all GPU workers before publishing the intentional stop.
    subprocess.run(['systemctl', '--user', 'stop', request['evaluation_service']],
                   check=True, timeout=180)
    if not service_idle(request['evaluation_service']):
        raise RuntimeError('Evaluator still owns GPU workers after the LS stop')
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    archive = root / 'archives' / ('after-LS-user-stop-' + now.replace(':', '').replace('+', '_'))
    archive.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / 'suite_status.json', archive / 'interrupted_suite_status.json')
    # Preserve any speculative next-dimension startup separately. Those partial
    # infrastructure attempts are not included in the retained comparison.
    for later in sorted((root / 'full').iterdir()):
        if later.name[:2].isdigit() and int(later.name[:2]) > 2:
            shutil.move(str(later), str(archive / later.name))
    verified, rows = {}, []
    for i, dimension in enumerate(('CR', 'DA', 'LS')):
        part = root / 'full' / f'{i:02d}_{dimension}'
        verified[dimension] = verify_folder(part, root, root / 'test_lists' / f'{i:02d}_{dimension}.txt', 20)
        rows.extend(read_records(part))
    partial = {'complete': False, 'intentional_stop': True, 'verified_trials': len(rows),
               'expected_full_suite_trials': 1800, 'completed_dimensions': verified,
               'checkpoint': request['evaluated_checkpoint'], 'manifest': request['manifest'],
               'fingerprint': request['experiment_fingerprint'], 'stop_rule': rule,
               'evaluated_subset_success_percent': 100 * statistics.mean(r['success'] for r in rows),
               'evaluated_subset_mean_time_s': statistics.mean(r['task_time_s'] for r in rows),
               'note': 'User stopped after LS based on interim performance; this is a partial evaluation, not a full Table 1 score.'}
    atomic(root / 'partial_summary.json', partial)
    lines = ['# Epoch-one evaluation stopped after LS', '',
             'Partial evaluation: 600 verified trials in CR, DA and LS; the remaining six dimensions were skipped.', '',
             '| Dimension | Success (%) | Trials | Mean time (s) |', '|---|---:|---:|---:|']
    for dimension, values in verified.items():
        lines.append(f"| {dimension} | {values['success_percent']:.2f} | {values['trials']} | {values['mean_task_time_s']:.2f} |")
    lines += ['', 'No full-suite overall Success or Time is available. The stop decision used the observed LS result.', '']
    (root / 'partial_table1_comparison.md').write_text('\n'.join(lines))
    atomic(root / 'suite_status.json', {'state': 'stopped_after_LS', 'updated_at_utc': now,
           'expected_trials': 1800, 'verified_trials': 600, 'intentional_stop': True,
           'complete': False, 'exit_code': 0, 'checkpoint': request['evaluated_checkpoint'],
           'stop_rule': rule, 'results': str(root), 'retained_dimensions': ['CR', 'DA', 'LS']})
    print(json.dumps({'phase': 'stopped_after_LS', 'LS_success_percent': result['success_percent'],
                      'cutoff_percent': rule['success_threshold_percent'], 'verified_trials': 600}), flush=True)


def ready(request):
    root = Path(request['evaluation_results'])
    status, summary = read(root / 'suite_status.json'), read(root / 'summary.json')
    partial_stop = status.get('state') == 'stopped_after_LS' and status.get('intentional_stop')
    if partial_stop:
        partial = read(root / 'partial_summary.json')
        rule = request.get('evaluation_stop_rule')
        if (not rule or partial.get('stop_rule') != rule or partial.get('verified_trials') != 600
                or partial['completed_dimensions']['LS']['success_percent'] > rule['success_threshold_percent']):
            raise ValueError('Partial evaluation does not satisfy the authorized LS stop rule')
    elif status.get('state') != 'complete' or not summary.get('complete'):
        rule = request.get('evaluation_stop_rule')
        ls = read(root / 'full/02_LS/verification.json')
        if rule and not ls:
            return 'waiting_for_200_trial_LS_decision'
        return 'waiting_for_full_1800_trial_evaluation'
    expected = 600 if partial_stop else 1800
    if (status.get('verified_trials') != expected
            or (not partial_stop and summary.get('verified_trials') != expected)
            or status.get('exit_code') != 0 or status.get('checkpoint') != request['evaluated_checkpoint']):
        raise ValueError('Evaluation is not the successful, verified epoch-one full suite')
    if not service_idle(request['evaluation_service']):
        return 'waiting_for_evaluator_gpu_cleanup'
    receipt = read(Path(request['training_root']) / 'wandb-evaluation/upload_status.json')
    report = read(Path(request['training_root']) / 'wandb-evaluation/report.json')
    uploaded = receipt.get('evaluation_finished') if partial_stop else receipt.get('full_suite_complete')
    if (not uploaded or receipt.get('evaluation_state') != status['state']
            or receipt.get('verified_closed_loop_trials') != expected
            or report.get('checkpoint') != request['evaluated_checkpoint']
            or not service_idle(request['wandb_upload_service'])):
        return 'waiting_for_final_wandb_upload'
    return None


def verify_complete_evaluation(request):
    root = Path(request['evaluation_results'])
    protocol = read(root / 'protocol.json')
    if (protocol['trained_checkpoint'] != request['evaluated_checkpoint']
            or protocol['training_manifest'] != request['manifest']):
        raise ValueError('Evaluation protocol belongs to a different training checkpoint')
    if sha(Path(request['evaluated_checkpoint']) / 'model.safetensors') != protocol['checkpoint_weights_sha256']:
        raise ValueError('Evaluated checkpoint weights changed')
    rows = []
    partial_stop = read(root / 'suite_status.json').get('state') == 'stopped_after_LS'
    dimensions = protocol['dimension_order'][:3] if partial_stop else protocol['dimension_order']
    for i, dimension in enumerate(dimensions):
        folder = root / 'full' / f'{i:02d}_{dimension}'
        verified = verify_folder(folder, root, root / 'test_lists' / f'{i:02d}_{dimension}.txt', 20)
        if verified['trials'] != 200 or verified['infrastructure_failures']:
            raise ValueError('A benchmark dimension is incomplete')
        rows.extend(read_records(folder))
    summary = read(root / ('partial_summary.json' if partial_stop else 'summary.json'))
    success = summary['evaluated_subset_success_percent'] if partial_stop else summary['observed']['Success']
    duration = summary['evaluated_subset_mean_time_s'] if partial_stop else summary['observed']['Time']
    if (len(rows) != (600 if partial_stop else 1800)
            or abs(100 * statistics.mean(r['success'] for r in rows) - success) > 1e-10
            or abs(statistics.mean(r['task_time_s'] for r in rows) - duration) > 1e-10):
        raise ValueError('Summary differs from the verified full-suite trial records')


def activate(request, archive):
    """Archive epoch-one evidence and extend only its external budget control."""
    root = Path(request['training_root'])
    run = Path(request['family_root']) / 'runs/krv100'
    archive.mkdir(parents=True, exist_ok=True)
    plan_path = archive / 'activation_plan.json'
    plan = read(plan_path)
    control_path, original_request_path = root / 'training_control.json', root / 'request.json'
    if not plan:
        latest = read(run / 'latest.json')
        if (latest['generation'] != Path(request['evaluated_checkpoint']).name
                or latest['fingerprint'] != request['experiment_fingerprint']):
            raise ValueError('Training latest pointer changed before the continuation')
        for source in (control_path, original_request_path, run / 'training_complete.json',
                       run / 'latest.json', run / 'final.json', run / 'epoch-001.json'):
            shutil.copy2(source, archive / source.name)
        old_control, original = read(control_path), read(original_request_path)
        if (sha(control_path) != request['previous_training_control_sha256']
                or sha(original_request_path) != request['previous_training_request_sha256']):
            raise ValueError('Training authorization changed since this continuation was queued')
        control = {**old_control, 'authorized_at_utc': request['authorized_at_utc'],
                   'stop_after_completed_epochs': request['stop_after_completed_epochs'],
                   'previous_control_sha256': request['previous_training_control_sha256'],
                   'dependency_evaluation': request['evaluation_results'],
                   'evaluation_stop_rule': request.get('evaluation_stop_rule'),
                   'reason': 'User requested same-run continuation after full evaluation or the authorized LS stop decision; existing recipe and early stopping retained'}
        control_target = archive / 'continued_training_control.json'
        atomic(control_target, control)
        original['epoch_limit']['completed_epochs'] = control['stop_after_completed_epochs']
        original['epoch_limit']['files_sha256']['training_control.json'] = sha(control_target)
        original['epoch_limit']['files_sha256']['epoch_training_control.py'] = sha(REPO / 'workstation/epoch_training_control.py')
        original.setdefault('post_evaluation_continuations', []).append({
            'authorized_at_utc': request['authorized_at_utc'], 'evaluation': request['evaluation_results'],
            'wandb_id': request['wandb_id'], 'resume_from_epoch': 1,
            'stop_after_completed_epochs': control['stop_after_completed_epochs'],
            'preserve_optimizer_scheduler_rng_cursor': True, 'epoch_one_evidence': str(archive)})
        request_target = archive / 'continued_training_request.json'
        atomic(request_target, original)
        plan = {'control_sha256': sha(control_target), 'request_sha256': sha(request_target),
                'previous_control_sha256': request['previous_training_control_sha256'],
                'previous_request_sha256': request['previous_training_request_sha256']}
        atomic(plan_path, plan)
    for destination, source, old_key, new_key in (
            (control_path, archive / 'continued_training_control.json', 'previous_control_sha256', 'control_sha256'),
            (original_request_path, archive / 'continued_training_request.json', 'previous_request_sha256', 'request_sha256')):
        if sha(destination) not in (plan[old_key], plan[new_key]) or sha(source) != plan[new_key]:
            raise ValueError('Continuation activation differs from its archived plan')
        atomic(destination, read(source))
    completion = read(run / 'training_complete.json')
    if completion.get('completion_reason') == 'user_epoch_limit' and completion.get('epoch') == 1:
        if sha(run / 'training_complete.json') != request['epoch_one_completion_sha256']:
            raise ValueError('Epoch-one completion record changed')
        (run / 'training_complete.json').unlink()
    # The epoch-001 pointer remains in the live run and protects its generation
    # from checkpoint pruning. Final evaluation links are also preserved.
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request = read(args.request)
    root = Path(request['training_root'])
    status_path = root / 'post_eval_resume_status.json'
    lock = (root / '.post-eval-resume.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(phase, **extra):
        value = {'phase': phase, 'updated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'variant': 'krv100', 'wandb_id': request['wandb_id'],
                 'resume_from_epoch': 1, 'maximum_completed_epochs': 100,
                 'evaluation_stop_rule': request.get('evaluation_stop_rule'), **extra}
        atomic(status_path, value); print(json.dumps(value), flush=True)
    if read(status_path).get('phase') == 'complete':
        return
    try:
        previous = None
        while True:
            validate(request)
            maybe_stop_after_ls(request)
            waiting = ready(request)
            if waiting is None:
                break
            if waiting != previous:
                status(waiting); previous = waiting
            time.sleep(request.get('poll_seconds', 10))
        verify_complete_evaluation(request)
        status('activating_same_run_continuation')
        plan = activate(request, Path(request['epoch_one_archive']))
        status('launching_training', activation=plan)
        command = ['/bin/bash', str(REPO / 'workstation/run-krv100-after-eval.sh')]
        with (root / 'continuation-coordinator.log').open('a') as log:
            process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            def interrupted(signum, frame):
                process.send_signal(signal.SIGTERM)
                raise KeyboardInterrupt('Continuation controller interrupted')
            signal.signal(signal.SIGTERM, interrupted)
            status('training', coordinator_pid=process.pid, wandb_run=request['wandb_run'])
            code = process.wait()
        if code:
            raise RuntimeError(f'Training coordinator exited with {code}; inspect continuation-coordinator.log and training.log')
        completed = read(Path(request['family_root']) / 'runs/krv100/training_complete.json')
        if completed.get('epoch', 0) <= 1 or completed.get('fingerprint') != request['experiment_fingerprint']:
            raise ValueError('Continuation exited without completing additional training')
        status('complete', completed_epochs=completed['epoch'], updates=completed['updates'])
    except BaseException as error:
        status('failed', error=str(error)); raise


if __name__ == '__main__':
    main()
