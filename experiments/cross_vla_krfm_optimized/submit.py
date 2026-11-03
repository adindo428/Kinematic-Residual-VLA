"""Submit only this isolated A100 campaign; persist every scheduler receipt."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'submissions.json'


def atomic(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def main():
    with (ROOT / '.submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(STATE.read_text())
        if state.get('pending_submission'):
            raise RuntimeError('Uncertain previous submission: reconcile scheduler records before retrying')
        from common import sha
        prepared = json.loads((ROOT / 'prepared.json').read_text())
        if not prepared.get('complete'):
            raise RuntimeError('Frozen preparation is incomplete')
        for key, filename in [('statistics_sha256', 'statistics.json'), ('split_sha256', 'split.json'),
                              ('checkpoints_sha256', 'checkpoints.json'),
                              ('implementation_sha256', 'implementation-lock.json')]:
            if sha(ROOT / filename) != prepared[key]:
                raise RuntimeError('Frozen preparation changed: ' + filename)
        for recipe_path in sorted((ROOT / 'recipes').glob('*.json')):
            name = recipe_path.stem
            # Setup is already verified complete and older than Slurm's live
            # dependency retention. Profiles validate its frozen hashes again.
            dependency = state.get('profile_dependency')
            for command in ('profile', 'train'):
                key = name + '_' + command
                if key in state:
                    dependency = state[key]['job_id']
                    continue
                script = ROOT / (key + '.sbatch')
                job_name = 'a100opt_' + name + '-' + command[:4]
                matching = []
                jobs = subprocess.check_output(
                    ['squeue', '-u', 'aparame', '-h', '-n', job_name, '-o', '%i'], text=True).split()
                for job in jobs:
                    details = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
                    if 'Command=' + str(script) in details:
                        matching.append(job)
                if matching:
                    raise RuntimeError('Unrecorded matching job exists: reconcile instead of submitting: ' + str(matching))
                state['pending_submission'] = {
                    'key': key, 'script': str(script), 'dependency': dependency,
                    'attempted_at': datetime.now(timezone.utc).isoformat()}
                atomic(STATE, state)
                result = subprocess.run(
                    ['sbatch', '--parsable'] + (['--dependency=afterok:' + dependency] if dependency else []) + [str(script)],
                    text=True, capture_output=True, timeout=45)
                if result.returncode:
                    state.setdefault('submission_errors', []).append({
                        **state.pop('pending_submission'), 'stderr': result.stderr, 'returncode': result.returncode})
                    atomic(STATE, state)
                    raise RuntimeError(result.stderr)
                job = result.stdout.strip().split(';')[0]
                if not job.isdigit():
                    raise RuntimeError('Uncertain scheduler response: reconcile pending_submission before retrying')
                state[key] = {
                    'job_id': job, 'recipe': name, 'command': command, 'dependency': dependency,
                    'script': str(script), 'gpus': 2, 'gpu': 'A100 80 GB',
                    'submitted_at': datetime.now(timezone.utc).isoformat()}
                state.pop('pending_submission')
                atomic(STATE, state)
                # Save discoverable production metadata immediately, not at the end of the DAG.
                if command == 'train':
                    recipe = json.loads(recipe_path.read_text())
                    from common import fingerprint
                    candidates = ROOT / 'submissions' / 'hardware_candidates.json'
                    records = json.loads(candidates.read_text()) if candidates.exists() else {}
                    records[name] = {
                        **state[key], 'world_size': 2, 'variant': name, 'name': name + '-a100-opt',
                        'method': recipe['method'] + (' + KRFM' if recipe['kr'] else ' baseline'),
                        'lr': recipe['lr'], 'target_epochs': None, 'target_updates': recipe['steps'],
                        'recipe_fingerprint': fingerprint(recipe)}
                    atomic(candidates, records)
                dependency = job
                print(key, job, flush=True)


if __name__ == '__main__':
    main()
