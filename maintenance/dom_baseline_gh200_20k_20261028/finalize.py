"""Finish this launch's active staging, freeze inputs, and release its held job."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def persist(value):
    path = ROOT / 'finalization.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)

def staging_alive():
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            if p.stat().st_uid != os.getuid():
                continue
            command = (p / 'cmdline').read_bytes().split(b'\0')
            if b'prefetch.py' in command and (p / 'cwd').resolve() == ROOT:
                return True
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    assert args.job_id.isdigit()
    lock = (ROOT / 'finalization.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'job_id': args.job_id, 'released': False, 'stage': 'waiting_for_verified_data'}
    start = time.monotonic()
    persist(state)
    try:
        while not (ROOT / 'verified-data.json').exists():
            assert staging_alive(), 'Dataset staging exited before writing its complete verification receipt'
            assert time.monotonic() - start < 3 * 3600, 'Dataset staging exceeded the three-hour launch bound'
            progress = ROOT / 'prefetch-progress.json'
            if progress.exists():
                try:
                    state['data_progress'] = json.loads(progress.read_text())
                except json.JSONDecodeError:
                    pass
            persist(state)
            time.sleep(10)
        receipt = json.loads((ROOT / 'verified-data.json').read_text())
        assert receipt['complete'] and receipt['state'] == 'complete' and not receipt['errors']
        assert receipt['completed_files'] == receipt['total_files'] == 66045
        state.update(stage='freezing', verification=receipt)
        persist(state)
        subprocess.run([sys.executable, str(ROOT / 'freeze.py')], check=True)
        info = subprocess.check_output(['scontrol', 'show', 'job', args.job_id, '-o'], text=True)
        assert 'JobName=dom100-20k-2n ' in info
        assert 'UserId=anayak2(' in info and 'Account=bicn-dtai-gh ' in info
        assert 'JobState=PENDING ' in info and 'Reason=JobHeldUser ' in info, info
        subprocess.run(['scontrol', 'release', args.job_id], check=True)
        queue = subprocess.check_output(['squeue', '-h', '-j', args.job_id,
            '-o', '%i|%u|%a|%j|%T|%R'], text=True).strip()
        state.update(stage='released_to_scheduler', released=True, queue=queue,
                     elapsed_seconds=time.monotonic()-start)
        persist(state)
        print(json.dumps(state), flush=True)
    except Exception as exc:
        state.update(stage='preparation_failed_job_remains_held', error=str(exc),
                     elapsed_seconds=time.monotonic()-start)
        persist(state)
        raise

if __name__ == '__main__':
    main()
