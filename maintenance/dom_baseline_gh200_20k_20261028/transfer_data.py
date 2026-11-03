"""Relay only missing dataset files over the two existing SSH connections."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import time

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / 'evidence'
DEST = '/work/nvme/bicn/anayak2/dynamicvla/dom100-20k-gh200-20261028'
OLD_WORK = '/work/nvme/bivv/aparameshwaran/dynamicvla'
OLD_DATA = OLD_WORK + '/huggingface/lerobot/hzxie/DOM'
NEW_DATA = '/work/nvme/bicn/anayak2/dynamicvla/huggingface/lerobot/hzxie/DOM'
OLD = ['ssh', '-S', '/tmp/delta-krvla.sock', '-o', 'BatchMode=yes',
       'aparameshwaran@login.delta.ncsa.illinois.edu']
NEW = ['ssh', '-S', '/tmp/deltaai-kinematic.sock', '-o', 'BatchMode=yes',
       'anayak2@gh-login01.delta.ncsa.illinois.edu']

def persist(state):
    p = EVIDENCE / 'dataset-transfer.json'
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(state, indent=2) + '\n')
    t.replace(p)

def main():
    state = {'stage': 'reading_transfer_plan', 'complete': False, 'job_id': '3479152',
             'started_at': datetime.now(timezone.utc).isoformat(),
             'source': OLD_DATA, 'destination': NEW_DATA, 'hub_downloads': False}
    persist(state)
    plan = subprocess.check_output(NEW + ['cat ' + shlex.quote(DEST + '/transfer-files.nul')])
    assert plan and plan.endswith(b'\0')
    names = [p.decode() for p in plan.split(b'\0') if p]
    assert all(n.startswith(('data/', 'videos/', 'meta/')) and '..' not in Path(n).parts for n in names)
    (EVIDENCE / 'transfer-files.nul').write_bytes(plan)
    remote_list = OLD_WORK + '/export-dom100-20k-20261028/dataset-transfer-files.nul'
    subprocess.run(OLD + ['cat > ' + shlex.quote(remote_list)], input=plan, check=True)
    source_command = 'tar -C ' + shlex.quote(OLD_DATA) + ' --null --verbatim-files-from -T ' + shlex.quote(remote_list) + ' -cf -'
    dest_command = 'tar -C ' + shlex.quote(NEW_DATA) + ' -xf -'
    state.update(stage='transferring', files=len(names), bytes_relayed=0)
    persist(state)
    started = last = time.monotonic()
    with (EVIDENCE / 'dataset-source.stderr').open('wb') as source_err, (EVIDENCE / 'dataset-destination.stderr').open('wb') as dest_err:
        source = subprocess.Popen(OLD + [source_command], stdout=subprocess.PIPE, stderr=source_err)
        destination = subprocess.Popen(NEW + [dest_command], stdin=subprocess.PIPE, stderr=dest_err)
        try:
            while True:
                block = source.stdout.read(4 * 1024**2)
                if not block:
                    break
                destination.stdin.write(block)
                state['bytes_relayed'] += len(block)
                now = time.monotonic()
                if now - last >= 20:
                    state['elapsed_seconds'] = now - started
                    persist(state)
                    print(json.dumps({k: state[k] for k in ['stage', 'files', 'bytes_relayed', 'elapsed_seconds']}), flush=True)
                    last = now
            destination.stdin.close()
            source.stdout.close()
            state['source_exit'] = source.wait()
            state['destination_exit'] = destination.wait()
            assert state['source_exit'] == state['destination_exit'] == 0, state
        except BaseException:
            source.terminate()
            destination.terminate()
            state.update(stage='transfer_failed_job_remains_held', elapsed_seconds=time.monotonic()-started)
            persist(state)
            raise
    state.update(stage='verifying', elapsed_seconds=time.monotonic()-started)
    persist(state)
    python = DEST + '/environment/bin/python'
    with (EVIDENCE / 'dataset-verification.log').open('w') as output:
        subprocess.run(NEW + [shlex.quote(python) + ' -u ' + shlex.quote(DEST + '/prefetch.py')],
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    state.update(stage='freezing_and_releasing')
    persist(state)
    with (EVIDENCE / 'finalization.log').open('w') as output:
        subprocess.run(NEW + [shlex.quote(python) + ' -u ' + shlex.quote(DEST + '/finalize.py') + ' --job-id 3479152'],
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    state.update(stage='verified_and_released', complete=True,
                 completed_at=datetime.now(timezone.utc).isoformat())
    persist(state)
    print(json.dumps(state), flush=True)

if __name__ == '__main__':
    main()
