"""Apply scene/model/noise grouping at the current completed-block boundary."""
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import time
import psutil

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO/'scratch/dom45/evaluations/velocity-uncertainty-cr200-20261116'


def main():
    state = json.loads((ROOT/'execution/status.json').read_text())
    job = state['job']
    folder = Path(job['folder'])
    parent = psutil.Process(int((ROOT/'runner.pid').read_text()))
    assert 'run_velocity_uncertainty_cr.py' in ' '.join(parent.cmdline())
    clients = [p for p in parent.children(recursive=True) if 'dom_krv_scene_client.py' in ' '.join(p.cmdline())]
    assert len(clients) == 1
    print('Waiting for completed block:', job['id'], flush=True)
    receipt = folder/'resident_client_completion.json'
    while not receipt.exists():
        assert parent.is_running() and parent.status() != psutil.STATUS_ZOMBIE
        time.sleep(.05)
    done = json.loads(receipt.read_text())
    assert done['complete'] and done['trials'] == job['repeats']
    clients[0].send_signal(signal.SIGTERM)
    parent.send_signal(signal.SIGTERM)
    for _ in range(1200):
        if not parent.is_running() or parent.status() == psutil.STATUS_ZOMBIE:
            break
        time.sleep(.1)
    else:
        raise TimeoutError('Evaluator did not stop for scheduling handoff')
    archive = ROOT/'attempts'/('schedule-regroup-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    archive.mkdir(parents=True)
    for name in ['execution', 'supervisor.log', 'status.json']:
        source = ROOT/name
        if source.exists():source.rename(archive/name)
    with (ROOT/'supervisor.log').open('a') as log:
        process = subprocess.Popen(['/bin/bash', '-c',
            'source workstation/env.local.sh\nexec python -u workstation/run_velocity_uncertainty_cr.py --after-vu200 --resume-interrupted'],
            cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (ROOT/'runner.pid').write_text(str(process.pid)+'\n')
    (ROOT/'schedule-regroup-handoff.json').write_text(json.dumps(dict(
        applied_utc=datetime.now(timezone.utc).isoformat(), boundary_job=job['id'],
        supervisor_pid=process.pid, archive=str(archive), all_completed_trials_retained=True), indent=2)+'\n')
    print('Grouped continuation launched:', process.pid, flush=True)


if __name__ == '__main__':
    main()
