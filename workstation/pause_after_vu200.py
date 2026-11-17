"""Pause the local two-model evaluation at the completed VU block boundary."""
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]/'scratch/dom45/evaluations/dom-krv-epoch4-1800-20261116'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix('.pause-tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n');temp.replace(path)


def main():
    supervisor = psutil.Process(int((ROOT/'scene-first/runner.pid').read_text()))
    assert 'extend_dom_krv_scene_first.py' in ' '.join(supervisor.cmdline())
    clients = [p for p in supervisor.children(recursive=True) if 'dom_krv_scene_client.py' in ' '.join(p.cmdline())]
    assert len(clients) == 1
    client = clients[0]
    plans = {phase: read(ROOT/phase/'plan.json') for phase in ['krv_epoch4', 'dom']}
    vu = {phase: [j for j in plan['jobs'] if j['dimension'] == 'VU'] for phase, plan in plans.items()}
    assert all(len(jobs) == 10 for jobs in vu.values())
    write(ROOT/'vu200-pause-request.json', dict(requested_utc=datetime.now(timezone.utc).isoformat(),
        target='200 VU trials each, including 100 retained trials', supervisor_pid=supervisor.pid,
        client_pid=client.pid, preserve_all_trials=True, next_eval='awaiting user clarification: noise sweep versus VI-on tests'))
    while True:
        ready = True
        for jobs in vu.values():
            for job in jobs:
                receipt = Path(job['folder'])/'resident_client_completion.json'
                if not receipt.exists():
                    ready = False;break
                done = read(receipt)
                assert done['complete'] and done['trials'] == 10
        if ready:
            break
        assert supervisor.is_running() and supervisor.status() != psutil.STATUS_ZOMBIE, 'Evaluation stopped before VU completed'
        time.sleep(.05)
    # Request client stop first; parent then tears down simulator/worker groups.
    client.send_signal(signal.SIGTERM)
    supervisor.send_signal(signal.SIGTERM)
    for _ in range(600):
        if not supervisor.is_running() or supervisor.status() == psutil.STATUS_ZOMBIE:
            break
        time.sleep(.1)
    else:
        raise TimeoutError('Evaluator did not release its resources after pause request')
    pending = []
    for phase, plan in plans.items():
        for job in plan['jobs']:
            folder = Path(job['folder'])
            if (folder/'resident_client_completion.json').exists():
                continue
            file = folder/'trial_results.jsonl'
            pending.append(dict(phase=phase, job=job,
                partial_trials=len(file.read_text().splitlines()) if file.exists() else 0))
    state = dict(verified=True, paused=True, paused_utc=datetime.now(timezone.utc).isoformat(),
                 boundary='VU 200 trials per model; first 100 retained plus 100 extension',
                 vu_trials_each=200, remaining_jobs=len(pending), pending=pending,
                 full_1800_each_complete=False)
    write(ROOT/'vu200-pause.json', state)
    write(ROOT/'status.json', dict(phase='scene-first', stage='paused_after_vu200',
        updated_utc=state['paused_utc'], vu_trials_each=200, remaining_jobs=len(pending),
        resume_manifest=str(ROOT/'vu200-pause.json')))
    print(json.dumps({'paused': True, 'vu_trials_each': 200, 'remaining_jobs': len(pending)}), flush=True)


if __name__ == '__main__':
    main()
