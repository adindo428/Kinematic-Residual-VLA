"""Run the matched 1 m/s initial-speed stress test after the baseline pilot."""
import argparse
from copy import deepcopy
import fcntl
import math
from pathlib import Path
import signal

import run_closed_loop_pilot as runner

PARENT = runner.ROOT
ROOT = PARENT.parent/'closed-loop-speed-ood-20261112'
runner.ROOT = ROOT
low = runner.low


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'plan.json').exists(): return low.read(ROOT/'plan.json')
    evidence = low.read(ROOT/'speed_evidence.json')
    speed = evidence['selected_initial_speed_mps']
    assert speed > evidence['configured_collection_initial_speed_range_mps'][1]
    base = low.read(PARENT/'plan.json'); plan = deepcopy(base)
    (ROOT/'configs').mkdir(exist_ok=True)
    mapping = {}; prompts = {}
    for test in plan['tests']:
        oldpath = Path(test['path']); cfg = low.read(oldpath)
        assert low.sha(oldpath) == test['config_sha256']
        before = deepcopy(cfg)
        velocity = cfg['scene']['object']['init_state']['lin_vel']
        norm = math.sqrt(sum(v*v for v in velocity))
        cfg['scene']['object']['init_state']['lin_vel'] = [v*speed/norm for v in velocity]
        path = ROOT/'configs'/oldpath.name; low.write(path, cfg)
        check = deepcopy(cfg); check['scene']['object']['init_state']['lin_vel'] = before['scene']['object']['init_state']['lin_vel']
        assert check == before, 'Only initial velocity may change'
        mapping[str(oldpath)] = str(path)
        prompts[str(path)] = base['saved_prompts'][str(oldpath)]
        test.update(path=str(path), config_sha256=low.sha(path), initial_speed_mps=speed,
            matched_baseline_path=str(oldpath), matched_baseline_sha256=low.sha(oldpath))
    tests = {t['path']: t for t in plan['tests']}
    for job in plan['jobs']:
        job['scene'] = mapping[job['scene']]
        job['scene_sha256'] = tests[job['scene']]['config_sha256']
        folder = ROOT/'blocks'/job['id']; folder.mkdir(parents=True, exist_ok=True)
        job['folder'] = str(folder); (folder/'case.txt').write_text(job['scene']+'\n')
    plan['saved_prompts'] = prompts
    plan['created_utc'] = runner.stamp()
    plan['protocol'].update(initial_speed_mps=speed, speed_sweep=False,
        benchmark='Matched high initial-speed closed-loop pilot',
        baseline=str(PARENT), change='Only initial object linear velocity magnitude; direction and every other config field identical',
        ood_scope=evidence['scope'], evidence_sha256=low.sha(ROOT/'speed_evidence.json'))
    plan['files_sha256'][str(Path(__file__).resolve())] = low.sha(__file__)
    low.write(ROOT/'plan.json', plan)
    low.write(ROOT/'protocol.json', {**plan['protocol'], 'tests': plan['tests']})
    low.write(ROOT/'request.json', dict(user_authorized=True, dependency=str(PARENT/'completion.json'),
        baseline_speed_mps=.1, next_speed_mps=speed, total_episodes=75,
        require_verified_baseline=True, no_automatic_resume_of_pi05=True))
    runner.status('queued_waiting_for_baseline_completion', total=75)
    return plan


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare-only', action='store_true'); args = parser.parse_args()
    plan = prepare()
    if args.prepare_only: return
    dependency = Path(low.read(ROOT/'request.json')['dependency']).parent
    signal.signal(signal.SIGTERM, lambda *_: runner.STOP.set())
    signal.signal(signal.SIGINT, lambda *_: runner.STOP.set())
    with (ROOT/'runner.lock').open('a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not runner.STOP.is_set():
            completion = dependency/'completion.json'
            if completion.exists():
                result = low.read(completion)
                assert result['verified'] and result['complete'] and set(result['trials'].values()) == {25}
                if dependency == PARENT or (dependency/'baseline_pairing.json').exists():
                    if dependency != PARENT: assert low.read(dependency/'baseline_pairing.json')['verified']
                    break
            parent = low.read(dependency/'status.json')
            if parent['stage'] == 'failed':
                runner.status('blocked_baseline_failed', parent_status=parent); return
            runner.status('queued_waiting_for_causal_gate' if dependency != PARENT else 'queued_waiting_for_baseline_completion', dependency=str(dependency), dependency_completed=parent.get('completed', 0), total=75)
            runner.STOP.wait(10)
        if runner.STOP.is_set(): return
        with (low.ROOT/'queue.lock').open('a') as lock:
            while not runner.STOP.is_set():
                try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); break
                except BlockingIOError: runner.STOP.wait(5)
            if runner.STOP.is_set(): return
            try:
                runner.status('starting', total=75)
                runner.launch(plan)
                # The shared runner's final summary has a baseline-specific
                # sentence; update it for this explicitly frozen speed.
                path = ROOT/'comparison.md'
                path.write_text(path.read_text().replace('initial speed 0.10 m/s', 'initial speed 1.00 m/s')+
                    '\nOOD scope: outside the saved collection initial-speed range; not an exhaustive bound on training velocities.\n')
            except BaseException as exc:
                runner.status('failed', error=repr(exc)); raise


if __name__ == '__main__': main()
