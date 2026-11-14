"""Evaluate inference-only causal gating on the pilot's exact KRV weights."""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import signal

import run_closed_loop_pilot as runner

PARENT = runner.ROOT
ROOT = PARENT.parent/'closed-loop-causal-gate-20261112'
runner.ROOT = ROOT
low = runner.low
_status = runner.status

def status(stage, **kw):
    kw['total'] = 25
    if stage == 'complete': kw['completed'] = 25
    _status(stage, **kw)

runner.status = status

def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'plan.json').exists(): return low.read(ROOT/'plan.json')
    base = low.read(PARENT/'plan.json')
    model = deepcopy(next(m for m in base['models'] if m['key'] == 'krv_best_constant'))
    parent = Path(model['checkpoint']); checkpoint = ROOT/'checkpoint'; checkpoint.mkdir()
    for path in parent.iterdir():
        if path.name != 'config.json': (checkpoint/path.name).symlink_to(path.resolve())
    cfg = low.read(parent/'config.json'); cfg['residual_phase_gate_mode'] = 'causal_rules'
    low.write(checkpoint/'config.json', cfg)
    model.update(key='krv_best_causal', label='KRV full epoch 3 + causal gate (inference only)',
        parent_checkpoint=str(parent), checkpoint=str(checkpoint), root=str(ROOT), gated=True,
        config_sha256=low.sha(checkpoint/'config.json'))
    assert low.sha(checkpoint/'model.safetensors') == model['weight_sha256']
    plan = deepcopy(base); plan['models'] = [model]; plan['jobs'] = []
    for original in base['jobs']:
        if original['model'] != 'krv_best_constant': continue
        job = deepcopy(original); job['model'] = model['key']
        job['id'] = job['id'].replace('krv_best_constant', model['key'])
        folder = ROOT/'blocks'/job['id']; folder.mkdir(parents=True)
        job['folder'] = str(folder); (folder/'case.txt').write_text(job['scene']+'\n')
        plan['jobs'].append(job)
    plan['protocol'].update(models=1, benchmark='Matched causal gate ablation', baseline=str(PARENT),
        gate='existing causal_rules', inference_only=True, trained_with_gate=False,
        change='Only reconstruction gate; same epoch-3 KRV weights, scenes, seeds, prompts, speed and controller')
    plan['files_sha256'][str(Path(__file__).resolve())] = low.sha(__file__)
    plan['created_utc'] = runner.stamp()
    low.write(ROOT/'plan.json', plan)
    low.write(ROOT/'protocol.json', {**plan['protocol'], 'tests': plan['tests']})
    low.write(ROOT/'request.json', dict(user_authorized=True, dependency=str(PARENT/'completion.json'),
        episodes=25, next='closed-loop-speed-ood-20261112', inference_only=True))
    status('queued_waiting_for_baseline_completion')
    return plan

def verify_matched(plan):
    import numpy as np
    import json
    baseline = low.read(PARENT/'plan.json')
    for job in plan['jobs']:
        ref = next(j for j in baseline['jobs'] if j['scene'] == job['scene'] and j['model'] == 'krv_best_constant')
        a = json.loads((Path(job['folder'])/'reset_states.jsonl').read_text().strip())['state']
        b = json.loads((Path(ref['folder'])/'reset_states.jsonl').read_text().strip())['state']
        for key in ['robot_joint_pos', 'robot_joint_vel']: np.testing.assert_allclose(a[key], b[key], atol=1e-5, rtol=0)
        assert a['objects'].keys() == b['objects'].keys()
        for key in a['objects']: np.testing.assert_allclose(a['objects'][key], b['objects'][key], atol=1e-5, rtol=0)
    low.write(ROOT/'baseline_pairing.json', dict(verified=True, configs=25, matched_reset_atol=1e-5,
        unchanged_weights=True, baseline=str(PARENT)))

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare-only', action='store_true'); args = parser.parse_args()
    plan = prepare()
    if args.prepare_only: return
    signal.signal(signal.SIGTERM, lambda *_: runner.STOP.set())
    signal.signal(signal.SIGINT, lambda *_: runner.STOP.set())
    with (ROOT/'runner.lock').open('a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not runner.STOP.is_set():
            complete = PARENT/'completion.json'
            if complete.exists():
                result = low.read(complete)
                assert result['verified'] and result['complete'] and set(result['trials'].values()) == {25}
                break
            parent = low.read(PARENT/'status.json')
            if parent['stage'] == 'failed': status('blocked_baseline_failed'); return
            status('queued_waiting_for_baseline_completion', baseline_completed=parent.get('completed', 0))
            runner.STOP.wait(10)
        if runner.STOP.is_set(): return
        with (low.ROOT/'queue.lock').open('a') as lock:
            while not runner.STOP.is_set():
                try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); break
                except BlockingIOError: runner.STOP.wait(5)
            if runner.STOP.is_set(): return
            try:
                status('starting')
                runner.launch(plan)
                verify_matched(plan)
                status('complete', baseline_pairing_verified=True)
            except BaseException as exc:
                status('failed', error=repr(exc)); raise

if __name__ == '__main__': main()
