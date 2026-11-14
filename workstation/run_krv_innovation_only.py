"""User-directed 75-trial KRV innovation run; reuse existing baselines."""
from copy import deepcopy
import fcntl
import inspect
import json
from pathlib import Path
import signal

import run_velocity_innovation as base

low = base.low
ROOT = low.WORK/'evaluations/closed-loop-krv-innovation-only-20261112'
base.ROOT = ROOT
base.runner.ROOT = ROOT


def render():
    from report_krv_innovation_only import render as report
    report()


def status(stage, **kw):
    kw['total'] = 75
    low.write(ROOT/'status.json', dict(stage=stage, updated_utc=base.runner.stamp(), **kw))
    render()


_code = inspect.getsource(base.verify).replace('len(scores) == 9', 'len(scores) == 3')
exec(compile(_code, '<innovation_only_verification>', 'exec'), base.__dict__)
_verify = base.verify


def verify(plan):
    result = _verify(plan)
    import numpy as np
    parents = {}
    for name in ('closed-loop-pilot-20261112', 'closed-loop-speed-mid-20261112', 'closed-loop-speed-ood-20261112'):
        p = low.read(ROOT.parent/name/'plan.json')
        for job in p['jobs']:
            if job['model'] == 'krv_best_constant':
                parents[job['scene']] = job
    for job in plan['jobs']:
        ref = parents[job['scene']]
        assert job['scene_sha256'] == ref['scene_sha256']
        a = json.loads((Path(job['folder'])/'reset_states.jsonl').read_text())['state']
        b = json.loads((Path(ref['folder'])/'reset_states.jsonl').read_text())['state']
        for key in ('robot_joint_pos', 'robot_joint_vel'):
            np.testing.assert_allclose(a[key], b[key], atol=1e-5, rtol=0)
        for key in a['objects']:
            np.testing.assert_allclose(a['objects'][key], b['objects'][key], atol=1e-5, rtol=0)
    result.update(historical_krv_resets_verified=True, policy_noise_paired=False)
    low.write(ROOT/'completion.json', result)


base.runner.status = status
base.runner.verify = verify


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'plan.json').exists():return low.read(ROOT/'plan.json')
    parent = ROOT.parent/'closed-loop-krv-velocity-innovation-20261112/plan.json'
    plan = deepcopy(low.read(parent))
    plan['jobs'] = [j for j in plan['jobs'] if j['correction_mode'] == 'innovation']
    for job in plan['jobs']:
        folder = ROOT/'blocks'/job['id'];folder.mkdir(parents=True)
        job['folder'] = str(folder);(folder/'case.txt').write_text(job['scene']+'\n')
    assert len(plan['jobs']) == 75
    plan['protocol'].update(modes=['innovation'], total_episodes=75, dependency=None,
        comparison='Reuse historical DOM and constant-KRV baselines; run only KRV+innovation',
        noise_pairing='Within new run per-observation seed schedule retained; historical baseline policy noise NOT paired',
        user_steering='Do not rerun DOM baseline; prioritize KRV+velocity innovation')
    plan['created_utc'] = base.runner.stamp()
    plan['parent_plan_sha256'] = low.sha(parent)
    for file in (Path(__file__), Path(__file__).with_name('report_krv_innovation_only.py')):
        plan['files_sha256'][str(file.resolve())] = low.sha(file)
    for m in plan['models']:
        assert low.sha(Path(m['checkpoint'])/'model.safetensors') == m['weight_sha256']
    low.write(ROOT/'plan.json', plan)
    low.write(ROOT/'protocol.json', {**plan['protocol'], 'tests':plan['tests']})
    (ROOT/'README.md').write_text('# KRV + velocity innovation only\n\n'
        '75 trials, 25 existing cases at each of 0.1/0.5/1.0 m/s. Same epoch-3 constant-KR checkpoint, '
        'same external controller and original parameters. No baseline reruns. '
        'Historical baseline policy noise is not paired; scene/reset comparisons are checked.\n\n'
        '[Live comparison](comparison.md). Supersedes the pending 150-trial two-arm KRV queue by user request.\n')
    status('prepared', completed=0)
    return plan


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    plan=prepare()
    if args.prepare_only:print(json.dumps(dict(root=str(ROOT),trials=75)));return
    signal.signal(signal.SIGTERM,lambda *_:base.runner.STOP.set())
    signal.signal(signal.SIGINT,lambda *_:base.runner.STOP.set())
    with (ROOT/'runner.lock').open('a') as own, (low.ROOT/'queue.lock').open('a') as gpu:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not base.runner.STOP.is_set():
            try:fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                status('waiting_for_gpu_handoff',completed=0);base.runner.STOP.wait(5)
        if base.runner.STOP.is_set():return
        try:
            status('starting',completed=0);base.runner.launch(plan)
        except BaseException as exc:
            (ROOT/'completion.json').unlink(missing_ok=True)
            status('failed',error=repr(exc));raise


if __name__ == '__main__':main()
