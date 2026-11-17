"""Continue the 1800/model extension, visiting each remaining scene once."""
import copy
import fcntl
import inspect
import json
from pathlib import Path
import signal

import extend_dom_krv_epoch4_1800 as old

runner = old.runner
low = old.low
ROOT = old.ROOT / 'scene-first'


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT/'plan.json').exists(), 'Use a separately audited continuation after interruption'
    phases = {k: old.read(old.ROOT/k/'plan.json') for k in ['krv_epoch4', 'dom']}
    models = []
    prompts = {}
    completed = []
    for phase, plan in phases.items():
        model = copy.deepcopy(plan['models'][0])
        model['deterministic_observation_noise'] = phase == 'dom'
        assert not model.get('vpd') and not model.get('vi')
        models.append(model)
        for scene, values in plan['saved_prompts'].items():
            for key, text in values.items():
                saved = prompts.setdefault(scene, {})
                assert key not in saved or saved[key] == text
                saved[key] = text
        for job in plan['jobs']:
            folder = Path(job['folder'])
            for row in old.records(folder/'prompt_trace.jsonl'):
                key = str(row['global_trial'])
                saved = prompts.setdefault(job['scene'], {})
                assert key not in saved or saved[key] == row['text']
                saved[key] = row['text']
            rows = old.records(folder/'trial_results.jsonl')
            receipt = folder/'resident_client_completion.json'
            if receipt.exists():
                done = old.read(receipt)
                assert done['complete'] and done['trials'] == len(rows) == job['repeats']
                completed.append(job['id'])
            else:
                assert not rows, f'Partial block requires explicit tail plan: {folder}'
    jobs = []
    for test in phases['dom']['tests']:
        for phase, plan in phases.items():
            job = next(j for j in plan['jobs'] if j['scene'] == test['path'])
            if job['id'] not in completed:
                assert not (Path(job['folder'])/'trial_results.jsonl').exists()
                jobs.append(copy.deepcopy(job))
    assert len(jobs) + len(completed) == 180
    files = list((low.REPO/'workstation').glob('*.py'))
    for source in {m['source'] for m in models}:
        files.extend(Path(source).rglob('*.py'))
    plan = dict(models=models, jobs=jobs, tests=phases['dom']['tests'], saved_prompts=prompts,
                success_history={}, files_sha256={str(p): low.sha(p) for p in files},
                protocol={'schedule': 'scene-first; one active inference worker',
                          'completed_blocks_retained': completed, 'target_each': 1800,
                          'KRV_noise': 'native ongoing RNG', 'DOM_noise': 'seed*1000003+observation_index',
                          'VPD': False, 'VI': False,
                          'runtime_change': 'shared passive profile/trajectory instrumentation; identity correction',
                          'original_phase_plans_retained': True}, created_utc=runner.stamp())
    old.write(ROOT/'plan.json', plan)
    old.write(old.ROOT/'scheduling-change.json', plan['protocol'])
    return plan


def verify(plan):
    for phase in ['krv_epoch4', 'dom']:
        runner.ROOT = old.ROOT/phase
        old.verify(old.read(runner.ROOT/'plan.json'))
    runner.ROOT = ROOT
    old.write(ROOT/'completion.json', {'verified': True, 'trials_per_model': 1800})
    old.write(old.ROOT/'completion.json', {'verified': True, 'trials_per_model': 1800,
                                         'new_trials': 1800, 'schedule': 'scene-first continuation'})
    old.render()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: runner.STOP.set())
    signal.signal(signal.SIGINT, lambda *_: runner.STOP.set())
    with (ROOT/'runner.lock').open('a') as own, (low.ROOT/'queue.lock').open('a') as gpu:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not runner.STOP.is_set():
            try:
                fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                runner.STOP.wait(5)
        if runner.STOP.is_set():
            return
        plan = prepare()
        runner.ROOT = ROOT
        runner.status = old.status
        runner.verify = verify
        code = inspect.getsource(old.ORIGINAL_LAUNCH)
        code = code.replace("'closed-loop-pilot'", "'dom-krv-1800-scene-first'")
        code = code.replace('total=75', "total=len(plan['jobs'])").replace('completed=75', "completed=len(plan['jobs'])")
        code = code.replace('closed_loop_pilot_server.py', 'dom_repeat_extension_server.py')
        code = code.replace('closed_loop_pilot_client.py', 'dom_krv_scene_client.py')
        exec(compile(code, __file__, 'exec'), runner.__dict__)
        try:
            runner.launch(plan)
        except BaseException as exc:
            old.status('failed', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
