"""Matched 0.5 m/s evaluation of DOM, DOMV, constant KR and causal KR."""
from copy import deepcopy
from pathlib import Path
import queue_closed_loop_speed_ood as queue

BASE = queue.PARENT
ROOT = BASE.parent/'closed-loop-speed-mid-20261112'
queue.ROOT = ROOT
queue.runner.ROOT = ROOT
low = queue.low
_status = queue.runner.status
_verify = queue.runner.verify

def status(stage, **kw):
    kw['total'] = 100
    if stage == 'complete': kw['completed'] = 100
    _status(stage, **kw)

def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'plan.json').exists(): return low.read(ROOT/'plan.json')
    base = low.read(BASE/'plan.json')
    gated = low.read(BASE.parent/'closed-loop-causal-gate-20261112/plan.json')
    plan = deepcopy(base)
    plan['models'].append(deepcopy(gated['models'][0]))
    mapping = {}; prompts = {}
    (ROOT/'configs').mkdir(exist_ok=True)
    for test in plan['tests']:
        old = Path(test['path']); cfg = low.read(old)
        assert low.sha(old) == test['config_sha256']
        v = cfg['scene']['object']['init_state']['lin_vel']
        cfg['scene']['object']['init_state']['lin_vel'] = [5*x for x in v]
        path = ROOT/'configs'/old.name; low.write(path, cfg)
        mapping[str(old)] = str(path); prompts[str(path)] = base['saved_prompts'][str(old)]
        test.update(path=str(path), config_sha256=low.sha(path), initial_speed_mps=.5,
            matched_baseline_path=str(old), matched_baseline_sha256=low.sha(old))
    tests = {t['path']: t for t in plan['tests']}
    jobs = []
    for original in base['jobs']:
        candidates = [deepcopy(original)]
        if original['model'] == 'krv_best_constant':
            extra = deepcopy(original); extra['model'] = 'krv_best_causal'
            extra['id'] = extra['id'].replace('krv_best_constant', 'krv_best_causal')
            candidates.append(extra)
        for job in candidates:
            job['scene'] = mapping[job['scene']]
            job['scene_sha256'] = tests[job['scene']]['config_sha256']
            folder = ROOT/'blocks'/job['id']; folder.mkdir(parents=True)
            job['folder'] = str(folder); (folder/'case.txt').write_text(job['scene']+'\n')
            jobs.append(job)
    plan.update(jobs=jobs, saved_prompts=prompts, created_utc=queue.runner.stamp())
    plan['protocol'].update(models=4, initial_speed_mps=.5, benchmark='Intermediate initial-speed comparison',
        baseline=str(BASE), change='Only initial velocity magnitude changes from 0.1 to 0.5 m/s; fourth arm uses existing inference-only causal gate',
        speed_scope='Within the saved collection initial-speed range; not a claim of full-task IID',
        causal_gate_training=False)
    plan['files_sha256'][str(Path(__file__).resolve())] = low.sha(__file__)
    plan['files_sha256'][str(Path(queue.__file__).resolve())] = low.sha(queue.__file__)
    assert len(jobs) == 100
    low.write(ROOT/'plan.json', plan)
    low.write(ROOT/'protocol.json', {**plan['protocol'], 'tests': plan['tests']})
    low.write(ROOT/'request.json', dict(user_authorized=True,
        dependency=str(BASE.parent/'closed-loop-ood-causal-gate-20261112/completion.json'),
        initial_speed_mps=.5, total_episodes=100))
    status('queued_waiting_for_ood_causal_gate')
    return plan

def verify(plan):
    result = _verify(plan)
    path = ROOT/'comparison.md'
    path.write_text(path.read_text().replace('initial speed 0.10 m/s', 'initial speed 0.50 m/s'))
    return result

# Reuse dependency/lock management, with the intermediate protocol and correct
# summary text (the OOD queue's launch epilogue is specific to 1.0 m/s).
_launch = queue.runner.launch
def launch(plan):
    _launch(plan)

queue.prepare = prepare
queue.runner.status = status
queue.runner.verify = verify
queue.runner.launch = launch

if __name__ == '__main__':
    queue.main()
    if (ROOT/'completion.json').exists():
        path = ROOT/'comparison.md'
        text = path.read_text()
        text = text.replace('\nOOD scope: outside the saved collection initial-speed range; not an exhaustive bound on training velocities.\n',
            '\nInitial speed 0.50 m/s is within the saved collection range. Causal gating is inference-only.\n')
        path.write_text(text)
