"""Queue unchanged KRV versus KRV+innovation behind the frozen DOM experiment."""
from pathlib import Path

# Derive the same launch/controller/verification protocol without editing files
# frozen by the active DOM experiment. Checked substitutions fail closed.
_parent = Path(__file__).with_name('run_velocity_innovation.py')
_text = _parent.read_text()
_edits = {
    "closed-loop-velocity-innovation-20261112": "closed-loop-krv-velocity-innovation-20261112",
    "MODES = ('none', 'innovation', 'naive')": "MODES = ('none', 'innovation')",
    "m['key'] == 'dom_pretrained'": "m['key'] == 'krv_best_constant'",
    "shift = test['seed'] % 3": "shift = test['seed'] % 2",
    "len(scores) == 9": "len(scores) == 6",
    '225': '150',
    "if __name__ == '__main__':\n    main()": '',
}
for _old, _new in _edits.items():
    assert _old in _text, _old
    _text = _text.replace(_old, _new)
exec(compile(_text, str(_parent), 'exec'))

DOM = ROOT.parent/'closed-loop-velocity-innovation-20261112'
_prepare, _verify = prepare, verify
PUBLISH_SHARED = False


def status(stage, **kwargs):
    kwargs['total'] = 150
    low.write(ROOT/'status.json', dict(stage=stage, updated_utc=runner.stamp(), **kwargs))
    from compare_krv_velocity_innovation import render
    render(shared=PUBLISH_SHARED)


def prepare():
    existing = (ROOT/'plan.json').exists()
    plan = _prepare()
    if existing:
        return plan
    cfg = low.read(Path(plan['models'][0]['checkpoint'])/'config.json')
    assert cfg['enable_residual_flow'] and cfg['condition_on_object_velocity']
    assert cfg['residual_phase_gate_mode'] == 'constant'
    dom = low.read(DOM/'plan.json')
    assert plan['protocol']['correction'] == dom['protocol']['correction']
    assert {t['path']:t['config_sha256'] for t in plan['tests']} == {t['path']:t['config_sha256'] for t in dom['tests']}
    plan['protocol'].update(observations='Unchanged native velocity-conditioned KRV; same oracle velocity to execution correction',
        native_kr_mode='constant; retained identically in both arms',
        comparison='KRV full-data epoch-3 validation-best, unchanged versus added velocity innovation; compare with DOM arms',
        total_episodes=150, dependency=str(DOM/'completion.json'),
        model_input_change=False, external_correction_only=True)
    for p in [Path(__file__), _parent, Path(__file__).with_name('compare_krv_velocity_innovation.py')]:
        plan['files_sha256'][str(p.resolve())] = low.sha(p)
    low.write(ROOT/'plan.json', plan)
    low.write(ROOT/'protocol.json', {**plan['protocol'], 'tests':plan['tests']})
    (ROOT/'README.md').write_text('# KRV with training-free velocity innovation\n\n'
        '150 trials: 25 cases × 3 speeds (0.10, 0.50, 1.00 m/s) × 2 arms. '
        'Uses the same full-data epoch-3 validation-best KRV checkpoint as the live comparison. '
        'Weights, learned velocity conditioning, and constant native KR are unchanged. '
        'The added correction is the same frozen controller used for DOM, applied after native action reconstruction.\n\n'
        'The unchanged KRV arm is rerun with the same per-observation noise schedule. '
        'No training, extra motion disturbances, or correction parameter tuning. '
        'GPU execution waits for the active DOM evaluation to finish and pass verification.\n\n'
        '[Live DOM/KRV comparison](comparison.md). Results also enter the main live_results.md when KRV begins.\n')
    status('queued_after_dom', completed=0)
    return plan


def verify(plan):
    # Intra-KRV reset equality, causal replay, videos, bounds, and unchanged
    # orientation/gripper are checked by the identical DOM verifier.
    result = _verify(plan)
    # Add cross-policy reset equality, rather than assuming matching JSON suffices.
    dom_plan = low.read(DOM/'plan.json')
    dom_jobs = {j['scene']:j for j in dom_plan['jobs'] if j['correction_mode'] == 'none'}
    for j in plan['jobs']:
        if j['correction_mode'] != 'none':
            continue
        a = json.loads((Path(j['folder'])/'reset_states.jsonl').read_text())['state']
        b = json.loads((Path(dom_jobs[j['scene']]['folder'])/'reset_states.jsonl').read_text())['state']
        for key in ('robot_joint_pos', 'robot_joint_vel'):
            np.testing.assert_allclose(a[key], b[key], atol=1e-5, rtol=0)
        for key in a['objects']:
            np.testing.assert_allclose(a['objects'][key], b['objects'][key], atol=1e-5, rtol=0)
    result['dom_krv_matched_resets_verified'] = True
    low.write(ROOT/'completion.json', result)
    return result


runner.status = status
runner.verify = verify


def main():
    global PUBLISH_SHARED
    parser = argparse.ArgumentParser();parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args();plan = prepare()
    if args.prepare_only:
        print(json.dumps(dict(plan=str(ROOT/'plan.json'), trials=len(plan['jobs']))));return
    signal.signal(signal.SIGTERM, lambda *_:runner.STOP.set())
    signal.signal(signal.SIGINT, lambda *_:runner.STOP.set())
    with (ROOT/'runner.lock').open('a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not runner.STOP.is_set():
            state = low.read(DOM/'status.json')
            if state['stage'] == 'complete':
                proof = low.read(DOM/'completion.json')
                assert proof['complete'] and proof['verified']
                break
            if state['stage'] == 'failed':
                status('blocked_dom_failed', completed=0, dependency_error=state.get('error'));return
            status('queued_after_dom', completed=0, dom_completed=state.get('completed', 0))
            runner.STOP.wait(10)
        if runner.STOP.is_set():
            return
        with (low.ROOT/'queue.lock').open('a') as lock:
            while not runner.STOP.is_set():
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB);break
                except BlockingIOError:
                    runner.STOP.wait(10)
            if runner.STOP.is_set():return
            PUBLISH_SHARED = True
            try:
                status('starting', completed=0)
                runner.launch(plan)
            except BaseException as exc:
                (ROOT/'completion.json').unlink(missing_ok=True)
                status('failed', error=repr(exc));raise


if __name__ == '__main__':
    main()
