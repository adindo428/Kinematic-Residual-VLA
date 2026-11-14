"""Frozen 225-episode DOM comparison on the three existing live-test speeds."""
import argparse
from collections import defaultdict
from copy import deepcopy
import fcntl
import inspect
import json
from pathlib import Path
import signal

import numpy as np
import run_closed_loop_pilot as runner

low = runner.low
ROOT = low.WORK/'evaluations/closed-loop-velocity-innovation-20261112'
runner.ROOT = ROOT
MODES = ('none', 'innovation', 'naive')


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'plan.json').exists():
        return low.read(ROOT/'plan.json')
    base = low.read(ROOT.parent/'closed-loop-pilot-20261112/plan.json')
    model = deepcopy(next(m for m in base['models'] if m['key'] == 'dom_pretrained'))
    plan = deepcopy(base)
    plan.update(models=[model], jobs=[], tests=[], saved_prompts={}, created_utc=runner.stamp())
    for name in ('closed-loop-pilot-20261112', 'closed-loop-speed-mid-20261112', 'closed-loop-speed-ood-20261112'):
        parent = low.read(ROOT.parent/name/'plan.json')
        assert low.read(ROOT.parent/name/'completion.json')['verified']
        for test in parent['tests']:
            assert low.sha(test['path']) == test['config_sha256']
            plan['tests'].append(deepcopy(test))
            plan['saved_prompts'][test['path']] = parent['saved_prompts'][test['path']]
            speed = test['initial_speed_mps']
            # Counterbalance arm order; model noise is tied to seed+observation index.
            shift = test['seed'] % 3
            for mode in MODES[shift:]+MODES[:shift]:
                uid = f"speed{speed:.2f}-{test['name']}-{mode}"
                folder = ROOT/'blocks'/uid
                folder.mkdir(parents=True)
                (folder/'case.txt').write_text(test['path']+'\n')
                plan['jobs'].append(dict(id=uid, model=model['key'], correction_mode=mode,
                    initial_speed_mps=speed, scene=test['path'], scene_sha256=test['config_sha256'],
                    repeats=1, repeat_offset=0, folder=str(folder), dimension='velocity_innovation',
                    scored=True, layout=test['layout'], seed=test['seed']))
    plan['protocol'] = dict(training_free=True, checkpoint=model['checkpoint'], checkpoint_sha256=model['weight_sha256'],
        modes=list(MODES), episodes_per_mode_per_speed=25, total_episodes=225,
        speeds_mps=[.1, .5, 1.], closed_loop=True, streaming=True,
        correction=dict(gain=1., radius_m=.05, max_offset_change_per_step_m=.005,
            innovation='integral(v(s)-v(t0), t0, t); t0 = source observation of active chunk',
            naive='v(t)*(t-t0)', gate='latched off on first closing command; rate-limited fade'),
        observations='Unchanged RGB+proprioception to frozen DOM; oracle robot-frame velocity only to controller',
        object_position_used_by_correction=False, destination_pose_used_by_correction=False,
        noise_pairing='fixed torch noise per simulator seed and source observation index; asynchronous scheduling may differ',
        novelty_scope='development comparison on previously inspected layouts, not a fresh IID/OOD generalization claim',
        perturbations='No injected disturbances; existing initial-speed cases with natural dynamics',
        tuning='One frozen parameter setting; no success-driven tuning', tolerance_m=.03)
    files = [Path(p) for p in base['files_sha256']]
    files += list((low.REPO/'workstation').glob('*velocity_innovation*.py'))
    plan['files_sha256'] = {str(p.resolve()):low.sha(p) for p in files}
    assert len(plan['jobs']) == 225
    low.write(ROOT/'plan.json', plan)
    low.write(ROOT/'protocol.json', {**plan['protocol'], 'tests':plan['tests']})
    (ROOT/'README.md').write_text('# Training-free velocity innovation\n\n'
        'Released DOM, unchanged weights. 25 matched cases × 3 initial speeds × 3 correction arms = 225 episodes.\n\n'
        'Uses oracle simulator velocity, no object/destination positions in correction. '
        'No disturbances injected; speeds and geometry are exactly those in the existing live test bed. '
        'The same fresh seeded DOM baseline is rerun. Original 300 trial records are preserved.\n\n'
        'Correction parameters are frozen in protocol.json. Main dashboard: '
        '[live_results.md](../closed-loop-pilot-20261112/live_results.md).\n')
    status('prepared', completed=0)
    return plan


def status(stage, **kwargs):
    kwargs['total'] = 225
    low.write(ROOT/'status.json', dict(stage=stage, updated_utc=runner.stamp(), **kwargs))
    from watch_closed_loop_combined_markdown import render
    render()


def verify(plan):
    states, prompts = {}, {}
    scores = defaultdict(lambda: [0, 0])
    corrections = defaultdict(list)
    for job in plan['jobs']:
        folder = Path(job['folder'])
        receipt = low.read(folder/'resident_client_completion.json')
        assert receipt['complete'] and receipt['trials'] == 1
        assert receipt['checkpoint_sha256'] == plan['models'][0]['weight_sha256']
        rows = [json.loads(s) for s in (folder/'trial_results.jsonl').read_text().splitlines()]
        assert len(rows) == 1 and rows[0]['status'] in (0, 1) and rows[0]['n_steps'] > 1
        assert rows[0]['env_name'] == Path(job['scene']).stem and rows[0]['trial'] == 1
        snap = json.loads((folder/'reset_states.jsonl').read_text().strip())['state']
        if job['scene'] in states:
            ref = states[job['scene']]
            for key in ('robot_joint_pos', 'robot_joint_vel'):
                np.testing.assert_allclose(snap[key], ref[key], atol=1e-5, rtol=0)
            assert snap['objects'].keys() == ref['objects'].keys()
            for key in snap['objects']:
                np.testing.assert_allclose(snap['objects'][key], ref['objects'][key], atol=1e-5, rtol=0)
        else:
            states[job['scene']] = snap
        prompt = json.loads((folder/'prompt_trace.jsonl').read_text().strip())['text']
        assert prompts.setdefault(job['scene'], prompt) == prompt
        assert list((folder/'videos').rglob('*.mp4'))
        summary = low.read(folder/'velocity_correction_summary.json')
        assert summary['mode'] == job['correction_mode']
        assert summary['steps'] == rows[0]['n_steps']
        assert summary['max_offset_m'] <= .050001
        assert not summary['object_position_used'] and not summary['destination_position_used']
        # Independently replay the controller from only causal velocity samples and actions.
        from velocity_innovation import VelocityInnovation
        replay = VelocityInnovation(job['correction_mode'], rows[0]['step_dt_s'])
        for row in map(json.loads, (folder/'velocity_correction.jsonl').read_text().splitlines()):
            replay.observe(row['index'], row['velocity'])
            if row['origin'] is not None:
                replay.set_origin(row['origin'])
            actual = replay.correct(np.asarray(row['base_action'], dtype=np.float32))
            np.testing.assert_allclose(actual, row['applied_action'], atol=1e-7, rtol=0)
            np.testing.assert_array_equal(actual[:, 3:], np.asarray(row['base_action'], dtype=np.float32)[:, 3:])
        if job['correction_mode'] == 'none':
            assert summary['max_offset_m'] == 0
        group = f"{job['initial_speed_mps']:.2f}/{job['correction_mode']}"
        scores[group][0] += int(rows[0]['success'])
        scores[group][1] += 1
        corrections[group].append(summary['max_offset_m'])
    assert all(v[1] == 25 for v in scores.values()) and len(scores) == 9
    result = dict(complete=True, verified=True, completed_utc=runner.stamp(),
        placement_successes_and_trials=dict(scores),
        max_correction_m={k:max(v) for k,v in corrections.items()},
        paired_scene_resets=True, controller_causal_replay_verified=True,
        orientation_and_gripper_unchanged=True, original_baseline_records_preserved=True)
    low.write(ROOT/'completion.json', result)
    return result


# Keep original process cleanup, watchdog, simulator arguments and timing.
source = inspect.getsource(runner.launch)
source = source.replace('closed_loop_pilot_server.py', 'velocity_innovation_server.py')
source = source.replace('closed_loop_pilot_client.py', 'velocity_innovation_client.py')
source = source.replace('total=75', "total=len(plan['jobs'])").replace('completed=75', "completed=len(plan['jobs'])")
runner.status = status
runner.verify = verify
exec(compile(source, '<velocity_innovation_launch>', 'exec'), runner.__dict__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    plan = prepare()
    if args.prepare_only:
        print(json.dumps({'plan':str(ROOT/'plan.json'), 'jobs':len(plan['jobs'])}))
        return
    signal.signal(signal.SIGTERM, lambda *_:runner.STOP.set())
    signal.signal(signal.SIGINT, lambda *_:runner.STOP.set())
    with (ROOT/'runner.lock').open('a') as own, (low.ROOT/'queue.lock').open('a') as lock:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not runner.STOP.is_set():
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                status('waiting_for_evaluation_gpu_lock', completed=0)
                runner.STOP.wait(10)
        if runner.STOP.is_set():
            return
        try:
            status('starting', completed=0)
            runner.launch(plan)
        except BaseException as exc:
            status('failed', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
