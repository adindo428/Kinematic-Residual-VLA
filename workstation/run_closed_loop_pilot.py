"""Prepare and queue 5 layouts x 5 seeds x 3 frozen native policies."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import threading
import time

import run_low_clutter_table1 as low

ROOT = low.WORK/'evaluations/closed-loop-pilot-20261112'
STOP = threading.Event()

def stamp(): return datetime.now(timezone.utc).isoformat()

def status(stage, **kw):
    low.write(ROOT/'status.json', dict(stage=stage, updated_utc=stamp(), **kw))

def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'plan.json').exists(): return low.read(ROOT/'plan.json')
    old = low.read(low.WORK/'evaluations/krv-full-original-dom-20261106/constant_resume/request.json')
    models = [deepcopy(next(m for m in old['models'] if m['key'] == key))
              for key in ['domv_best', 'krv_best_constant']]
    base = low.read(low.WORK/'results/dom-paper45-a6000-20261010-02/protocol.json')
    models.append(dict(key='dom_pretrained', label='Released DOM pretrained', root=str(ROOT),
        checkpoint=base['checkpoint'], source=str(low.WORK/'official/source'), kr=False, gated=False,
        weight_sha256=base['checkpoint_weights_sha256'], config_sha256=base['checkpoint_config_sha256']))
    for model in models:
        cp = Path(model['checkpoint'])
        assert low.sha(cp/'model.safetensors') == model['weight_sha256']
        assert low.sha(cp/'config.json') == model['config_sha256']
    source_paths = (low.WORK/'evaluations/krv-full-original-dom-20261106/test_lists/00_CR.txt').read_text().splitlines()[:5]
    jobs, tests, prompts = [], [], {}
    (ROOT/'configs').mkdir(exist_ok=True)
    for layout, source in enumerate(source_paths):
        original = low.read(source)
        assert original['instruction']['task'] == 'place'
        assert [k for k in original['scene'] if k.startswith('object')] == ['object']
        for seed_index in range(5):
            seed = 711200 + layout*100 + seed_index
            rng = random.Random(seed)
            cfg = deepcopy(original)
            cfg['seed'] = seed
            # Small independently seeded offsets keep the original support surface
            # and assets, while producing physically distinct configurations.
            changes = {}
            for key in ['object', 'container']:
                shift = [rng.choice([-1, 1])*rng.uniform(.015, .03), rng.uniform(-.025, .025)]
                for axis in range(2): cfg['scene'][key]['init_state']['pos'][axis] += shift[axis]
                changes[key] = shift
            velocity = cfg['scene']['object']['init_state']['lin_vel']
            norm = math.hypot(*velocity[:2]); assert norm > 0
            cfg['scene']['object']['init_state']['lin_vel'] = [.10*velocity[0]/norm, .10*velocity[1]/norm, 0.]
            cfg['events'].pop('perturbation', None)
            name = f'pilot_layout{layout+1:02d}_seed{seed}'
            prompt = f"Place the {cfg['instruction']['objects'][0]} into the {cfg['instruction']['containers'][0]}."
            cfg['evaluation_case_id'] = name
            cfg['evaluation_prompts'] = [prompt]*20
            path = ROOT/'configs'/f'{name}.json'; low.write(path, cfg)
            prompts[str(path)] = {'1': prompt}
            tests.append(dict(name=name, path=str(path), config_sha256=low.sha(path),
                layout=layout+1, seed=seed, source=source, source_sha256=low.sha(source),
                offsets_world_xy_m=changes, horizon_seconds=cfg['episode_length_s'], initial_speed_mps=.10))
            for model in models:
                uid = name+'-'+model['key']; folder = ROOT/'blocks'/uid
                folder.mkdir(parents=True); (folder/'case.txt').write_text(str(path)+'\n')
                jobs.append(dict(id=uid, model=model['key'], scene=str(path), scene_sha256=low.sha(path),
                    repeats=1, repeat_offset=0, folder=str(folder), dimension='pilot', scored=True,
                    layout=layout+1, seed=seed))
    assert len(jobs) == 75 and len({t['config_sha256'] for t in tests}) == 25
    files = list((low.REPO/'workstation').glob('closed_loop_pilot_*.py')) + [Path(__file__)]
    files += [low.REPO/'workstation'/n for n in ['resident_table1_worker.py', 'resident_table1_client.py',
        'resident_table1_server.py', 'resident_low_clutter_server.py', 'low_clutter_table1_server.py',
        'collection_matched_server.py', 'causal_gate_diagnostic_server.py', 'table1_video_sampling.py']]
    for source in {m['source'] for m in models}:
        files += sorted(Path(source).rglob('*.py'))
    plan = dict(models=models, jobs=jobs, tests=tests, success_history={}, saved_prompts=prompts,
        all_trials_excluded_from_table1=True, created_utc=stamp(),
        files_sha256={str(p.resolve()): low.sha(p) for p in files},
        protocol=dict(episodes_per_model=25, models=3, closed_loop=True, streaming=True,
            initial_speed_mps=.10, speed_sweep=False, perturbations=False,
            seed_scope='25 frozen physical layouts; simulator seeds matched; policy sampling RNG not paired',
            novelty='New seeded offsets using existing DOM room/object assets; not certified unseen training assets',
            video_policy='Every episode (one repeat per unique config)', tolerance_m=.03))
    low.write(ROOT/'plan.json', plan)
    low.write(ROOT/'protocol.json', {**plan['protocol'], 'tests': tests})
    status('prepared', trials=75)
    return plan

def verify(plan):
    import numpy as np
    states = {}; prompts = {}; counts = Counter(); successes = Counter(); picked = Counter()
    for job in plan['jobs']:
        folder = Path(job['folder'])
        receipt = low.read(folder/'resident_client_completion.json')
        model = next(m for m in plan['models'] if m['key'] == job['model'])
        assert receipt['complete'] and receipt['trials'] == 1
        assert receipt['checkpoint_sha256'] == model['weight_sha256']
        rows = [json.loads(s) for s in (folder/'trial_results.jsonl').read_text().splitlines()]
        assert len(rows) == 1 and rows[0]['status'] in (0, 1)
        assert rows[0]['env_name'] == Path(job['scene']).stem and rows[0]['trial'] == 1
        assert rows[0]['n_steps'] > 1
        snap = json.loads((folder/'reset_states.jsonl').read_text().strip())['state']
        if job['scene'] in states:
            ref = states[job['scene']]
            for k in ['robot_joint_pos', 'robot_joint_vel']: np.testing.assert_allclose(snap[k], ref[k], atol=1e-5, rtol=0)
            assert snap['objects'].keys() == ref['objects'].keys()
            for k in snap['objects']: np.testing.assert_allclose(snap['objects'][k], ref['objects'][k], atol=1e-5, rtol=0)
        else: states[job['scene']] = snap
        text = json.loads((folder/'prompt_trace.jsonl').read_text().strip())['text']
        assert prompts.setdefault(job['scene'], text) == text
        assert list((folder/'videos').rglob('*.mp4')), 'Missing episode video'
        counts[job['model']] += 1; successes[job['model']] += rows[0]['success']
        picked[job['model']] += low.read(folder/'pickup_diagnostic.json')['heuristic_pickup']
    assert set(counts.values()) == {25}
    result = dict(complete=True, verified=True, trials=dict(counts), placement_successes=dict(successes),
        heuristic_pickups=dict(picked), matched_reset_atol=1e-5, completed_utc=stamp())
    low.write(ROOT/'completion.json', result)
    lines = ['# Closed-loop pilot', '', '25 episodes/model; new layouts using existing DOM assets; initial speed 0.10 m/s.', '',
             '| Model | Placement success | Heuristic pickup |', '|---|---:|---:|']
    for model in plan['models']:
        key = model['key']; lines.append(f"| {model['label']} | {successes[key]}/25 | {picked[key]}/25 |")
    lines += ['', 'Pickup is a lift/proximity heuristic, not contact-verified. This small pilot does not establish overfitting.']
    (ROOT/'comparison.md').write_text('\n'.join(lines)+'\n')
    return result

def launch(plan):
    execution = ROOT/'execution'; execution.mkdir(exist_ok=True)
    assert not any((Path(j['folder'])/'trial_results.jsonl').exists() for j in plan['jobs']), 'Do not overwrite a partial pilot'
    for path, checksum in plan['files_sha256'].items(): assert low.sha(path) == checksum, path
    for model in plan['models']:
        assert low.sha(Path(model['checkpoint'])/'model.safetensors') == model['weight_sha256']
        assert low.sha(Path(model['checkpoint'])/'config.json') == model['config_sha256']
    kit = low.WORK/'kit/closed-loop-pilot'
    kit_args = os.environ['KRVLA_KIT_ARGS'].replace('--portable-root='+str(low.WORK/'kit'), '--portable-root='+str(kit))
    for token, sub in [('omni_global_cache', 'cache'), ('omni_global_logs', 'logs'), ('omni_global_data', 'data')]:
        kit_args = kit_args.replace(f'--/app/tokens/{token}='+str(low.WORK/sub/'omniverse'), f'--/app/tokens/{token}='+str(kit/sub))
    kit_args += ' --/app/extensions/fsWatcherEnabled=false --/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false'
    server = [str(low.REPO/'workstation/isaaclab-paper45.sh'), '-p', str(low.REPO/'workstation/closed_loop_pilot_server.py'),
        '--headless', '--enable_cameras', '--device', 'cuda:0', '--num_envs', '1', '--tolerance', '.03',
        '--scene_dir', str(low.WORK/'assets/scenes'), '--object_dir', str(low.WORK/'assets/objects'),
        '--env_cfg', plan['jobs'][0]['scene'], '--output_dir', str(execution/'videos'), '-n', '1', '--save',
        '--img_port', '27600', '--act_port', '27601', '--kit_args', kit_args]
    client = [str(low.WORK/'envs/train/bin/python'), '-u', str(low.REPO/'workstation/closed_loop_pilot_client.py'),
        '--plan', str(ROOT/'plan.json'), '--output', str(execution)]
    env = dict(os.environ, DOM_OFFICIAL_SOURCE=str(low.WORK/'official/source'), DOM_RESIDENT_PLAN=str(ROOT/'plan.json'),
        DOM_TRIAL_RESULTS=str(execution/'unused.jsonl'), DOM_EXPECTED_SIM='4.5.0', DOM_EXPECTED_LAB='2.2.1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    low.write(execution/'commands.json', dict(server=server, client=client))
    processes = []
    try:
        with (execution/'server_shared.log').open('w') as sl, (execution/'client_shared.log').open('w') as cl:
            processes.append(subprocess.Popen(server, env=dict(env, CUDA_VISIBLE_DEVICES='0'), stdout=sl, stderr=subprocess.STDOUT, start_new_session=True))
            processes.append(subprocess.Popen(client, env=dict(env, CUDA_VISIBLE_DEVICES='1'), stdout=cl, stderr=subprocess.STDOUT, start_new_session=True))
            previous = -1; changed = time.monotonic()
            while processes[1].poll() is None:
                if STOP.wait(5): raise InterruptedError('Pilot stopped')
                state = low.read(execution/'status.json') if (execution/'status.json').exists() else {}
                if processes[0].poll() is not None and state.get('stage') != 'complete': raise RuntimeError('Pilot simulator exited')
                count = sum((Path(j['folder'])/'resident_client_completion.json').exists() for j in plan['jobs'])
                if count != previous: previous = count; changed = time.monotonic()
                if time.monotonic()-changed > 1800: raise TimeoutError('Pilot made no completed-block progress for 30 minutes')
                status(state.get('stage', 'starting'), completed=count, total=75, job=state.get('job'))
            if processes[1].returncode: raise RuntimeError('Pilot client failed; inspect execution/client_shared.log')
        verify(plan); status('complete', completed=75, total=75)
    finally:
        for process in reversed(processes): low.stop_group(process)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare-only', action='store_true'); args = parser.parse_args()
    plan = prepare()
    if args.prepare_only: print(json.dumps(dict(plan=str(ROOT/'plan.json'), jobs=len(plan['jobs'])))); return
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    with (ROOT/'runner.lock').open('a') as own, (low.ROOT/'queue.lock').open('a') as lock:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not STOP.is_set():
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except BlockingIOError:
                status('queued_waiting_for_evaluation_gpu_lock', total=75)
                STOP.wait(10)
        if STOP.is_set(): return
        try: launch(plan)
        except BaseException as exc:
            status('failed', error=repr(exc)); raise

if __name__ == '__main__': main()
