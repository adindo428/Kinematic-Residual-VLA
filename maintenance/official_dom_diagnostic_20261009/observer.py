"""Run official server with per-step EE/object/finger telemetry, without control changes."""
from collections import Counter
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys

def instrument(main, app, args, namespace):
    from isaacsim.core.version import get_version
    versions = {'isaacsim_runtime': list(get_version()),
                'isaacsim_version_file': Path('/isaac-sim/VERSION').read_text().strip(),
                'isaaclab_version_file': Path('/opt/IsaacLab/VERSION').read_text().strip()}
    for package in ('isaacsim', 'isaaclab'):
        try: versions[package+'_package'] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    destination = Path(os.environ['DOM_TRIAL_RESULTS'])
    (destination.parent/'simulator_versions.json').write_text(json.dumps(versions, indent=2)+'\n')
    # Match language randomization across diagnostic arms at startup.
    random.seed(20261009)
    original_get = namespace['get_sim_results']
    original_simulate = namespace['simulate']
    original_state = namespace['sim'].get_curr_state
    counts = Counter()
    context = {}

    def convert(value):
        return value.detach().cpu().tolist()

    def get_state(*a, **kw):
        state = original_state(*a, **kw)
        if context.get('active'):
            context['state'] = {
                key: {k: convert(v) for k,v in vals.items()}
                for key,vals in state.items()
            }
        return state

    def get_results(sim_cfg, path, obs, act):
        context['name'] = Path(path).stem
        return original_get(sim_cfg, path, obs, act)

    def simulate(env, obs, act, init):
        name = context['name']
        frames = []
        context.update(active=True, state=None)
        original_step = env.step
        robot = env.unwrapped.scene['robot']
        finger_ids = [i for i,n in enumerate(robot.joint_names) if n.startswith('panda_finger_joint')]
        def step(action):
            frames.append({'index':len(frames), 'action':convert(action),
                           'state':context['state'],
                           'finger_joint_positions':convert(robot.data.joint_pos[:,finger_ids]),
                           'wall_time':namespace['time'].perf_counter()})
            return original_step(action)
        env.step = step
        try: result = original_simulate(env, obs, act, init)
        finally:
            env.step = original_step
            context['active'] = False
        n_steps = len(result['ee_path'])
        if n_steps > 1:
            counts[name] += 1
            trace = destination.parent/'telemetry'/f'{name}_{counts[name]:02d}.json'
            trace.parent.mkdir(exist_ok=True)
            trace.write_text(json.dumps(frames, allow_nan=False)+'\n')
            row = {'env_name':name, 'trial':counts[name], 'status':int(result['status']),
                   'success':result['status']==0, 'n_steps':n_steps,
                   'step_dt_s':float(env.env.step_dt), 'task_time_s':n_steps*float(env.env.step_dt),
                   'telemetry_file':str(trace),
                   'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            with destination.open('a') as f: f.write(json.dumps(row)+'\n')
        return result
    namespace['sim'].get_curr_state = get_state
    namespace['get_sim_results'] = get_results
    namespace['simulate'] = simulate
    return main(app,args)

if __name__ == '__main__':
    source = Path(os.environ['DOM_OFFICIAL_SOURCE'])/'simulations/evaluate.py'
    text = source.read_text()
    needle = '    main(app_launcher.app, args)'
    assert text.count(needle) == 1
    text = text.replace(needle, '    _instrument(main, app_launcher.app, args, globals())')
    sys.path.insert(0,str(source.parent.parent))
    exec(compile(text,str(source),'exec'),
         {'__name__':'__main__','__file__':str(source),'_instrument':instrument})
