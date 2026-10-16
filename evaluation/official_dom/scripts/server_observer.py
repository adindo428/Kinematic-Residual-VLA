"""Execute upstream evaluation, adding only durable per-trial result recording."""
from collections import Counter
import datetime
import json
import os
from pathlib import Path
import sys

def instrument_and_run(main,app,args,namespace):
    original_get=namespace['get_sim_results'];original_simulate=namespace['simulate']
    counts=Counter();context={}
    destination=Path(os.environ['DOM_TRIAL_RESULTS'])
    if os.environ.get('DOM_EXPECTED_SIM'):
        import isaaclab
        from isaacsim.core.version import get_version
        module_path=Path(isaaclab.__file__).resolve()
        if os.environ.get('ISAACLAB_ROOT'):
            lab_root=Path(os.environ['ISAACLAB_ROOT']).resolve()
        else:
            lab_root=next(parent for parent in module_path.parents if (parent/'VERSION').is_file())
        runtime={'isaacsim':get_version()[0], 'isaaclab':(lab_root/'VERSION').read_text().strip(),
                 'python':sys.version, 'executable':sys.executable,
                 'isaaclab_source':str(lab_root),
                 'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
        destination.parent.mkdir(parents=True,exist_ok=True)
        (destination.parent/'simulator_runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
        if lab_root not in module_path.parents:
            raise RuntimeError('Isaac Lab imported from a different source than the declared runtime')
        if runtime['isaacsim']!=os.environ['DOM_EXPECTED_SIM'] or runtime['isaaclab']!=os.environ['DOM_EXPECTED_LAB']:
            raise RuntimeError(f'Unexpected benchmark simulator runtime: {runtime}')
    def observed_get(sim_cfg,env_cfg_file_path,obs_socket,act_socket):
        context['env_name']=Path(env_cfg_file_path).stem
        return original_get(sim_cfg,env_cfg_file_path,obs_socket,act_socket)
    def observed_simulate(env,obs_socket,act_socket,init_poses):
        result=original_simulate(env,obs_socket,act_socket,init_poses)
        steps=len(result['ee_path'])
        # Upstream skips extra single-frame reset artifacts; retain identical counting.
        if steps>1:
            name=context['env_name'];counts[name]+=1
            dt=float(env.env.step_dt)
            row={'env_name':name,'trial':counts[name],'status':int(result['status']),
                 'success':result['status']==0,'n_steps':steps,'step_dt_s':dt,
                 'task_time_s':steps*dt,
                 'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('a') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
        return result
    namespace['get_sim_results']=observed_get
    namespace['simulate']=observed_simulate
    return main(app,args)

if __name__=='__main__':
    source=Path(os.environ['DOM_OFFICIAL_SOURCE'])/'simulations/evaluate.py'
    text=source.read_text();needle='    main(app_launcher.app, args)'
    assert text.count(needle)==1
    # Preserve the upstream CLI/AppLauncher; replace only the top-level call.
    text=text.replace(needle,'    _instrument_and_run(main, app_launcher.app, args, globals())')
    sys.path.insert(0,str(source.parent.parent))
    namespace={'__name__':'__main__','__file__':str(source),'_instrument_and_run':instrument_and_run}
    exec(compile(text,str(source),'exec'),namespace)
