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
