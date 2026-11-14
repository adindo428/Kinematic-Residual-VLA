"""One Isaac application; checkpoint blocks reuse its cached scene with resets."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import uuid

import sampled_video_table1_server as wrapper
from causal_gate_diagnostic_server import patch_source as observation_patch

CURRENT={};COUNTS=Counter();SESSION=uuid.uuid4().hex;RUNTIME={};HANDLER=None
PROMPTS={};LAST_PROMPT=None


def begin_block(action,args):
    global CURRENT,COUNTS,HANDLER
    plan=json.loads(Path(os.environ['DOM_RESIDENT_PLAN']).read_text())
    for scene,prompts in plan.get('saved_prompts',{}).items():
        PROMPTS.setdefault(scene,{}).update(prompts)
    job=next(j for j in plan['jobs'] if j['id']==action['resident_job'])
    assert job['model']==action['vla'] and 1<=job['repeats']<=10
    assert hashlib.sha256(Path(job['scene']).read_bytes()).hexdigest()==job['scene_sha256']
    CURRENT=job;COUNTS=Counter()
    folder=Path(job['folder']);folder.mkdir(parents=True,exist_ok=True)
    os.environ['DOM_TRIAL_RESULTS']=str(folder/'trial_results.jsonl')
    os.environ['DOM_REPEAT_OFFSET']=str(job.get('repeat_offset',0))
    args.output_dir=str(folder/'videos')
    import table1_video_sampling as video
    video._HISTORY_LOADED=True
    video._FIRST_SUCCESS=set(plan.get('success_history',{}).get(job['model'],[]))
    logger=logging.getLogger()
    if HANDLER is not None:logger.removeHandler(HANDLER);HANDLER.close()
    HANDLER=logging.FileHandler(folder/'server.log')
    HANDLER.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s %(message)s'));logger.addHandler(HANDLER)
    (folder/'simulator_runtime.json').write_text(json.dumps({**RUNTIME,'resident_session':SESSION,'server_pid':os.getpid()},indent=2)+'\n')
    (folder/'resident_block.json').write_text(json.dumps({'job':job,'session':SESSION,'server_pid':os.getpid()},indent=2)+'\n')
    return job


def patch_source(text):
    text=observation_patch(text)
    old='        vla_name = action["vla"]'
    assert text.count(old)==1
    text=text.replace(old,old+'\n        _resident_job = __import__("resident_table1_server").begin_block(action, args)\n        test_envs = [_resident_job["scene"]]\n        args.n_tests = _resident_job["repeats"]')
    old='        if action is None or "vla" not in action:'
    assert text.count(old)==1
    text=text.replace(old,'        if action is not None and action.get("resident_stop"):\n            return\n'+old)
    old='            logging.debug("No VLA clients connected.")\n            time.sleep(10)'
    assert text.count(old)==1
    text=text.replace(old,'            time.sleep(.05)')
    return text


def instrument(main,app,args,namespace):
    global RUNTIME
    import isaaclab
    from isaacsim.core.version import get_version
    lab=Path(os.environ['ISAACLAB_ROOT'])
    RUNTIME={'isaacsim':get_version()[0],'isaaclab':(lab/'VERSION').read_text().strip(),
             'python':sys.version,'executable':sys.executable,'isaaclab_source':str(lab),
             'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
    assert RUNTIME['isaacsim']=='4.5.0' and RUNTIME['isaaclab']=='2.2.1'
    assert lab.resolve() in Path(isaaclab.__file__).resolve().parents
    original_get=namespace['get_sim_results'];original_simulate=namespace['simulate'];context={}
    original_generator=namespace['InstructionGenerator']
    class MatchedInstructions:
        @staticmethod
        def generate_instruction(metadata):
            global LAST_PROMPT
            scene=CURRENT['scene'];name=Path(scene).stem
            trial=str(COUNTS[name]+CURRENT.get('repeat_offset',0)+1)
            prompts=PROMPTS.setdefault(scene,{})
            if trial not in prompts:
                # Use the released generator, then cache its actual output so
                # every checkpoint sees the same wording for this repeat.
                prompts[trial]=original_generator.generate_instruction(metadata)
            LAST_PROMPT=prompts[trial]
            return LAST_PROMPT
    namespace['InstructionGenerator']=MatchedInstructions
    def get(*a,**kw):
        context['env_name']=Path(a[1]).stem
        return original_get(*a,**kw)
    def simulate(env,*a,**kw):
        name=context['env_name'];folder=Path(CURRENT['folder'])
        # Snapshot reset state before the unchanged upstream scored loop.
        scene=env.unwrapped.scene
        state={'robot_joint_pos':scene['robot'].data.joint_pos.detach().cpu().tolist(),
               'robot_joint_vel':scene['robot'].data.joint_vel.detach().cpu().tolist(),
               'objects':{k:v.data.root_state_w.detach().cpu().tolist() for k,v in scene.rigid_objects.items()}}
        result=original_simulate(env,*a,**kw);steps=len(result['ee_path'])
        if steps>1:
            COUNTS[name]+=1;dt=float(env.env.step_dt)
            with (folder/'reset_states.jsonl').open('a') as f:
                f.write(json.dumps({'env_name':name,'trial':COUNTS[name],'scene_instance':id(env.unwrapped),
                                    'scene_path':CURRENT['scene'],'state':state,'session':SESSION})+'\n')
            with (folder/'prompt_trace.jsonl').open('a') as f:
                f.write(json.dumps({'env_name':name,'trial':COUNTS[name],
                                    'global_trial':COUNTS[name]+CURRENT.get('repeat_offset',0),'text':LAST_PROMPT})+'\n')
            row={'env_name':name,'trial':COUNTS[name],'status':int(result['status']),'success':result['status']==0,
                 'n_steps':steps,'step_dt_s':dt,'task_time_s':steps*dt,'recorded_at_utc':datetime.now(timezone.utc).isoformat()}
            with (folder/'trial_results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
        return result
    namespace['get_sim_results']=get;namespace['simulate']=simulate
    return main(app,args)


def main():
    sys.modules['resident_table1_server']=sys.modules[__name__]
    scripts=Path(__file__).resolve().parents[1]/'evaluation/official_dom/scripts'
    sys.path.insert(0,str(scripts))
    import server_observer
    server_observer.instrument_and_run=instrument
    wrapper.patch_source=patch_source
    wrapper.main()


if __name__=='__main__':main()
