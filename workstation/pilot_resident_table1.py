#!/usr/bin/env python3
"""Prepare and run an excluded scene-reuse and resident-policy qualification."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal
import threading

import run_low_clutter_table1 as low
import run_krv_full_two_suites as krv

ROOT=krv.ROOT/'resident_pilot'
STOP=threading.Event()

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'plan.json').exists():return low.read(ROOT/'plan.json')
    models=low.read(krv.ROOT/'constant_resume/request.json')['models']
    tests={t['name']:t for t in low.read(krv.BASE/'protocol.json')['tests']}
    jobs=[]
    for d in ['CR','SR']:
        path=(krv.BASE/'test_lists'/f'{low.DIMS.index(d):02d}_{d}.txt').read_text().splitlines()[0]
        order=[*models,models[0]] if d=='CR' else models
        for i,m in enumerate(order):
            key=f'{d}-{i}-{m["key"]}';folder=ROOT/'blocks'/key
            folder.mkdir(parents=True)
            (folder/'case.txt').write_text(path+'\n')
            jobs.append({'id':key,'model':m['key'],'scene':path,'scene_sha256':tests[Path(path).stem]['config_sha256'],
                         'repeats':1,'repeat_offset':0,'folder':str(folder),'dimension':d,'scored':False})
    plan={'models':models,'jobs':jobs,'success_history':{},'all_trials_excluded_from_table1':True,
          'qualification':'same-noise native actions, latency with idle resident policies, repeated scene reset and checkpoint return'}
    low.write(ROOT/'plan.json',plan);return plan

def launch(plan_path,output):
    plan=low.read(plan_path);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    first=plan['jobs'][0];listing=output/'initial_case.txt';listing.write_text(first['scene']+'\n')
    catalog=low.read(low.ROOT/'catalog.json');krv.CATALOG=catalog
    server,_=krv.commands('original_dom',output,listing,1,0)
    server[server.index('-p')+1]=str(low.REPO/'workstation/resident_table1_server.py')
    k=server.index('--kit_args')+1
    directory=low.WORK/'kit/table1-resident'
    server[k]=server[k].replace('--portable-root='+str(low.WORK/'kit'),'--portable-root='+str(directory))
    for token,sub in [('omni_global_cache','cache'),('omni_global_logs','logs'),('omni_global_data','data')]:
        server[k]=server[k].replace(f'--/app/tokens/{token}='+str(low.WORK/sub/'omniverse'),f'--/app/tokens/{token}='+str(directory/sub))
    server[k]+=' --/app/extensions/fsWatcherEnabled=false'
    client=[str(low.WORK/'envs/train/bin/python'),'-u',str(low.REPO/'workstation/resident_table1_client.py'),
            '--plan',str(plan_path),'--output',str(output)]
    env=dict(os.environ,DOM_OFFICIAL_SOURCE=catalog['official_source'],DOM_RESIDENT_PLAN=str(plan_path),
             DOM_TRIAL_RESULTS=str(output/'unused.jsonl'),DOM_EXPECTED_SIM='4.5.0',DOM_EXPECTED_LAB='2.2.1',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    low.write(output/'commands.json',{'server':server,'client':client})
    processes=[]
    try:
        with (output/'server_shared.log').open('w') as sl,(output/'client_shared.log').open('w') as cl:
            processes.append(subprocess.Popen(server,env=dict(env,CUDA_VISIBLE_DEVICES='0'),stdout=sl,stderr=subprocess.STDOUT,start_new_session=True))
            processes.append(subprocess.Popen(client,env=dict(env,CUDA_VISIBLE_DEVICES='1'),stdout=cl,stderr=subprocess.STDOUT,start_new_session=True))
            while processes[1].poll() is None:
                if STOP.wait(2):raise InterruptedError('Resident runner stopped')
                if processes[0].poll() is not None:
                    status=low.read(output/'status.json') if (output/'status.json').exists() else {}
                    if status.get('stage')!='complete':raise RuntimeError('Resident simulator exited')
            if processes[1].returncode:raise RuntimeError('Resident policy client failed')
    finally:
        for p in reversed(processes):low.stop_group(p)

def verify(plan):
    import numpy as np
    scene_ids={};resets={};prompts={};counts=0
    for job in plan['jobs']:
        model=next(m for m in plan['models'] if m['key']==job['model'])
        folder=Path(job['folder'])
        low.runtime.verify_trial_folder(folder,Path(model['root']),folder/'case.txt',job['repeats'])
        records=low.runtime.read_records(folder);counts+=len(records)
        snapshots=[json.loads(l) for l in (folder/'reset_states.jsonl').read_text().splitlines()]
        assert len(snapshots)==job['repeats']
        for row in [json.loads(l) for l in (folder/'prompt_trace.jsonl').read_text().splitlines()]:
            key=(job['scene'],row['global_trial'])
            assert prompts.setdefault(key,row['text'])==row['text'],'Checkpoint prompts differ'
        for snapshot in snapshots:
            key=job['scene'];scene_ids.setdefault(key,set()).add(snapshot['scene_instance'])
            state=snapshot['state']
            if key in resets:
                baseline=resets[key]
                for name in ['robot_joint_pos','robot_joint_vel']:np.testing.assert_allclose(state[name],baseline[name],atol=1e-5,rtol=0)
                assert state['objects'].keys()==baseline['objects'].keys()
                for name in state['objects']:np.testing.assert_allclose(state['objects'][name],baseline['objects'][name],atol=1e-5,rtol=0)
            else:resets[key]=state
    assert all(len(ids)==1 for ids in scene_ids.values())
    proof=low.read(ROOT/'execution/resident_profile.json');assert proof['passed']
    result={'passed':True,'trials':counts,'all_trials_unscored':True,'scene_instances_reused':True,
            'identical_reset_states_within_1e5':True,'matched_prompts':True,'checkpoints_resident':len(plan['models']),
            'profile':proof,'reset_scene_ids':{k:list(v) for k,v in scene_ids.items()}}
    low.write(ROOT/'verification.json',result);return result

def main():
    signal.signal(signal.SIGTERM,lambda *_:STOP.set());signal.signal(signal.SIGINT,lambda *_:STOP.set())
    plan=prepare();launch(ROOT/'plan.json',ROOT/'execution');print(json.dumps(verify(plan)))

if __name__=='__main__':main()
