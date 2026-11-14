#!/usr/bin/env python3
"""Queue full-data validation-best DOM+velocity after both KRV_Full suites."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time

import run_low_clutter_table1 as low
import run_krv_full_two_suites as krv
from monitor_table1_latency import log_samples, coverage
from capture_table1_runtime import stats
from table1_video_sampling import wrap_command
import table1_two_pass as passes

ROOT=low.WORK/'evaluations/domv-full-original-dom-20261106'
TRAIN=low.WORK/'training/domv-full-deltaai-20261106'
CHECKPOINT=TRAIN/'best-epoch003'
SOURCE=low.WORK/'training/krv100-full-20261010/source'
STOP=threading.Event()


def stamp():return datetime.now(timezone.utc).isoformat()


def status(state,**extra):
    low.write(ROOT/'status.json',{'state':state,'model':'DOMV_Full','selection':'validation-best, epoch 3',
        'updated_utc':stamp(),'checkpoint_trial_target':1800,**extra})


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    receipt=low.read(TRAIN/'transfer_receipt.json');assert receipt['verified']
    cfg=low.read(CHECKPOINT/'config.json')
    assert cfg['condition_on_object_velocity'] and not cfg['enable_residual_flow']
    assert cfg['n_obs_steps']==2 and cfg['n_action_steps']==cfg['chunk_size']==20
    base=low.read(krv.BASE/'protocol.json')
    for t in base['tests']:assert low.sha(t['path'])==t['config_sha256']
    protocol={**base,'benchmark':'DOMV_Full on original DOM suite','checkpoint_kind':'locally_trained',
        'method_label':'DOMV_Full (validation-best, epoch 3)','checkpoint_revision':'u000190656-95e1db0d',
        'trained_checkpoint':str(CHECKPOINT),'checkpoint_weights_sha256':receipt['files']['model.safetensors']['sha256'],
        'checkpoint_config_sha256':receipt['files']['config.json']['sha256'],'policy_source':str(SOURCE),
        'selection':'lowest recorded full-data validation reconstructed-action MSE; epoch 3, 0.1654408333767772',
        'training_dataset':low.read(TRAIN/'checkpoint_inventory.json')['dataset'],
        'server':str(low.REPO/'workstation/collection_matched_server.py'),
        'observation_difference':'Privileged simulator target pose/velocity; same collection-consistent selector as KRV_Full; original upstream prompts, physics, clutter and success rules.',
        'files_sha256':{str(p):low.sha(p) for p in [Path(__file__),low.REPO/'workstation/probe_domv_full_checkpoint.py',low.REPO/'workstation/inference_krvla.py',low.REPO/'workstation/krvla_policy_runtime.py',low.REPO/'workstation/collection_matched_server.py',low.REPO/'workstation/table1_two_pass.py',low.REPO/'workstation/table1_video_sampling.py',low.REPO/'workstation/sampled_video_table1_server.py',*SOURCE.rglob('*.py')]}}
    if (ROOT/'protocol.json').exists():assert low.read(ROOT/'protocol.json')==protocol
    else:low.write(ROOT/'protocol.json',protocol)
    (ROOT/'test_lists').mkdir(exist_ok=True)
    for f in (krv.BASE/'test_lists').glob('*.txt'):(ROOT/'test_lists'/f.name).write_text(f.read_text())
    low.write(ROOT/'queue_request.json',{'user_authorized':True,'requested_utc':stamp(),'checkpoint':str(CHECKPOINT),
        'selection':'validation-best epoch 3, not final epoch 6','scored_trials':1800,
        'dependencies':[str(krv.ROOT/'summary.json')],
        'requires_previous_service_inactive':True,'execution':'serial; original DOMV follows original KRV, then low-clutter KRV and DOMV',
        'followups':['KRV100_Full_low_clutter','DOMV_Full_low_clutter'],'other_models':'remain paused'})
    comparison()


def verify_files():
    p=low.read(ROOT/'protocol.json')
    for path,sha in p['files_sha256'].items():assert low.sha(path)==sha
    for t in p['tests']:assert low.sha(t['path'])==t['config_sha256']
    receipt=low.read(TRAIN/'transfer_receipt.json')
    for n,r in receipt['files'].items():
        f=CHECKPOINT/n;assert f.stat().st_size==r['bytes'] and f.stat().st_mtime_ns==r['mtime_ns']


def phase(dimension=None,low_clutter=False,pass_index=None):
    root=low.ROOT/'results/domv_full' if low_clutter else ROOT
    pilot=dimension is None
    listing=root/'test_lists'/('pilot.txt' if pilot else (f'{dimension}.txt' if low_clutter else f'{low.DIMS.index(dimension):02d}_{dimension}.txt'))
    folder=root/('pilot' if pilot else 'full/'+f'{low.DIMS.index(dimension):02d}_{dimension}');repeats=1 if pilot else 20
    offset=0
    if pass_index:
        assert not pilot
        repeats,offset=passes.block(root,pass_index)
        folder=root/'passes'/f'{pass_index:02d}'/'full'/f'{low.DIMS.index(dimension):02d}_{dimension}'
    verify=low.verify if low_clutter else low.runtime.verify_trial_folder
    if pass_index:verify=lambda folder,root,listing,repeats:passes.verify(folder,root,listing,repeats,offset,low_clutter)
    if (folder/'verification.json').exists():return verify(folder,root,listing,repeats)
    if folder.exists():raise FileExistsError('Partial phase preserved: '+str(folder))
    verify_files();folder.mkdir(parents=True)
    dummy=low.read(low.ROOT/'catalog.json')
    server,_=krv.production.commands(dummy,'krv100_full',folder,listing,repeats,0)
    if not low_clutter:server[server.index('-p')+1]=str(low.REPO/'workstation/collection_matched_server.py')
    if not low_clutter and '--tolerance' in server:
        i=server.index('--tolerance');del server[i:i+2]
    server=wrap_command(server,folder)
    client=[str(low.WORK/'envs/train/bin/python'),'-u',str(low.REPO/'workstation/inference_krvla.py'),
        '--source',str(SOURCE),'-p',str(CHECKPOINT),'-a','DOMV_Full','-r','euler','-d','-s',
        '--img_port','27600','--act_port','27601','-o',str(folder/'actions')]
    env=dict(os.environ,DOM_OFFICIAL_SOURCE=dummy['official_source'],DOM_TRIAL_RESULTS=str(folder/'trial_results.jsonl'),
        DOM_EXPECTED_SIM='4.5.0',DOM_EXPECTED_LAB='2.2.1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',DOM_REPEAT_OFFSET=str(offset))
    if pass_index==2:env['DOM_VIDEO_SELECTION_HISTORY']=str(root/'passes/01/full'/folder.name/'video_selection.jsonl')
    low.write(folder/'commands.json',{'server':server,'client':client})
    low.write(folder/'runtime.json',{'execution_mode':'isolated_serial','simulation_gpu':0,'inference_gpu':1,'streaming':True,'repeats':repeats,'pass_index':pass_index,'repeat_offset':offset})
    processes=[];changed=time.monotonic();count=-1
    try:
        with (folder/'server.log').open('w') as slog,(folder/'client.log').open('w') as clog:
            processes.append(subprocess.Popen(server,env=dict(env,CUDA_VISIBLE_DEVICES='0'),stdout=slog,stderr=subprocess.STDOUT,start_new_session=True))
            processes.append(subprocess.Popen(client,env=dict(env,CUDA_VISIBLE_DEVICES='1'),stdout=clog,stderr=subprocess.STDOUT,start_new_session=True))
            while processes[1].poll() is None:
                if STOP.is_set():raise InterruptedError('DOMV evaluation stopped')
                if processes[0].poll() is not None:raise RuntimeError('DOMV simulator exited; inspect '+str(folder))
                f=folder/'trial_results.jsonl';n=len(f.read_text().splitlines()) if f.exists() else 0
                if n!=count:
                    count=n;changed=time.monotonic();status('evaluating',dimension=dimension,phase='pilot' if pilot else 'full',completed_records_in_phase=n,phase_trial_target=(3 if low_clutter else 9) if pilot else 10*repeats,pass_index=pass_index)
                    if low_clutter:
                        row={'state':'evaluating','model':'domv_full','lane':0,'dimension':dimension,'phase':'pilot' if pilot else 'full',
                            'completed_records_in_phase':n,'phase_trial_target':3 if pilot else 10*repeats,'checkpoint_trial_target':1800,'image_port':27600,'pass_index':pass_index}
                        low.write(root/'suite_status.json',row)
                        low.status('evaluating',model='domv_full',dimension=dimension,execution_mode='isolated_serial',active_evaluations=[row],lanes=[row])
                        from render_table1_live_status import render
                        render(low.ROOT,True)
                if time.monotonic()-changed>1200:raise TimeoutError('No completed trial for 20 minutes')
                STOP.wait(2)
            if processes[1].returncode:raise RuntimeError('DOMV policy failed; inspect '+str(folder/'client.log'))
        v=verify(folder,root,listing,repeats)
        samples=log_samples(folder/'client.log')
        low.write(folder/'performance.json',{**stats([r['ms'] for r in samples]),'action_coverage':coverage(folder/'client.log'),'scored':not pilot})
        if low_clutter:low.publish(low.read(low.ROOT/'catalog.json'),upload=False)
        else:comparison(sync=True)
        return v
    except BaseException as error:
        low.write(folder/'failure.json',{'error':repr(error),'infrastructure_failure':True});raise
    finally:
        for p in reversed(processes):low.stop_group(p)


def comparison(sync=False):
    base=low.read(krv.BASE/'summary.json')['observed'];paper=low.read(krv.BASE/'protocol.json')['paper_reference']
    k=low.read(krv.ROOT/'summary.json')['observed'] if (krv.ROOT/'summary.json').exists() else {}
    observed={};records=[]
    for i,d in enumerate(low.DIMS):
        f=ROOT/'full'/f'{i:02d}_{d}'/'verification.json'
        if f.exists():observed[d]=low.read(f)['success_percent'];records+=low.runtime.read_records(f.parent)
    if len(observed)==9:
        observed.update(Success=100*statistics.mean(r['success'] for r in records),Time=statistics.mean(r['task_time_s'] for r in records))
        low.write(ROOT/'summary.json',{'complete':True,'verified_trials':1800,'observed':observed,'protocol':low.read(ROOT/'protocol.json')})
    keys=[*low.DIMS,'Success','Time']
    md='# Original full DOM suite comparison\n\n90 original configurations × 20 repeats = 1,800 scored trials per checkpoint. Historical full-data/continuation models are separate from the primary 20k comparison. DOMV_Full and KRV_Full receive privileged simulator pose/velocity; training histories differ. Released DynamicVLA is the archived original-suite local result.\n\n| Method | '+' | '.join(keys)+' |\n|---|'+'---:|'*len(keys)+'\n'
    for name,v in [('DynamicVLA paper',paper),('Released DynamicVLA (local original suite)',base),('KRV100_Full (validation-best epoch 3)',k),('DOMV_Full (validation-best, epoch 3)',observed)]:
        md+='| '+name+' | '+' | '.join(f'{v[x]:.2f}' if x in v else 'pending' for x in keys)+' |\n'
    (ROOT/'table1_comparison.md').write_text(md)
    if sync:(krv.ROOT/'table1_comparison.md').write_text(md)


def low_clutter_followups():
    import shutil
    from render_table1_live_status import render
    catalog=low.read(low.ROOT/'catalog.json');low.validate(catalog)
    low.ORDER=list(catalog['order']);krv.CATALOG=catalog;krv.MODE='serial'
    reference=krv.CONTROL/'qualification.json'
    if reference.exists():krv.REFERENCES.update(low.read(reference)['serial'])
    status('running_low_clutter_krv_followup')
    root=low.ROOT/'results/krv100_full'
    krv.run('low_clutter',root/'pilot',root/'test_lists/pilot.txt',1,0,None,False)
    for pass_index in [1,2]:
        repeats,_=passes.block(root,pass_index)
        for i,d in enumerate(low.DIMS):
            listing=root/'test_lists'/f'{d}.txt'
            krv.run('low_clutter',root/'passes'/f'{pass_index:02d}'/'full'/f'{i:02d}_{d}',listing,repeats,0,d,True,pass_index)
            if pass_index==2:passes.merge(root,d,i,listing,True)
            low.publish(catalog,upload=False)
    passes.summarize(root,low.DIMS,catalog['models']['krv100_full'])
    receipt=low.read(TRAIN/'transfer_receipt.json');inventory=low.read(TRAIN/'checkpoint_inventory.json')
    f=CHECKPOINT/'model.safetensors';c=CHECKPOINT/'config.json'
    entry={'label':'DOMV_Full (low-clutter, validation-best epoch 3)','client_kind':'native','kr':False,
        'method':'dom_full_velocity','source':str(SOURCE),'checkpoint':str(CHECKPOINT),
        'weight_path':str(f),'weight_bytes':f.stat().st_size,'weight_mtime_ns':f.stat().st_mtime_ns,
        'checkpoint_sha256':receipt['files']['model.safetensors']['sha256'],
        'config_sha256':receipt['files']['config.json']['sha256'],
        'progress':inventory['pointers']['best.json']['metadata']['progress'],
        'historical_user_exception':True,'primary_20k_cohort':False,'training_scope':'full_dataset_only','full_dataset_provenance_verified':True,
        'selection':'validation-best full-data epoch 3; explicitly authorized low-clutter follow-up'}
    catalog['models']['domv_full']=entry
    if 'domv_full' not in catalog['order']:catalog['order'].insert(catalog['order'].index('krv100_full')+1,'domv_full')
    if 'domv_full' not in catalog['historical_order']:catalog['historical_order'].append('domv_full')
    root=low.ROOT/'results/domv_full';root.mkdir(parents=True,exist_ok=True)
    shutil.copytree(low.ROOT/'test_lists',root/'test_lists',dirs_exist_ok=True)
    low.write(root/'protocol.json',{**low.read(low.ROOT/'protocol.json'),'checkpoint':entry})
    low.write(root/'two_pass_plan.json',passes.plan(ROOT))
    low.write(root/'user_request.json',{'explicit_user_request':True,'order':'after original DOM KRV, original DOM DOMV, low-clutter KRV','checkpoint':entry})
    low.write(low.ROOT/'catalog.json',catalog);low.validate(catalog)
    low.ORDER=list(catalog['order']);status('running_low_clutter_domv_followup')
    phase(low_clutter=True)
    for pass_index in [1,2]:
        for i,d in enumerate(low.DIMS):
            phase(d,low_clutter=True,pass_index=pass_index)
            if pass_index==2:passes.merge(root,d,i,root/'test_lists'/f'{d}.txt',True)
    passes.summarize(root,low.DIMS,entry);low.publish(catalog,upload=False)
    low.status('paused_by_user',requested_historical_suites_complete=True,active_evaluations=[],lanes=[])
    render(low.ROOT,False)
    low.write(krv.CONTROL/'followup_complete.json',{'state':'complete','updated_utc':stamp(),
        'order':['KRV_Full original','DOMV_Full original','KRV100_Full low-clutter','DOMV_Full low-clutter'],
        'other_models':'remain paused'})


def main():
    signal.signal(signal.SIGTERM,lambda *_:STOP.set());signal.signal(signal.SIGINT,lambda *_:STOP.set())
    prepare();status('queued_after_krv_original_suite')
    while not STOP.is_set():
        ready=all(p.exists() and low.read(p).get('complete') and low.read(p).get('verified_trials')==1800 for p in [krv.ROOT/'summary.json'])
        active=subprocess.run(['systemctl','--user','is-active','krvla-low-clutter-full-table1-20261102.service'],capture_output=True,text=True).stdout.strip()
        if ready and active not in ['active','activating','deactivating','reloading']:break
        STOP.wait(30)
    if STOP.is_set():return
    with (low.ROOT/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Validate the transferred weights before acquiring any GPU memory.
        assert low.sha(CHECKPOINT/'model.safetensors')==low.read(TRAIN/'transfer_receipt.json')['files']['model.safetensors']['sha256']
        verify_files();comparison(sync=True)
        probe=[str(low.WORK/'envs/train/bin/python'),'-u',str(low.REPO/'workstation/probe_domv_full_checkpoint.py'),
            '--source',str(SOURCE),'--checkpoint',str(CHECKPOINT),'--output',str(ROOT/'policy_probe.json')]
        status('probing')
        with (ROOT/'probe.log').open('w') as log:
            subprocess.run(probe,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
        phase()
        for pass_index in [1,2]:
            for i,d in enumerate(low.DIMS):
                phase(d,pass_index=pass_index)
                if pass_index==2:passes.merge(ROOT,d,i,ROOT/'test_lists'/f'{i:02d}_{d}.txt',False)
        comparison(sync=True);status('complete',verified_trials=1800)
        low_clutter_followups()
        status('all_four_requested_suites_complete',verified_trials=1800)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        ROOT.mkdir(parents=True,exist_ok=True);status('failed',error=repr(error));raise
