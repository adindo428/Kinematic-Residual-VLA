#!/usr/bin/env python3
"""Resume constant-gate epoch 3 with qualified, monitored column parallelism."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse
import fcntl
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import threading
import time

import run_low_clutter_table1 as low
import run_krv_full_two_suites as krv
import table1_two_pass as passes
from monitor_table1_latency import log_samples,coverage
from capture_table1_runtime import stats
from table1_resume_records import compose

ROOT=krv.ROOT/'constant_resume'
STOP=threading.Event();CANCEL=threading.Event();LOCK=threading.RLock()
STATE={};ACTIVE={};SELECTED={};REFERENCE=None;PARALLEL=False

class ParallelDrop(RuntimeError):pass
def stamp():return datetime.now(timezone.utc).isoformat()

def publish(**kw):
    with LOCK:
        STATE.update(kw)
        low.write(ROOT/'status.json',{**STATE,'updated_utc':stamp(),'parallel':PARALLEL,'lanes':list(ACTIVE.values())})
        from render_constant_table1_resume import render
        render()

def heartbeat():
    while not STOP.wait(5):publish()

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'request.json').exists():return low.read(ROOT/'request.json')
    old=low.read(krv.ROOT/'gated_queue/request.json')
    first={**old['models'][0],'key':'krv_best_constant','label':'KRV100_Full epoch 3 (validation-best, constant gate)',
           'root':str(krv.ROOT),'checkpoint':old['models'][0]['parent_checkpoint'],'gated':False}
    first['config_sha256']=low.sha(Path(first['checkpoint'])/'config.json')
    assert low.read(Path(first['checkpoint'])/'config.json')['residual_phase_gate_mode']=='constant'
    models=[first,*[{**m,'gated':m['kr']} for m in old['models'][1:]]]
    tasks={};prefixes={}
    for m in models:
        root=Path(m['root']);items=[]
        for i,d in enumerate(low.DIMS):
            folder=root/'passes/01/full'/f'{i:02d}_{d}';listing=root/'test_lists'/f'{i:02d}_{d}.txt'
            if (folder/'verification.json').exists():
                passes.verify(folder,root,listing,10,0,False);continue
            lines=listing.read_text().splitlines()
            if m['key']=='krv_best_constant' and d=='DA' and folder.exists():
                rows=low.runtime.read_records(folder);assert len(rows)==14
                names=[Path(l).stem for l in lines]
                assert [sum(r['env_name']==n for r in rows) for n in names]==[10,4,0,0,0,0,0,0,0,0]
                from report import verify_records
                tests={t['name']:t for t in low.read(root/'protocol.json')['tests']}
                for name,n in [(names[0],10),(names[1],4)]:
                    verify_records([r for r in rows if r['env_name']==name],[name],n,{name:tests[name]['horizon_seconds']})
                archive=root/'preserved_serial/01_DA-prefix14';archive.parent.mkdir(exist_ok=True)
                assert not archive.exists();folder.rename(archive)
                prefixes[d]={'folder':str(archive),'prefix':True,'counts':{names[0]:10,names[1]:4},
                             'records_sha256':low.sha(archive/'trial_results.jsonl')}
                for key,ls,n,offset in [('DA_tail',lines[1:2],6,4),('DA_remaining',lines[2:],10,0)]:
                    target=ROOT/(key+'.txt');target.write_text('\n'.join(ls)+'\n')
                    items.append({'key':key,'dimension':d,'listing':str(target),'repeats':n,'repeat_offset':offset,'target':len(ls)*n})
            else:
                assert not folder.exists(),'Preserve unexpected partial output: '+str(folder)
                items.append({'key':d,'dimension':d,'listing':str(listing),'repeats':10,'repeat_offset':0,'target':100})
        tasks[m['key']]=items
    plan={'enabled':True,'user_authorized':True,'requested_utc':stamp(),'models':models,'tasks':tasks,'prefixes':prefixes,
          'saved_constant_trials':114,'first_pass_trials_per_checkpoint':900,'trials_per_column':100,
          'remaining_ten_repeats_deferred':True,'minimum_latency_retention':.95,
          'gated_epoch3_paused':True,'qualification_unscored':True,
          'resume_caveat':'Saved completed records retained; resumed blocks use fresh processes and do not claim bitwise-continuous prompt/noise replay.'}
    low.write(ROOT/'request.json',plan)
    low.write(ROOT/'task_selection.json',{})
    return plan

def execute(model,task,folder,lane,scored):
    folder.mkdir(parents=True,exist_ok=False)
    catalog=low.read(low.ROOT/'catalog.json');krv.CATALOG=catalog
    server,_=krv.commands('original_dom',folder,Path(task['listing']),task['repeats'],lane)
    # Each Kit process owns its writable state; extension hot reload is unused
    # during frozen evaluation and exhausted the user's file-watch quota.
    lane_root=low.WORK/'kit'/f'table1-resume-lane{lane}'
    lane_root.mkdir(parents=True,exist_ok=True)
    k=server.index('--kit_args')+1
    server[k]=server[k].replace('--portable-root='+str(low.WORK/'kit'),'--portable-root='+str(lane_root))
    for token,directory in [('omni_global_cache','cache'),('omni_global_logs','logs'),('omni_global_data','data')]:
        server[k]=server[k].replace(f'--/app/tokens/{token}='+str(low.WORK/directory/'omniverse'),
                                    f'--/app/tokens/{token}='+str(lane_root/directory))
    server[k]+=' --/app/extensions/fsWatcherEnabled=false'
    if model['gated']:server[server.index('-p')+1]=str(low.REPO/'workstation/causal_gate_diagnostic_server.py')
    script='causal_gate_diagnostic_client.py' if model['gated'] else 'inference_krvla.py'
    port=27600+lane*10
    client=[str(low.WORK/'envs/train/bin/python'),'-u',str(low.REPO/'workstation'/script),'--source',model['source'],
            '-p',model['checkpoint'],'-a',model['key'],'-r','euler','-d','-s','--img_port',str(port),'--act_port',str(port+1),'-o',str(folder/'actions')]
    env=dict(os.environ,DOM_OFFICIAL_SOURCE=catalog['official_source'],DOM_TRIAL_RESULTS=str(folder/'trial_results.jsonl'),
             DOM_GATE_TRACE=str(folder/'gate_trace.jsonl'),DOM_GATE_WORKER_TRACE=str(folder/'gate_worker_trace.jsonl'),
             DOM_RESIDUAL_TARGET_LEAD_S='0',DOM_REPEAT_OFFSET=str(task.get('repeat_offset',0)),
             DOM_EXPECTED_SIM='4.5.0',DOM_EXPECTED_LAB='2.2.1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    env.pop('DOM_VIDEO_SELECTION_HISTORY',None)
    low.write(folder/'commands.json',{'server':server,'client':client,'scored':scored})
    low.write(folder/'runtime.json',{'lane':lane,'parallel':PARALLEL,'scored':scored,'checkpoint_sha256':model['weight_sha256'],
              'gate':'causal_rules' if model['gated'] else 'constant','repeat_offset':task.get('repeat_offset',0),'target_lead_s':0,
              'kit_portable_root':str(lane_root),'extension_file_watching':False})
    processes=[];last=-1;changed=time.monotonic();health=0;breaches=0;started=time.monotonic()
    try:
        with (folder/'server.log').open('w') as sl,(folder/'client.log').open('w') as cl:
            processes.append(subprocess.Popen(server,env=dict(env,CUDA_VISIBLE_DEVICES='0'),stdout=sl,stderr=subprocess.STDOUT,start_new_session=True))
            processes.append(subprocess.Popen(client,env=dict(env,CUDA_VISIBLE_DEVICES='1'),stdout=cl,stderr=subprocess.STDOUT,start_new_session=True))
            while processes[1].poll() is None:
                if STOP.is_set():raise InterruptedError('Resume controller stopped')
                if CANCEL.is_set():raise ParallelDrop('Parallel lane cancelled by latency guard')
                if processes[0].poll() is not None:raise RuntimeError('Simulator exited: '+str(folder))
                p=folder/'trial_results.jsonl';n=len(p.read_text().splitlines()) if p.exists() else 0
                if n!=last:last=n;changed=time.monotonic()
                with LOCK:ACTIVE[lane]={'lane':lane,'model':model['key'],'dimension':task.get('dimension'),'task':task['key'],
                                        'records':n,'target':task['target'],'scored':scored,'folder':str(folder)}
                if scored and time.monotonic()-health>=15:
                    health=time.monotonic();samples=log_samples(folder/'client.log')[-100:];current=stats([r['ms'] for r in samples])
                    ok=not PARALLEL or len(samples)<50 or (current['p50_ms']<=REFERENCE['p50_ms']/.95 and current['p95_ms']<=REFERENCE['p95_ms']/.95)
                    breaches=0 if ok else breaches+1
                    low.write(folder/'latency_health.json',{'updated_utc':stamp(),'rolling':current,'reference':REFERENCE,
                              'meets_95pct_retention':ok,'consecutive_breaches':breaches,'parallel':PARALLEL})
                    if PARALLEL and breaches>=3:
                        CANCEL.set();raise ParallelDrop('95% latency retention failed for three rolling checks')
                if time.monotonic()-changed>1200:raise TimeoutError('No trial progress for 20 minutes')
                STOP.wait(2)
            if CANCEL.is_set():raise ParallelDrop('Parallel guard triggered before verification')
            if processes[1].returncode:raise RuntimeError('Policy exited: '+str(folder/'client.log'))
        verified=low.runtime.verify_trial_folder(folder,Path(model['root']),Path(task['listing']),task['repeats'])
        samples=log_samples(folder/'client.log');stale=[s['stale'] for s in samples if s['stale'] is not None]
        result={**stats([r['ms'] for r in samples]),'action_coverage':coverage(folder/'client.log'),
                'expired_fraction':sum(x>=20 for x in stale)/len(stale) if stale else 0,
                'wall_seconds':time.monotonic()-started,'verified':verified,'scored':scored}
        low.write(folder/'performance.json',result);return result
    except BaseException as e:
        low.write(folder/'attempt_failure.json',{'error':repr(e),'excluded_from_scoring':True,'records_retained':True})
        raise
    finally:
        for p in reversed(processes):low.stop_group(p)
        with LOCK:ACTIVE.pop(lane,None)

def qualify(model):
    global PARALLEL,REFERENCE
    parent=ROOT/'qualification'/model['key'];receipt=parent/'decision.json'
    if receipt.exists():
        r=low.read(receipt);REFERENCE=r['serial'];PARALLEL=r['parallel_accepted'];return
    parent.mkdir(parents=True,exist_ok=True)
    attempt=1
    while (parent/f'attempt{attempt:02d}').exists():attempt+=1
    root=parent/f'attempt{attempt:02d}';root.mkdir()
    listing=root/'cases.txt';listing.write_text('\n'.join((Path(model['root'])/'test_lists'/f'{low.DIMS.index(d):02d}_{d}.txt').read_text().splitlines()[0] for d in ['CR','LS','SR'])+'\n')
    task={'key':'qualification','listing':str(listing),'repeats':1,'target':3}
    PARALLEL=False;publish(stage='qualifying_serial',model=model['key'])
    serial=execute(model,task,root/'serial',0,False)
    PARALLEL=True;publish(stage='qualifying_parallel',model=model['key'])
    parallel=[None,None];parallel_error=None
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(execute,model,task,root/f'parallel_lane{i}',i,False):i for i in [0,1]}
        try:
            for future in as_completed(futures):parallel[futures[future]]=future.result()
        except Exception as error:
            CANCEL.set();parallel_error=repr(error)
    CANCEL.clear()
    if STOP.is_set():raise InterruptedError('Qualification stopped')
    if parallel_error:
        result={'serial':serial,'parallel':parallel,'parallel_error':parallel_error,'parallel_accepted':False,
                'excluded_from_table1':True,'fallback':'serial','minimum_retention':.95}
        low.write(receipt,result);REFERENCE=serial;PARALLEL=False;return
    checks=[{'p50_ratio':p['p50_ms']/serial['p50_ms'],'p95_ratio':p['p95_ms']/serial['p95_ms'],
             'coverage_retention':p['action_coverage']/serial['action_coverage'],
             'expired_fraction_increase':p['expired_fraction']-serial['expired_fraction']} for p in parallel]
    speedup=2*serial['wall_seconds']/max(p['wall_seconds'] for p in parallel)
    accepted=serial['samples']>=50 and all(p['samples']>=50 for p in parallel) and speedup>1.1 and all(
        c['p50_ratio']<=1/.95 and c['p95_ratio']<=1/.95 and c['coverage_retention']>=.95 and c['expired_fraction_increase']<=.02 for c in checks)
    result={'serial':serial,'parallel':parallel,'checks':checks,'aggregate_wall_speedup':speedup,
            'parallel_accepted':accepted,'excluded_from_table1':True,'minimum_retention':.95,
            'limitation':'Runtime qualification does not prove identical stochastic closed-loop outcomes.'}
    low.write(receipt,result);REFERENCE=serial;PARALLEL=accepted

def aggregate(plan,model,dimension):
    root=Path(model['root']);target=root/'passes/01/full'/f'{low.DIMS.index(dimension):02d}_{dimension}'
    listing=root/'test_lists'/f'{low.DIMS.index(dimension):02d}_{dimension}.txt'
    if (target/'verification.json').exists():return
    tasks=[t for t in plan['tasks'][model['key']] if t['dimension']==dimension]
    if not all(model['key']+'/'+t['key'] in SELECTED for t in tasks):return
    sources=[]
    if model['key']=='krv_best_constant' and dimension in plan['prefixes']:sources.append(plan['prefixes'][dimension])
    for t in tasks:
        sources.append({'folder':SELECTED[model['key']+'/'+t['key']],'listing':t['listing'],
                        'repeats':t['repeats'],'repeat_offset':t['repeat_offset']})
    compose(target,root,listing,10,sources);passes.verify(target,root,listing,10,0,False)

def run_tasks(plan,model):
    global PARALLEL
    tasks=plan['tasks'][model['key']]
    def wave(lanes):
        pending=queue.Queue()
        for t in tasks:
            if model['key']+'/'+t['key'] not in SELECTED:pending.put(t)
        def worker(lane):
            while not STOP.is_set() and not CANCEL.is_set():
                try:t=pending.get_nowait()
                except queue.Empty:return
                base=Path(model['root'])/'resume_tasks'/t['key'];base.mkdir(parents=True,exist_ok=True)
                index=1
                while (base/f'attempt{index:02d}').exists():index+=1
                folder=base/f'attempt{index:02d}'
                try:
                    execute(model,t,folder,lane,True)
                    with LOCK:
                        SELECTED[model['key']+'/'+t['key']]=str(folder)
                        low.write(ROOT/'task_selection.json',SELECTED)
                        aggregate(plan,model,t['dimension'])
                except ParallelDrop:return
                finally:pending.task_done()
        with ThreadPoolExecutor(max_workers=lanes) as pool:
            futures=[pool.submit(worker,i) for i in range(lanes)]
            for f in futures:f.result()
    publish(stage='scoring',model=model['key'])
    wave(2 if PARALLEL else 1)
    if CANCEL.is_set():
        PARALLEL=False;CANCEL.clear()
        low.write(ROOT/'serial_fallback.json',{'at':stamp(),'model':model['key'],'reason':'Rolling 95% latency guard',
                  'excluded_attempts_retained':True,'unverified_attempts_rerun_serially':True})
        publish(stage='scoring_serial_fallback');wave(1)
    if STOP.is_set():raise InterruptedError('Resume stopped')
    root=Path(model['root'])
    for i,d in enumerate(low.DIMS):passes.verify(root/'passes/01/full'/f'{i:02d}_{d}',root,root/'test_lists'/f'{i:02d}_{d}.txt',10,0,False)
    low.write(root/'first_pass_summary.json',{'first_pass_complete':True,'verified_trials':900,'checkpoint':model,
              'remaining_trials_deferred':900,'updated_utc':stamp()})

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    plan=prepare()
    if args.prepare_only:return
    signal.signal(signal.SIGTERM,lambda *_:STOP.set());signal.signal(signal.SIGINT,lambda *_:STOP.set())
    global SELECTED
    SELECTED=low.read(ROOT/'task_selection.json')
    with (low.ROOT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        threading.Thread(target=heartbeat,daemon=True).start()
        try:
            for m in plan['models']:
                assert low.sha(Path(m['checkpoint'])/'config.json')==m['config_sha256']
                assert low.sha(Path(m['checkpoint'])/'model.safetensors')==m['weight_sha256']
                qualify(m);run_tasks(plan,m)
            publish(stage='first_pass_queue_complete')
        except BaseException as e:publish(stage='paused' if STOP.is_set() else 'failed',error=repr(e));raise
        finally:STOP.set()

if __name__=='__main__':main()
