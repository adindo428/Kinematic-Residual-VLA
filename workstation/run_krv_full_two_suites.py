#!/usr/bin/env python3
"""Evaluate the verified full-data KRV checkpoint on two preserved DOM suites."""
from concurrent.futures import ThreadPoolExecutor
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
import run_parallel_low_clutter_table1 as production
from monitor_table1_latency import log_samples, coverage
from capture_table1_runtime import stats
from render_table1_live_status import render
from table1_video_sampling import wrap_command
import table1_two_pass as passes

BASE = low.WORK/'results/dom-paper45-a6000-20261010-02'
ROOT = low.WORK/'evaluations/krv-full-original-dom-20261106'
CONTROL = low.ROOT/'krv_full_two_suites'
STOP = threading.Event()
SERIAL = threading.Event()
LOCK = threading.Lock()
ACTIVE = {}
CATALOG = None
MODE = 'serial'
REFERENCES = {}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def publish_status():
    with LOCK:
        active = list(ACTIVE.values())
        low_active = [r for r in active if r['suite']=='low_clutter']
        row = low_active[0] if low_active else {}
        low.status('evaluating' if active else 'preparing', **{k:v for k,v in row.items() if k!='state'},
            execution_mode=MODE, concurrency=len(active), requested_concurrency=2 if MODE=='parallel_qualified_95pct' else 1,
            active_evaluations=low_active, lanes=low_active, only_enabled_checkpoint='krv100_full',
            other_model_evaluations_paused=True, full_dom_status=str(ROOT/'status.json'))
        low.write(CONTROL/'status.json', {'state':'evaluating' if active else 'preparing',
            'updated_utc':stamp(),'execution_mode':MODE,'active_evaluations':active,
            'parallel_retention_min':.95,'serial_fallback_requested':SERIAL.is_set()})
        render(low.ROOT, True)


def prepare():
    global CATALOG
    CATALOG=low.read(low.ROOT/'catalog.json');low.validate(CATALOG)
    low.ORDER=list(CATALOG['order'])
    assert CATALOG['enabled_models']==['krv100_full']
    entry=CATALOG['models']['krv100_full']
    assert entry['training_scope']=='full_dataset_only' and entry['full_dataset_provenance_verified']
    request=low.read(low.ROOT/'full_data_checkpoint_request.json')
    assert entry['checkpoint_sha256']==request['checkpoint_sha256']
    assert low.sha(entry['weight_path'])==entry['checkpoint_sha256']
    assert low.sha(Path(entry['checkpoint'])/'config.json')==entry['config_sha256']
    # Reuse the previously verified weight SHA/size/mtime receipt, without a copy.
    ROOT.mkdir(parents=True,exist_ok=True);CONTROL.mkdir(parents=True,exist_ok=True)
    baseline=low.read(BASE/'protocol.json')
    for test in baseline['tests']:
        assert low.sha(test['path'])==test['config_sha256']
    assert len(baseline['tests'])==90
    protocol={**baseline,'benchmark':'KRV_Full on original DOM physical configurations',
        'checkpoint_kind':'locally_trained','method_label':'KRV100_Full (validation-best epoch 3)',
        'trained_checkpoint':entry['checkpoint'],'checkpoint_revision':entry['generation'],
        'checkpoint_weights_sha256':entry['checkpoint_sha256'],'checkpoint_config_sha256':entry['config_sha256'],
        'policy_source':entry['source'],'baseline_protocol_sha256':low.sha(BASE/'protocol.json'),
        'baseline_summary_sha256':low.sha(BASE/'summary.json'),
        'auxiliary_observation_server':str(low.REPO/'workstation/collection_matched_server.py'),
        'auxiliary_observation_server_sha256':low.sha(low.REPO/'workstation/collection_matched_server.py'),
        'observation_difference':'KRV receives privileged target pose/velocity; collection-consistent target selection. Original configs, upstream prompts, physics and termination retained.',
        'parallel_rule':{'min_throughput_retention':.95,'p50_and_p95_latency_ratio_max':1/.95,
            'action_coverage_retention_min':.95,'qualification_trials_scored':False}}
    if (ROOT/'protocol.json').exists():
        assert low.read(ROOT/'protocol.json')==protocol
    else:low.write(ROOT/'protocol.json',protocol)
    (ROOT/'test_lists').mkdir(exist_ok=True)
    for f in (BASE/'test_lists').glob('*.txt'):
        (ROOT/'test_lists'/f.name).write_text(f.read_text())
    low.write(CONTROL/'request.json',{'user_authorized':True,'created_utc':stamp(),
        'checkpoint':entry,'suites':{'low_clutter':str(low.ROOT/'results/krv100_full'),'original_dom':str(ROOT)},
        'scored_trials_per_suite':1800,'other_models':'paused; no automatic resume',
        'threshold_interpretation':'At least 95% of isolated inference throughput: latency ratio <= 1/0.95 for both p50 and p95.',
        'runner_sha256':low.sha(__file__)})
    comparison()


def commands(suite,folder,listing,repeats,lane):
    server,client=production.commands(CATALOG,'krv100_full',folder,listing,repeats,lane)
    if suite=='original_dom':
        server[server.index('-p')+1]=str(low.REPO/'workstation/collection_matched_server.py')
        # Match original server default tolerance rather than override it.
        if '--tolerance' in server:
            i=server.index('--tolerance');del server[i:i+2]
    return wrap_command(server,folder),client


def run(suite,folder,listing,repeats,lane,dimension,scored,pass_index=None):
    root=low.ROOT/'results/krv100_full' if suite=='low_clutter' else ROOT
    verify=low.verify if suite=='low_clutter' else low.runtime.verify_trial_folder
    offset=passes.block(root,pass_index)[1] if pass_index else 0
    if pass_index:
        verify=lambda folder,root,listing,repeats:passes.verify(folder,root,listing,repeats,offset,suite=='low_clutter')
    if (folder/'verification.json').exists():
        return verify(folder,root,listing,repeats)
    if folder.exists():raise FileExistsError('Partial output retained; inspect before retry: '+str(folder))
    low.validate(CATALOG)
    if suite=='original_dom':
        for t in low.read(ROOT/'protocol.json')['tests']:assert low.sha(t['path'])==t['config_sha256']
    folder.mkdir(parents=True)
    server,client=commands(suite,folder,listing,repeats,lane)
    env=dict(os.environ,DOM_OFFICIAL_SOURCE=CATALOG['official_source'],DOM_TRIAL_RESULTS=str(folder/'trial_results.jsonl'),
        DOM_EXPECTED_SIM='4.5.0',DOM_EXPECTED_LAB='2.2.1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',DOM_REPEAT_OFFSET=str(offset))
    if pass_index==2:
        env['DOM_VIDEO_SELECTION_HISTORY']=str(root/'passes/01/full'/folder.name/'video_selection.jsonl')
    low.write(folder/'commands.json',{'server':server,'client':client,'scored':scored})
    low.write(folder/'runtime.json',{'simulation_gpu':0,'inference_gpu':1,'lane':lane,
        'execution_mode':MODE,'parallel_lanes':2 if MODE=='parallel_qualified_95pct' else 1,
        'repeats':repeats,'pass_index':pass_index,'repeat_offset':offset,'streaming':True,'scored':scored,'scheduling_receipt':str(CONTROL/'request.json')})
    target=len(listing.read_text().splitlines())*repeats
    processes=[];started=time.monotonic();changed=started;count=-1;checks=0;last_health=0
    try:
        with (folder/'server.log').open('w') as slog,(folder/'client.log').open('w') as clog:
            processes.append(subprocess.Popen(server,env=dict(env,CUDA_VISIBLE_DEVICES='0'),stdout=slog,stderr=subprocess.STDOUT,start_new_session=True))
            processes.append(subprocess.Popen(client,env=dict(env,CUDA_VISIBLE_DEVICES='1'),stdout=clog,stderr=subprocess.STDOUT,start_new_session=True))
            while processes[1].poll() is None:
                if STOP.is_set():raise InterruptedError('KRV suite stopped')
                if processes[0].poll() is not None:raise RuntimeError('Simulator exited: '+str(folder))
                records=folder/'trial_results.jsonl';n=len(records.read_text().splitlines()) if records.exists() else 0
                if n!=count:
                    count=n;changed=time.monotonic()
                    row={'suite':suite,'lane':lane,'model':'krv100_full','phase':'full' if scored else 'qualification',
                        'dimension':dimension,'state':'evaluating','completed_records_in_phase':n,
                        'phase_trial_target':target,'checkpoint_trial_target':1800,'image_port':27600+lane*10,'pass_index':pass_index}
                    with LOCK:ACTIVE[suite]=row
                    low.write(root/'suite_status.json',row)
                    if suite=='original_dom':low.write(ROOT/'status.json',{**row,'updated_utc':stamp()})
                    publish_status()
                if time.monotonic()-last_health>30:
                    last_health=time.monotonic();samples=log_samples(folder/'client.log')[-100:]
                    if len(samples)>=50:
                        current=stats([r['ms'] for r in samples]);ref=REFERENCES.get(suite)
                        ok=not ref or (current['p50_ms']<=ref['p50_ms']/.95 and current['p95_ms']<=ref['p95_ms']/.95)
                        checks=0 if ok else checks+1
                        low.write(folder/'latency_health.json',{'updated_utc':stamp(),'rolling':current,'isolated_reference':ref,
                            'meets_95pct_retention':ok,'consecutive_breaches':checks,'mode':MODE})
                        if scored and MODE=='parallel_qualified_95pct' and checks>=3:
                            SERIAL.set()
                            low.write(CONTROL/'serial_fallback.json',{'suite':suite,'folder':str(folder),'at':stamp(),
                                'reason':'95% latency-retention threshold exceeded for three rolling checks; finish current phases, then serialize. Preserve and flag all trials.'})
                if time.monotonic()-changed>1200:raise TimeoutError('No completed trials for 20 minutes')
                STOP.wait(2)
            if processes[1].returncode:raise RuntimeError('Policy exited: '+str(folder))
        result=verify(folder,root,listing,repeats)
        samples=log_samples(folder/'client.log')
        performance={**stats([r['ms'] for r in samples]),'action_coverage':coverage(folder/'client.log'),
            'outcomes':[r['success'] for r in low.runtime.read_records(folder)],'scored':scored}
        low.write(folder/'performance.json',performance)
        if scored and suite=='original_dom':comparison()
        return result
    except BaseException as error:
        low.write(folder/'failure.json',{'error':repr(error),'infrastructure_failure':True});raise
    finally:
        for process in reversed(processes):low.stop_group(process)
        with LOCK:ACTIVE.pop(suite,None)
        publish_status()


def qualification():
    global MODE,REFERENCES
    receipt=CONTROL/'qualification.json'
    if receipt.exists():
        d=low.read(receipt);REFERENCES=d['serial'];return d['parallel_accepted']
    q=CONTROL/'qualification';q.mkdir(exist_ok=True);references={};parallel={};listings={}
    for suite in ['low_clutter','original_dom']:
        paths=[]
        for i,dim in [(0,'CR'),(2,'LS'),(4,'SR')]:
            f=low.ROOT/'test_lists'/f'{dim}.txt' if suite=='low_clutter' else ROOT/'test_lists'/f'{i:02d}_{dim}.txt'
            paths.append(f.read_text().splitlines()[0])
        listing=q/(suite+'.txt');listing.write_text('\n'.join(paths)+'\n');listings[suite]=listing
        folder=q/'serial'/suite
        run(suite,folder,listing,2,0,'qualification',False)
        references[suite]=low.read(folder/'performance.json')
    REFERENCES=references
    MODE='parallel_qualification'
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(run,s,q/'parallel'/s,listings[s],2,i,'qualification',False) for i,s in enumerate(listings)]
            for f in futures:f.result()
        parallel={s:low.read(q/'parallel'/s/'performance.json') for s in listings}
        checks={s:{'p50_ratio':parallel[s]['p50_ms']/references[s]['p50_ms'],
            'p95_ratio':parallel[s]['p95_ms']/references[s]['p95_ms'],
            'coverage_retention':parallel[s]['action_coverage']/references[s]['action_coverage'] if references[s]['action_coverage'] else 0,
            'sample_count':parallel[s]['samples'],'outcome_pattern_equal':parallel[s]['outcomes']==references[s]['outcomes']} for s in listings}
        accepted=all(c['p50_ratio']<=1/.95 and c['p95_ratio']<=1/.95 and c['coverage_retention']>=.95
            and c['sample_count']>=50 and references[s]['samples']>=50 and c['outcome_pattern_equal'] for s,c in checks.items())
        result={'checks':checks,'parallel_accepted':accepted}
    except Exception as error:
        if STOP.is_set():raise
        result={'parallel_accepted':False,'parallel_error':repr(error)}
    low.write(receipt,{**result,'created_utc':stamp(),'serial':references,'parallel':parallel,
        'threshold_latency_ratio_max':1/.95,'all_qualification_trials_excluded':True,
        'limitation':'Six trials per suite per schedule test runtime feasibility; they do not establish success-rate equivalence.'})
    return result['parallel_accepted']


def comparison():
    baseline=low.read(BASE/'summary.json')['observed'];paper=low.read(BASE/'protocol.json')['paper_reference']
    observed={};all_records=[]
    for i,dim in enumerate(low.DIMS):
        f=ROOT/'full'/f'{i:02d}_{dim}'/'verification.json'
        if f.exists():
            observed[dim]=low.read(f)['success_percent'];all_records+=low.runtime.read_records(f.parent)
    if len(observed)==9:
        observed.update(Success=100*statistics.mean(r['success'] for r in all_records),Time=statistics.mean(r['task_time_s'] for r in all_records))
        low.write(ROOT/'summary.json',{'complete':True,'verified_trials':1800,'observed':observed,'protocol':low.read(ROOT/'protocol.json')})
    keys=[*low.DIMS,'Success','Time']
    md='# Original full DOM suite — KRV_Full comparison\n\n90 original configurations × 20 repeats = 1,800 scored trials. Released DynamicVLA scores below are the archived local original-suite evaluation; paper values are reported separately. KRV_Full uses privileged velocity and a different training history.\n\n| Method | '+' | '.join(keys)+' |\n|---|'+'---:|'*len(keys)+'\n'
    for label,values in [('DynamicVLA paper',paper),('Released DynamicVLA (local original suite)',baseline),('KRV100_Full (validation-best epoch 3)',observed)]:
        md+='| '+label+' | '+' | '.join(f'{values[k]:.2f}' if k in values else 'pending' for k in keys)+' |\n'
    (ROOT/'table1_comparison.md').write_text(md)


def main():
    global MODE
    signal.signal(signal.SIGTERM,lambda *_:STOP.set());signal.signal(signal.SIGINT,lambda *_:STOP.set())
    CONTROL.mkdir(parents=True,exist_ok=True)
    with (low.ROOT/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare()
        plan=CONTROL/'requested_followup_order.json'
        if plan.exists() and low.read(plan).get('krv_schedule')=='original_dom_only_after_current_phase':
            MODE='serial'
            receipt=CONTROL/'qualification.json'
            if receipt.exists():REFERENCES.update(low.read(receipt)['serial'])
            run('original_dom',ROOT/'pilot',ROOT/'test_lists/pilot.txt',1,0,None,False)
            for pass_index in [1,2]:
                repeats,_=passes.block(ROOT,pass_index)
                for i,dim in enumerate(low.DIMS):
                    listing=ROOT/'test_lists'/f'{i:02d}_{dim}.txt'
                    run('original_dom',ROOT/'passes'/f'{pass_index:02d}'/'full'/f'{i:02d}_{dim}',listing,repeats,0,dim,True,pass_index)
                    if pass_index==2:passes.merge(ROOT,dim,i,listing,False)
            comparison()
            low.write(ROOT/'status.json',{'state':'complete','verified_trials':1800,'updated_utc':stamp()})
            low.write(CONTROL/'status.json',{'state':'original_dom_complete_low_clutter_deferred_until_after_domv',
                'updated_utc':stamp(),'next':'DOMV_Full original DOM; then both low-clutter suites'})
            low.status('paused_for_domv_original_suite',active_evaluations=[],lanes=[])
            render(low.ROOT,False)
            return
        # Excluded pilots also verify policy, velocity observations and accounting.
        run('low_clutter',low.ROOT/'results/krv100_full/pilot',low.ROOT/'results/krv100_full/test_lists/pilot.txt',1,0,None,False)
        run('original_dom',ROOT/'pilot',ROOT/'test_lists/pilot.txt',1,0,None,False)
        parallel=qualification()
        MODE='parallel_qualified_95pct' if parallel else 'serial'
        previous=low.read(low.ROOT/'parallel_execution.json')
        low.write(CONTROL/'execution_before_two_suites.json',previous)
        low.write(low.ROOT/'parallel_execution.json',{'execution_mode':MODE,'concurrency':2 if parallel else 1,
            'started_utc':previous['started_utc'],'resumed_utc':stamp(),'runner_sha256':low.sha(__file__),'only_enabled_checkpoint':'krv100_full',
            'qualification_receipt':str(CONTROL/'qualification.json'),'runtime_note':'Only historical checkpoint, two separately reported suites; other models paused.'})
        if parallel:
            for i,dim in enumerate(low.DIMS):
                if SERIAL.is_set():MODE='serial'
                tasks=[('low_clutter',low.ROOT/'results/krv100_full/full'/f'{i:02d}_{dim}',low.ROOT/'results/krv100_full/test_lists'/f'{dim}.txt'),
                    ('original_dom',ROOT/'full'/f'{i:02d}_{dim}',ROOT/'test_lists'/f'{i:02d}_{dim}.txt')]
                if MODE=='parallel_qualified_95pct':
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        futures=[pool.submit(run,s,f,l,20,j,dim,True) for j,(s,f,l) in enumerate(tasks)]
                        for future in futures:future.result()
                else:
                    for s,f,l in tasks:run(s,f,l,20,0,dim,True)
                low.publish(CATALOG,upload=False)
        else:
            for suite in ['low_clutter','original_dom']:
                for i,dim in enumerate(low.DIMS):
                    root=low.ROOT/'results/krv100_full' if suite=='low_clutter' else ROOT
                    listing=root/'test_lists'/(f'{dim}.txt' if suite=='low_clutter' else f'{i:02d}_{dim}.txt')
                    run(suite,root/'full'/f'{i:02d}_{dim}',listing,20,0,dim,True)
                    low.publish(CATALOG,upload=False)
                if suite=='low_clutter':low.summarize(CATALOG,'krv100_full')
        low.summarize(CATALOG,'krv100_full');comparison();low.publish(CATALOG,upload=False)
        low.status('paused_by_user',only_enabled_checkpoint='krv100_full',requested_suites_complete=True,active_evaluations=[],lanes=[])
        low.write(ROOT/'status.json',{'state':'complete','verified_trials':1800,'updated_utc':stamp()})
        low.write(CONTROL/'status.json',{'state':'complete','scored_trials':3600,'other_models':'remain paused','updated_utc':stamp()})
        render(low.ROOT,False)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        low.write(CONTROL/'failure.json',{'error':repr(error),'updated_utc':stamp()});raise
