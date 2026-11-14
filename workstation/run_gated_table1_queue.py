#!/usr/bin/env python3
"""Original-DOM first-pass queue: gated KRV best, DOMV best, gated KRV latest."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

import run_low_clutter_table1 as low
import run_krv_full_two_suites as krv
import table1_two_pass as passes
from monitor_table1_latency import log_samples, coverage
from capture_table1_runtime import stats

CONTROL = krv.ROOT/'gated_queue'
STOP = threading.Event()


def stamp(): return datetime.now(timezone.utc).isoformat()


def publish(state, **kw):
    low.write(CONTROL/'status.json', {'state':state,'updated_utc':stamp(),**kw})
    from render_gated_original_table1 import render
    render()


def prepare():
    CONTROL.mkdir(exist_ok=True,parents=True)
    request = CONTROL/'request.json'
    if request.exists(): return low.read(request)
    catalog=low.read(low.ROOT/'catalog.json');low.validate(catalog)
    inventory=low.read(low.WORK/'training/krv100-full-deltaai-20261106/remote_inventory.json')
    source=low.WORK/'training/krv100-full-deltaai-20261106/source'
    definitions=[
        ('krv_best_gated','KRV100_Full epoch 3 (validation-best, gated)',
         low.WORK/'training/krv100-full-deltaai-20261106/best-epoch003',
         krv.ROOT/'variants/krv100_epoch003_gated',source,'best'),
        ('domv_best','DOMV_Full epoch 3 (validation-best)',
         low.WORK/'training/domv-full-deltaai-20261106/best-epoch003',
         low.WORK/'evaluations/domv-full-original-dom-20261106',
         low.WORK/'training/krv100-full-20261010/source',None),
        ('krv_latest_gated','KRV100_Full latest (4 epochs + partial epoch 5, gated)',
         low.WORK/'training/krv100-full-deltaai-20261106/latest-after-epoch004-u000304483',
         krv.ROOT/'variants/krv100_after_epoch004_partial005_gated',source,'latest'),
    ]
    models=[]
    for key,label,parent,root,policy_source,pointer in definitions:
        root.mkdir(parents=True,exist_ok=True)
        cfg=low.read(parent/'config.json')
        is_kr=pointer is not None
        if is_kr:
            entry=inventory['checkpoints'][pointer]
            assert low.sha(parent/'model.safetensors')==entry['files']['model.safetensors']['sha256']
            checkpoint=root/'checkpoint';checkpoint.mkdir()
            os.link(parent/'model.safetensors',checkpoint/'model.safetensors')
            shutil.copy2(parent/'provenance.json',checkpoint/'provenance.json')
            cfg['residual_phase_gate_mode']='causal_rules'
            assert cfg['enable_residual_flow'] and cfg['condition_on_object_velocity']
            assert cfg.get('residual_target_lead_s',0)==0
            low.write(checkpoint/'config.json',cfg)
            generation=entry['metadata']['generation'];progress=entry['metadata']['progress']
        else:
            checkpoint=parent
            receipt=low.read(low.WORK/'training/domv-full-deltaai-20261106/transfer_receipt.json')
            assert low.sha(parent/'model.safetensors')==receipt['files']['model.safetensors']['sha256']
            assert cfg['condition_on_object_velocity'] and not cfg['enable_residual_flow']
            generation='u000190656-95e1db0d';progress={'epoch':3,'cursor':0}
        model={'key':key,'label':label,'root':str(root),'checkpoint':str(checkpoint),
               'parent_checkpoint':str(parent),'source':str(policy_source),'kr':is_kr,
               'generation':generation,'progress':progress,'weight_sha256':low.sha(checkpoint/'model.safetensors'),
               'config_sha256':low.sha(checkpoint/'config.json'),'target_lead_s':0}
        protocol={**low.read(krv.BASE/'protocol.json'),'checkpoint_kind':'locally_trained',
                  'checkpoint_repo':None,'checkpoint':str(checkpoint),'trained_checkpoint':str(checkpoint),
                  'checkpoint_revision':generation,'method_label':label,'policy_source':str(policy_source),
                  'checkpoint_weights_sha256':model['weight_sha256'],'checkpoint_config_sha256':model['config_sha256'],
                  'training_scope':'full_dataset_only','intervention':{'causal_gate':is_kr,'target_lead_s':0,
                  'weights_unchanged':True,'training_gate_mode':'constant' if is_kr else 'not_applicable'},
                  'schedule':'First 10 repeats of each of 90 configurations; remaining 10 deferred',
                  'first_pass_trials':900,'final_suite_trials':1800,'checkpoint_progress':progress}
        if is_kr:low.write(root/'protocol.json',protocol)
        else:low.write(root/'gated_queue_execution_protocol.json',protocol)
        (root/'test_lists').mkdir(exist_ok=True)
        for path in (krv.BASE/'test_lists').glob('*.txt'):
            target=root/'test_lists'/path.name
            if target.exists():assert target.read_text()==path.read_text()
            else:shutil.copy2(path,target)
        pilot=(root/'test_lists/pilot.txt').read_text().splitlines()[:3]
        (root/'test_lists/gated_queue_pilot.txt').write_text('\n'.join(pilot)+'\n')
        if not (root/'two_pass_plan.json').exists():
            low.write(root/'two_pass_plan.json',{'repeats_per_pass':[10,10],'remaining_pass_requires_user_request':True})
        models.append(model)
    code=[low.REPO/'workstation'/name for name in ['run_gated_table1_queue.py','causal_gate_diagnostic_client.py',
          'causal_gate_diagnostic_runtime.py','causal_gate_diagnostic_server.py','krv_target_timing.py',
          'probe_causal_gate_diagnostic.py','probe_domv_full_checkpoint.py','inference_krvla.py',
          'krvla_policy_runtime.py','table1_two_pass.py','table1_video_sampling.py',
          'sampled_video_table1_server.py','collection_matched_server.py']]
    code += [p for s in {m['source'] for m in models} for p in Path(s).rglob('*.py')]
    plan={'enabled':True,'user_authorized':True,'requested_utc':stamp(),'models':models,
          'repeats_per_configuration_now':10,'trials_per_column_now':100,'columns':low.DIMS,
          'trials_per_checkpoint_now':900,'remaining_repeats_deferred':10,
          'suite':'original DOM; low-clutter remains paused','execution':'serial',
          'latest_checkpoint_explicitly_approved':True,'code_sha256':{str(p):low.sha(p) for p in code}}
    low.write(request,plan)
    publish('queued',queue=[m['key'] for m in models])
    return plan


def check(plan,model):
    for filename,digest in plan['code_sha256'].items():assert low.sha(filename)==digest,filename
    root=Path(model['root']);cp=Path(model['checkpoint'])
    assert low.sha(cp/'config.json')==model['config_sha256']
    assert low.sha(cp/'model.safetensors')==model['weight_sha256']
    for test in low.read(root/'protocol.json')['tests']:assert low.sha(test['path'])==test['config_sha256']


def probe(model):
    root=Path(model['root']);output=root/'gated_queue_policy_probe.json'
    if output.exists():return
    publish('probing',model=model['key'],label=model['label'])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',DOM_GATE_DIAGNOSTIC_ROOT=str(root))
    if model['kr']:
        cmd=[str(low.WORK/'envs/train/bin/python'),str(low.REPO/'workstation/probe_causal_gate_diagnostic.py')]
    else:
        cmd=[str(low.WORK/'envs/train/bin/python'),str(low.REPO/'workstation/probe_domv_full_checkpoint.py'),
             '--source',model['source'],'--checkpoint',model['checkpoint'],'--output',str(output)]
    with (root/'gated_queue_probe.log').open('a') as log:
        p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while p.poll() is None:
                if STOP.wait(2):raise InterruptedError('Queue stopped during probe')
                publish('probing',model=model['key'],label=model['label'])
            if p.returncode:raise RuntimeError('Checkpoint probe failed: '+str(root))
        finally:low.stop_group(p)
    if model['kr']:
        proof=low.read(root/'policy_probe.json');assert proof['same_noise_gate_one_matches_constant_exactly']
        low.write(output,{**proof,'checkpoint_sha256':model['weight_sha256'],'config_sha256':model['config_sha256']})


def phase(plan,model,dimension=None,remaining_only=False):
    root=Path(model['root']);pilot=dimension is None
    reuse=low.read(root/'cr_trial_reuse.json') if (root/'cr_trial_reuse.json').exists() else None
    if reuse and not remaining_only:
        if pilot:
            # The same checkpoint/configuration already passed a strict GPU
            # probe and 30 scored rollouts. These are credited once, not pilots.
            low.write(root/'pilot_waiver.json',{'basis':'Existing verified 30-trial gated diagnostic plus strict checkpoint probe',
                      'diagnostic_root':reuse['source_root'],'diagnostic_trials_are_scored_once':True})
            return
        if dimension=='CR':
            folder=root/'passes/01/full/00_CR';listing=root/'test_lists/00_CR.txt'
            if (folder/'verification.json').exists():return passes.verify(folder,root,listing,10,0,False)
            phase(plan,model,dimension,remaining_only=True)
            from table1_trial_reuse import compose
            sources=[{'root':reuse['source_root'],'folder':reuse['source_folder'],'listing':reuse['source_listing']},
                     {'root':str(root),'folder':str(root/'segments/CR_remaining70'),
                      'listing':str(root/'test_lists/00_CR_remaining70.txt')}]
            compose(folder,root,listing,10,sources)
            return passes.verify(folder,root,listing,10,0,False)
    name='pilot' if pilot else f'{low.DIMS.index(dimension):02d}_{dimension}'
    folder=root/'gated_queue_pilot' if pilot else root/'passes/01/full'/name
    listing=root/'test_lists'/('gated_queue_pilot.txt' if pilot else name+'.txt')
    if remaining_only:
        assert reuse and dimension=='CR'
        folder=root/'segments/CR_remaining70';listing=root/'test_lists/00_CR_remaining70.txt'
    repeats=1 if pilot else 10
    if (folder/'verification.json').exists():
        return passes.verify(folder,root,listing,repeats,0,False)
    if folder.exists():raise FileExistsError('Partial phase retained: '+str(folder))
    check(plan,model);folder.mkdir(parents=True)
    catalog=low.read(low.ROOT/'catalog.json');krv.CATALOG=catalog
    server,_=krv.commands('original_dom',folder,listing,repeats,0)
    client_script='causal_gate_diagnostic_client.py' if model['kr'] else 'inference_krvla.py'
    if model['kr']:server[server.index('-p')+1]=str(low.REPO/'workstation/causal_gate_diagnostic_server.py')
    client=[str(low.WORK/'envs/train/bin/python'),'-u',str(low.REPO/'workstation'/client_script),
            '--source',model['source'],'-p',model['checkpoint'],'-a',model['key'],'-r','euler','-d','-s',
            '--img_port','27600','--act_port','27601','-o',str(folder/'actions')]
    env=dict(os.environ,DOM_OFFICIAL_SOURCE=catalog['official_source'],DOM_TRIAL_RESULTS=str(folder/'trial_results.jsonl'),
             DOM_GATE_TRACE=str(folder/'gate_trace.jsonl'),DOM_GATE_WORKER_TRACE=str(folder/'gate_worker_trace.jsonl'),
             DOM_RESIDUAL_TARGET_LEAD_S='0',DOM_EXPECTED_SIM='4.5.0',DOM_EXPECTED_LAB='2.2.1',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',DOM_REPEAT_OFFSET='0')
    env.pop('DOM_VIDEO_SELECTION_HISTORY',None)
    low.write(folder/'commands.json',{'server':server,'client':client,'scored':not pilot})
    low.write(folder/'runtime.json',{'execution_mode':'isolated_serial','simulation_gpu':0,'inference_gpu':1,
              'model':model['key'],'repeats':repeats,'repeat_offset':0,'pass_index':1,'target_lead_s':0,'scored':not pilot,
              'reused_trials_in_column':30 if remaining_only else 0})
    processes=[];last=-1;changed=time.monotonic();health_tick=0
    try:
        with (folder/'server.log').open('w') as sl,(folder/'client.log').open('w') as cl:
            processes.append(subprocess.Popen(server,env=dict(env,CUDA_VISIBLE_DEVICES='0'),stdout=sl,stderr=subprocess.STDOUT,start_new_session=True))
            processes.append(subprocess.Popen(client,env=dict(env,CUDA_VISIBLE_DEVICES='1'),stdout=cl,stderr=subprocess.STDOUT,start_new_session=True))
            while processes[1].poll() is None:
                if STOP.is_set():raise InterruptedError('Gated Table 1 queue stopped')
                if processes[0].poll() is not None:raise RuntimeError('Simulator exited: '+str(folder))
                f=folder/'trial_results.jsonl';n=len(f.read_text().splitlines()) if f.exists() else 0
                if n!=last:last=n;changed=time.monotonic()
                row={'model':model['key'],'label':model['label'],'dimension':dimension,'phase':'pilot' if pilot else 'full',
                     'completed_records_in_phase':n+(30 if remaining_only else 0),'phase_trial_target':3 if pilot else 100,
                     'new_records_in_phase':n,'reused_records':30 if remaining_only else 0,'pass_index':1,'folder':str(folder)}
                publish('evaluating',**row)
                if time.monotonic()-health_tick>=30:
                    health_tick=time.monotonic();samples=log_samples(folder/'client.log')[-100:]
                    reference=low.read(root/'gated_queue_pilot/performance.json') if (root/'gated_queue_pilot/performance.json').exists() else None
                    current=stats([r['ms'] for r in samples])
                    low.write(folder/'latency_health.json',{'updated_utc':stamp(),'rolling':current,'isolated_pilot_reference':reference,
                              'execution':'serial; no competing evaluation lanes'})
                if time.monotonic()-changed>1200:raise TimeoutError('No completed trials for 20 minutes')
                STOP.wait(5)
            if processes[1].returncode:raise RuntimeError('Policy exited: '+str(folder/'client.log'))
        result=passes.verify(folder,root,listing,repeats,0,False)
        if model['kr']:
            trace=[json.loads(l) for l in (folder/'gate_worker_trace.jsonl').read_text().splitlines()]
            assert trace and all(r['target_lead_s']==0 for r in trace)
        low.write(folder/'performance.json',{**stats([r['ms'] for r in log_samples(folder/'client.log')]),
                  'action_coverage':coverage(folder/'client.log'),'scored':not pilot})
        return result
    except BaseException as error:
        low.write(folder/'failure.json',{'error':repr(error),'partial_records_preserved':True});raise
    finally:
        for p in reversed(processes):low.stop_group(p)


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    signal.signal(signal.SIGTERM,lambda *_:STOP.set());signal.signal(signal.SIGINT,lambda *_:STOP.set())
    plan=prepare()
    if args.prepare_only:return
    with (low.ROOT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for model in plan['models']:
                check(plan,model);probe(model);phase(plan,model)
                for d in low.DIMS:phase(plan,model,d)
                low.write(Path(model['root'])/'first_pass_summary.json',{'first_pass_complete':True,'verified_trials':900,
                    'checkpoint':model,'remaining_trials_deferred':900,'updated_utc':stamp()})
            publish('first_pass_queue_complete',remaining_repeats_deferred=10)
        except BaseException as error:
            publish('paused_by_user' if STOP.is_set() else 'failed',error=repr(error));raise


if __name__=='__main__':main()
