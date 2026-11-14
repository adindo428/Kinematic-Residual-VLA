#!/usr/bin/env python3
"""L40S-only closed-loop evaluation of one pinned model on the shared mini-suite."""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'evaluation/official_dom/scripts'))
from prepare import sha256,write_json
from report import read_records,verify_folder
from run_suite import stop_group


def read(path):return json.loads(Path(path).read_text())


def phase(catalog,name,mode,index):
    root=ROOT/'results'/name;protocol=read(root/'protocol.json')
    folder=root/'pilot' if mode=='pilot' else root/'full'/f"{index:02d}_{protocol['dimension_order'][index]}"
    listing=root/'test_lists'/('pilot.txt' if mode=='pilot' else f"{index:02d}_{protocol['dimension_order'][index]}.txt")
    repeats=1 if mode=='pilot' else 20
    if folder.exists():
        if (folder/'verification.json').exists():return verify_folder(folder,root,listing,repeats)
        raise FileExistsError(f'Partial evaluation retained: {folder}; archive it before retrying')
    if mode!='pilot' and not read(root/'pilot_verified.json')['verified']:raise ValueError('Pilot gate not passed')
    for test in protocol['tests']:
        if sha256(test['path'])!=test['config_sha256']:raise ValueError('Scenario changed')
    for path,expected in catalog['files_sha256'].items():
        if sha256(path)!=expected:raise ValueError('Frozen evaluation code changed: '+path)
    folder.mkdir(parents=True)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(visible)!=2:raise ValueError('Exactly two allocated L40S GPUs are required')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True)
    if any('L40S' not in line for line in gpu.splitlines()):raise ValueError('L40S-only node required')
    write_json(folder/'runtime.json',{'gpu':gpu,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
        'cuda_visible_devices':visible,'sim_gpu':visible[0],'policy_gpu':visible[1],
        'streaming':True,'rotation':'euler','delta_actions':True,'repeats':repeats})
    env=dict(os.environ,DOM_OFFICIAL_SOURCE=catalog['official_source'],
        DOM_TRIAL_RESULTS=str(folder/'trial_results.jsonl'),DOM_RECORD_RUNTIME='1',
        HF_HOME='/scratch/aparame/.cache/huggingface',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2')
    client_env=dict(env,CUDA_VISIBLE_DEVICES=visible[1])
    port=22000+int(os.environ['SLURM_JOB_ID'])%18000
    client=[catalog['python'],'-u',str(ROOT/'workstation/multivla_table1_client.py'),
            '--catalog',str(ROOT/'catalog.json'),'--model',name,'--output',str(folder),'--port',str(port)]
    if mode=='pilot':subprocess.run([*client,'--probe'],env=client_env,check=True,timeout=1200)
    runtime=ROOT/'runtime'/f"{name}-{os.environ['SLURM_JOB_ID']}-{mode}-{index}"
    for part in ('kit_data','kit_cache','tmp'):(runtime/part).mkdir(parents=True,exist_ok=True)
    server=[catalog['apptainer'],'exec','--cleanenv','--nv','-B','/scratch/aparame:/scratch/aparame',
        '-B','/home/aparame:/home/aparame','-B',f'{runtime}/kit_data:/isaac-sim/kit/data',
        '-B',f'{runtime}/kit_cache:/isaac-sim/kit/cache','-B',f'{runtime}/tmp:/tmp',
        '--env','PYTHONPATH=/scratch/aparame/Dynamic_VLA/isaac_eval_deps/python:'+catalog['official_source'],
        '--env','CUDA_VISIBLE_DEVICES='+visible[0],'--env','OMNI_KIT_ACCEPT_EULA=YES','--env','ACCEPT_EULA=Y',
        '--env','TERM=xterm','--env','DOM_OFFICIAL_SOURCE='+catalog['official_source'],
        '--env','DOM_TRIAL_RESULTS='+str(folder/'trial_results.jsonl'),'--env','DOM_RECORD_RUNTIME=1',
        '--pwd',catalog['official_source'],catalog['simulator_container'],'/opt/IsaacLab/isaaclab.sh','-p',
        str(ROOT/'workstation/krvla_table1_server.py'),'--headless','--enable_cameras','--device','cuda:0',
        '--num_envs','1','--scene_dir',catalog['assets']+'/scenes','--object_dir',catalog['assets']+'/objects',
        '--env_cfg',str(listing),'--output_dir',str(folder/'videos'),'-n',str(repeats),'--save',
        '--img_port',str(port),'--act_port',str(port+1)]
    write_json(folder/'commands.json',{'server':server,'client':client})
    processes=[]
    try:
        with (folder/'server.log').open('w') as slog,(folder/'client.log').open('w') as clog:
            processes.append(subprocess.Popen(server,env=env,stdout=slog,stderr=subprocess.STDOUT,start_new_session=True))
            processes.append(subprocess.Popen(client,env=client_env,stdout=clog,stderr=subprocess.STDOUT,start_new_session=True))
            changed=time.monotonic();size=-1
            while processes[1].poll() is None:
                if processes[0].poll() is not None:raise RuntimeError('Simulator exited early; see server.log')
                path=folder/'trial_results.jsonl';new=path.stat().st_size if path.exists() else 0
                if new!=size:changed=time.monotonic();size=new
                if time.monotonic()-changed>1200:raise RuntimeError('No completed trial for 20 minutes')
                time.sleep(5)
            if processes[1].returncode:raise RuntimeError('Policy client failed; see client.log')
        verified=verify_folder(folder,root,listing,repeats)
        if mode=='pilot':write_json(root/'pilot_verified.json',verified)
        return verified
    except BaseException as error:
        write_json(folder/'failure.json',{'error':str(error),'infrastructure_failure':True});raise
    finally:
        for process in reversed(processes):stop_group(process)


def summarize(root,catalog,name):
    protocol=read(root/'protocol.json');records=[];conditions=[]
    for i,dim in enumerate(protocol['dimension_order']):
        folder=root/'full'/f'{i:02d}_{dim}'
        verified=verify_folder(folder,root,root/'test_lists'/f'{i:02d}_{dim}.txt',20)
        if verified['trials']!=40:raise ValueError('Expected 40 verified trials per condition')
        records.extend(read_records(folder));conditions.append({'Condition':dim,'Trials':40,
            'Success':verified['success_percent'],'Time':verified['mean_task_time_s']})
    if len(records)!=360:raise ValueError('Incomplete 360-trial suite')
    tests={t['name']:t for t in protocol['tests']};tasks=defaultdict(list)
    for row in records:tasks[tests[row['env_name']]['task']].append(row)
    observed={r['Condition']:r['Success'] for r in conditions}
    observed.update(Success=100*statistics.mean(r['success'] for r in records),
                    Time=statistics.mean(r['task_time_s'] for r in records))
    simulator=read(root/'pilot/simulator_runtime.json') if (root/'pilot/simulator_runtime.json').exists() else {}
    write_json(root/'summary.json',{'complete':True,'verified_trials':360,'observed':observed,
        'conditions':conditions,'tasks':[{'Task':t,'Trials':len(rows),
            'Success':100*statistics.mean(r['success'] for r in rows),
            'Time':statistics.mean(r['task_time_s'] for r in rows)} for t,rows in tasks.items()],
        'model':name,'checkpoint':catalog['models'][name],'simulator_runtime':simulator,'protocol':protocol})
    fields=['Method',*protocol['dimension_order'],'Success','Time'];label=catalog['models'][name]['label']
    with (root/'table1_task_subset.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerow({'Method':label,**observed})
    (root/'table1_task_subset.md').write_text('# '+label+': task-aligned Table 1\n\n| '+' | '.join(fields)+
        ' |\n|'+'|'.join(['---']+['---:']*11)+'|\n| '+label+' | '+
        ' | '.join(f'{observed[k]:.2f}' for k in fields[1:])+' |\n\n360 verified manipulation trials; all-trial time includes failures. L40S simulation/inference; runtime versions recorded.\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);args=parser.parse_args()
    catalog=read(ROOT/'catalog.json');name=args.model;root=ROOT/'results'/name
    def status(value,**details):write_json(root/'suite_status.json',{'state':value,'model':name,
        'expected_trials':360,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),**details})
    def stop(signum,frame):raise KeyboardInterrupt(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    try:
        status('pilot');phase(catalog,name,'pilot',0)
        for i,dim in enumerate(catalog['dimension_order']):
            status('running',dimension=dim);phase(catalog,name,'full',i)
        summarize(root,catalog,name);status('complete',verified_trials=360)
    except BaseException as error:status('failed',error=str(error));raise


if __name__=='__main__':main()
