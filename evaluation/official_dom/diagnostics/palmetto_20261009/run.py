"""Controlled six-trial comparisons on one or two allocated L40S GPUs."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

def write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def stop(process):
    if process is None: return
    try: os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError: return
    try: process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait()

def summarize(root, labels):
    results = {}
    labels = sorted(set(labels) | {p.name for p in root.iterdir() if p.is_dir() and (p/'verification.json').exists()})
    for label in labels:
        folder = root/label
        if not (folder/'verification.json').exists(): continue
        verification = json.loads((folder/'verification.json').read_text())
        trace = [json.loads(x) for x in (folder/'client_trace.jsonl').read_text().splitlines()]
        lag = [r['observation_index']-r['action_index'] for r in trace]
        trials = [json.loads(x) for x in (folder/'trial_results.jsonl').read_text().splitlines()]
        grasp = []
        for trial in trials:
            frames = json.loads(Path(trial['telemetry_file']).read_text())
            first = frames[0]['state']['object']['pos'][0][2]
            max_lift = max(f['state']['object']['pos'][0][2] for f in frames)-first
            closed = [f for f in frames if f['action'][0][-1]<0]
            gap = min((sum(f['finger_joint_positions'][0]) for f in closed),default=None)
            grasp.append({'env_name':trial['env_name'],'trial':trial['trial'],
                          'max_object_height_change_m':max_lift,
                          'minimum_finger_gap_while_commanded_closed_m':gap})
        results[label] = dict(verification,
            simulator=json.loads((folder/'simulator_versions.json').read_text()),
            client_actions=len(trace), late_action_count=sum(x>0 for x in lag),
            future_action_count=sum(x<0 for x in lag), maximum_action_lag_steps=max(lag,default=0),
            exact_index_action_percent=100*sum(x==0 for x in lag)/max(1,len(lag)),
            grasp_telemetry=grasp)
    write(root/'diagnostic_summary.json',results)
    with (root/'comparison.csv').open('w') as f:
        columns=['variant','trials','successes','success_percent','late_action_count',
                 'maximum_action_lag_steps','exact_index_action_percent']
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for label,row in results.items():
            writer.writerow({'variant':label,**{key:row[key] for key in columns[1:]}})
    print(json.dumps(results,indent=2),flush=True)

def run(stage,root,labels,gpus):
    from report import verify_folder
    tokens=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    assert len(tokens)>=gpus
    sim_gpu=tokens[0];policy_gpu=tokens[gpus-1]
    protocol=json.loads((root/'protocol.json').read_text())
    runtime=stage/'runtime'/os.environ['SLURM_JOB_ID']
    for name in ('kit_data','kit_cache','tmp'): (runtime/name).mkdir(parents=True,exist_ok=True)
    def interrupted(sig,frame): raise RuntimeError(f'Interrupted: {sig}')
    signal.signal(signal.SIGTERM,interrupted)
    for number,label in enumerate(labels):
        variant=label.removesuffix('_dual')
        source=stage/'sources'/variant
        folder=root/label;folder.mkdir(exist_ok=False)
        server=client=None
        env=dict(os.environ,HF_HOME='/scratch/aparame/.cache/huggingface',HF_HUB_OFFLINE='1',
                 TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',
                 CUDA_VISIBLE_DEVICES=policy_gpu,DOM_OFFICIAL_SOURCE=str(source),
                 DOM_TRIAL_RESULTS=str(folder/'trial_results.jsonl'),DOM_CLIENT_TRACE=str(folder/'client_trace.jsonl'))
        env['HF_LEROBOT_HOME']=env['HF_HOME']+'/lerobot';env.pop('LEROBOT_HOME',None)
        port=22000+(int(os.environ['SLURM_JOB_ID'])%17000)+2*number
        assets='/scratch/aparame/Dynamic_VLA/dom_benchmark'
        server_cmd=['apptainer','exec','--cleanenv','--nv','-B','/scratch/aparame:/scratch/aparame',
                    '-B','/home/aparame/Research/Dynamic_VLA:/home/aparame/Research/Dynamic_VLA',
                    '-B',f'{runtime}/kit_data:/isaac-sim/kit/data','-B',f'{runtime}/kit_cache:/isaac-sim/kit/cache',
                    '-B',f'{runtime}/tmp:/tmp','--pwd',str(source)]
        values={'PYTHONPATH':'/scratch/aparame/Dynamic_VLA/isaac_eval_deps/python:'+str(source),
                'CUDA_VISIBLE_DEVICES':sim_gpu,'OMNI_KIT_ACCEPT_EULA':'YES','ACCEPT_EULA':'Y','TERM':'xterm',
                'DOM_OFFICIAL_SOURCE':str(source),'DOM_TRIAL_RESULTS':env['DOM_TRIAL_RESULTS']}
        for k,v in values.items(): server_cmd+=['--env',k+'='+v]
        server_cmd+=['/scratch/aparame/isaaclab.sif','/opt/IsaacLab/isaaclab.sh','-p',str(stage/'scripts/observer.py'),
                    '--headless','--enable_cameras','--device','cuda:0','--num_envs','1',
                    '--scene_dir',assets+'/scenes','--object_dir',assets+'/objects',
                    '--env_cfg',str(root/'test-envs.txt'),'--output_dir',str(folder/'videos'),'-n','2','--save',
                    '--img_port',str(port),'--act_port',str(port+1)]
        client_cmd=[sys.executable,str(source/'scripts/inference.py'),'-p',str(stage/'checkpoint'),
                    '-a',label,'-r','euler','-d','-s','--img_port',str(port),'--act_port',str(port+1),
                    '-o',str(folder/'actions')]
        write(folder/'commands.json',{'server':server_cmd,'client':client_cmd})
        write(folder/'runtime.json',{'slurm_job_id':os.environ['SLURM_JOB_ID'],'sim_gpu':sim_gpu,
              'policy_gpu':policy_gpu,'allocated_gpus':gpus,'variant':protocol['variants'][variant],
              'gpu_info':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True)})
        try:
            with (folder/'server.log').open('w') as sl,(folder/'client.log').open('w') as cl:
                server=subprocess.Popen(server_cmd,env=env,cwd=source,stdout=sl,stderr=subprocess.STDOUT,start_new_session=True)
                client=subprocess.Popen(client_cmd,env=env,cwd=source,stdout=cl,stderr=subprocess.STDOUT,start_new_session=True)
                progress=time.monotonic();size=-1
                while client.poll() is None:
                    if server.poll() is not None: raise RuntimeError(f'Simulator exited {server.returncode}')
                    path=folder/'trial_results.jsonl';current=path.stat().st_size if path.exists() else 0
                    if current!=size: progress=time.monotonic();size=current
                    if time.monotonic()-progress>900: raise RuntimeError('No completed trial in 15 minutes')
                    time.sleep(5)
                if client.returncode!=0: raise RuntimeError(f'Policy exited {client.returncode}')
            verify_folder(folder,root,root/'test-envs.txt',2)
        except BaseException as e:
            write(folder/'failure.json',{'error':str(e)})
            raise
        finally:
            stop(client);stop(server)
        summarize(root,labels)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--gpus',type=int,default=1)
    p.add_argument('--labels',nargs='+',default=['original','gripper','gripper_aligned'])
    a=p.parse_args();run(a.stage,a.root,a.labels,a.gpus)
