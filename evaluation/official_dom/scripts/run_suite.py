"""Launch upstream simulator/client on one allocated GPU and verify all trials."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from prepare import write_json
from report import verify_folder

def stop_group(process):
    if process is None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()

def run(stage,root,mode,index):
    source=stage/'source';scripts=stage/'scripts';assets=Path('/scratch/aparame/Dynamic_VLA/dom_benchmark')
    p=json.loads((root/'protocol.json').read_text())
    assert p['requested_gpu'] in ('l40s:1', 'L40S')
    if mode=='pilot':
        folder=root/'pilot';env_list=root/'test_lists/pilot.txt';trials=1
    else:
        assert json.loads((root/'pilot_verified.json').read_text())['verified']
        dimension=p['dimension_order'][index]
        folder=root/'full'/f'{index:02d}_{dimension}';env_list=root/'test_lists'/f'{index:02d}_{dimension}.txt';trials=20
    folder.mkdir(parents=True,exist_ok=False)
    runtime=stage/'runtime'/f"{mode}_{os.environ['SLURM_JOB_ID']}"
    for name in ('kit_data','kit_cache','tmp'):(runtime/name).mkdir(parents=True,exist_ok=True)
    environment=dict(os.environ,HF_HOME='/scratch/aparame/.cache/huggingface',HF_HUB_OFFLINE='1',
                     TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',
                     DOM_OFFICIAL_SOURCE=str(source),DOM_TRIAL_RESULTS=str(folder/'trial_results.jsonl'))
    environment['HF_LEROBOT_HOME']=environment['HF_HOME']+'/lerobot'
    environment.pop('LEROBOT_HOME',None)
    gpu_info=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True)
    write_json(folder/'runtime.json',{'slurm_job_id':os.environ['SLURM_JOB_ID'],'gpu_info':gpu_info,
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'source_commit':p['official_source_commit'],
        'requested_trials':trials,'env_list':str(env_list),'streaming':True})
    if mode=='pilot':
        subprocess.run([sys.executable,str(scripts/'probe_policy.py'),'--source',str(source),
                        '--checkpoint',str(stage/'checkpoint'),'--output',str(root/'policy_probe.json')],
                       env=environment,cwd=source,check=True,timeout=1200)
    port=22000+int(os.environ['SLURM_JOB_ID'])%18000
    isaacdeps=Path('/scratch/aparame/Dynamic_VLA/isaac_eval_deps/python')
    simulation_env='PYTHONPATH='+str(isaacdeps)+':'+str(source)
    server_command=['apptainer','exec','--cleanenv','--nv','-B','/scratch/aparame:/scratch/aparame',
        '-B','/home/aparame/Research/Dynamic_VLA:/home/aparame/Research/Dynamic_VLA',
        '-B',f'{runtime}/kit_data:/isaac-sim/kit/data','-B',f'{runtime}/kit_cache:/isaac-sim/kit/cache',
        '-B',f'{runtime}/tmp:/tmp','--env',simulation_env,
        '--env','CUDA_VISIBLE_DEVICES='+os.environ['CUDA_VISIBLE_DEVICES'],
        '--env','OMNI_KIT_ACCEPT_EULA=YES','--env','ACCEPT_EULA=Y','--env','TERM=xterm',
        '--env','DOM_OFFICIAL_SOURCE='+str(source),'--env','DOM_TRIAL_RESULTS='+str(folder/'trial_results.jsonl'),
        '--pwd',str(source),'/scratch/aparame/isaaclab.sif','/opt/IsaacLab/isaaclab.sh','-p',
        str(scripts/'server_observer.py'),'--headless','--enable_cameras','--device','cuda:0','--num_envs','1',
        '--scene_dir',str(assets/'scenes'),'--object_dir',str(assets/'objects'),'--env_cfg',str(env_list),
        '--output_dir',str(folder/'videos'),'-n',str(trials),'--save',
        '--img_port',str(port),'--act_port',str(port+1)]
    client_command=[sys.executable,str(source/'scripts/inference.py'),'-p',str(stage/'checkpoint'),
        '-a','official_dynamicvla_DOM','-r','euler','-d','-s',
        '--img_port',str(port),'--act_port',str(port+1),'-o',str(folder/'actions')]
    write_json(folder/'commands.json',{'server':server_command,'client':client_command})
    server=None;client=None
    def interrupted(signum,frame):raise RuntimeError(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        with (folder/'server.log').open('w') as server_log,(folder/'client.log').open('w') as client_log:
            server=subprocess.Popen(server_command,env=environment,cwd=source,stdout=server_log,stderr=subprocess.STDOUT,start_new_session=True)
            client=subprocess.Popen(client_command,env=environment,cwd=source,stdout=client_log,stderr=subprocess.STDOUT,start_new_session=True)
            last_progress=time.monotonic();last_size=-1
            while client.poll() is None:
                if server.poll() is not None:raise RuntimeError(f'Simulator exited early with {server.returncode}')
                result=folder/'trial_results.jsonl';size=result.stat().st_size if result.exists() else 0
                if size!=last_size:last_progress=time.monotonic();last_size=size
                if time.monotonic()-last_progress>1200:raise RuntimeError('No completed trial in 20 minutes; inspect simulator/policy logs')
                time.sleep(10)
            if client.returncode!=0:raise RuntimeError(f'Policy exited with {client.returncode}')
        result=verify_folder(folder,root,env_list,trials)
        if mode=='pilot':
            assert result['trials']==9
            write_json(root/'pilot_verified.json',result)
        print(json.dumps(result,indent=2),flush=True)
    except BaseException as error:
        write_json(folder/'failure.json',{'error':str(error),'benchmark_complete':False})
        raise
    finally:
        stop_group(client);stop_group(server)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,type=Path);p.add_argument('--root',required=True,type=Path);p.add_argument('--mode',required=True,choices=['pilot','full']);p.add_argument('--index',type=int,default=0)
    a=p.parse_args();run(a.stage,a.root,a.mode,a.index)
