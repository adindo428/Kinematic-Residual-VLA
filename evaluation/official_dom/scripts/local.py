#!/usr/bin/env python3
"""Portable, serial DOM manipulation evaluation; no Slurm or Palmetto paths."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from prepare import prepare, sha256, write_json
from report import verify_folder, summarize
from run_suite import stop_group

SOURCE_COMMIT = '434dc7105cf6d559018d15af5fecbcdad44f036e'
PAPER45_SOURCE_COMMIT = 'a27a06d2ca74d0e987a5e552e01013073e93cfd8'
SOURCE_COMMITS = {'current51': SOURCE_COMMIT, 'paper45': PAPER45_SOURCE_COMMIT}
MODEL_REVISION = 'c9609d94c5940eb42436a31927d7e80ae454279e'
WEIGHTS_SHA256 = 'e3da8b1e0326d7a3ee6873f7fa17cdf7015201251901be5107377cb851ce3af8'
SCRIPTS = Path(__file__).resolve().parent


def fetch(stage, source_commit=SOURCE_COMMIT):
    stage.mkdir(parents=True, exist_ok=True)
    source = stage/'source'
    if not source.exists():
        subprocess.run(['git', 'clone', 'https://github.com/hzxie/DynamicVLA.git', str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'checkout', '--detach', source_commit], check=True)
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != source_commit:
        raise ValueError('Existing upstream checkout is at a different commit; use a fresh --stage')
    from huggingface_hub import snapshot_download
    snapshot_download('hzxie/dynamic-vla-DOM', revision=MODEL_REVISION,
                      allow_patterns=['config.json', 'model.safetensors', 'README.md'], local_dir=stage/'checkpoint')
    if sha256(stage/'checkpoint/model.safetensors') != WEIGHTS_SHA256:
        raise ValueError('Released weight checksum mismatch')
    print('Pinned upstream source and released checkpoint ready:', stage)


def commands(args, env_list, folder, trials):
    source=args.stage/'source'; checkpoint=args.stage/'checkpoint'
    server=[str(args.isaaclab), '-p', str(SCRIPTS/'server_observer.py'), '--headless', '--enable_cameras',
            '--device', 'cuda:0', '--num_envs', '1', '--scene_dir', str(args.assets/'scenes'),
            '--object_dir', str(args.assets/'objects'), '--env_cfg', str(env_list),
            '--output_dir', str(folder/'videos'), '-n', str(trials), '--save',
            '--img_port', str(args.port), '--act_port', str(args.port+1)]
    if getattr(args,'kit_args',''):
        server += ['--kit_args', args.kit_args]
    client=[str(args.policy_python), '-u', str(source/'scripts/inference.py'), '-p', str(checkpoint),
            '-a', 'official_dynamicvla_DOM', '-r', 'euler', '-d', '-s',
            '--img_port', str(args.port), '--act_port', str(args.port+1), '-o', str(folder/'actions')]
    return server,client


def process_environments(args, folder):
    environment=dict(os.environ,DOM_OFFICIAL_SOURCE=str(args.stage/'source'),
                     DOM_TRIAL_RESULTS=str(folder/'trial_results.jsonl'),TOKENIZERS_PARALLELISM='false')
    server_environment=dict(environment)
    client_environment=dict(environment)
    if args.sim_gpu is not None:
        server_environment['CUDA_VISIBLE_DEVICES']=str(args.sim_gpu)
    if args.policy_gpu is not None:
        client_environment['CUDA_VISIBLE_DEVICES']=str(args.policy_gpu)
    return environment,server_environment,client_environment


def run_one(args, mode, index=0):
    protocol=json.loads((args.root/'protocol.json').read_text())
    if Path(protocol['checkpoint']).resolve() != args.stage/'checkpoint':
        raise ValueError('Protocol/checkpoint stage mismatch; prepare a fresh output directory')
    dimension=protocol['dimension_order'][index]
    folder=args.root/'pilot' if mode=='pilot' else args.root/'full'/f'{index:02d}_{dimension}'
    env_list=args.root/'test_lists'/('pilot.txt' if mode=='pilot' else f'{index:02d}_{dimension}.txt')
    trials=1 if mode=='pilot' else 20
    if mode=='full' and not args.dry_run:
        if not json.loads((args.root/'pilot_verified.json').read_text())['verified']:
            raise ValueError('A verified pilot is required before full evaluation')
    server_command,client_command=commands(args,env_list,folder,trials)
    if args.dry_run:
        print(json.dumps({'mode':mode,'dimension':dimension if mode=='full' else 'nine-dimension pilot',
                          'server':server_command,'client':client_command},indent=2));return
    if folder.exists():
        if mode=='full' and args.resume_verified:
            verify_folder(folder,args.root,env_list,trials)
            print('Skipping verified dimension:',dimension,flush=True);return
        raise FileExistsError(f'Refusing to overwrite {folder}; select a fresh result root')
    if not args.isaaclab.is_file() or not os.access(args.isaaclab,os.X_OK):
        raise ValueError('--isaaclab must point to the executable IsaacLab isaaclab.sh')
    if sha256(args.stage/'checkpoint/model.safetensors') != protocol['checkpoint_weights_sha256']:
        raise ValueError('Checkpoint changed after preparation')
    # Revalidate source/JSON/assets before simulation; this produces no new protocol.
    for path,expected in protocol['asset_sha256'].items():
        if sha256(path)!=expected:raise ValueError(f'Asset changed: {path}')
    for item in protocol['tests']:
        if sha256(item['path'])!=item['config_sha256']:raise ValueError('Test config changed')
    actual_commit=subprocess.check_output(['git','-C',str(args.stage/'source'),'rev-parse','HEAD'],text=True).strip()
    if actual_commit!=protocol['official_source_commit'] or actual_commit not in SOURCE_COMMITS.values() or subprocess.check_output(['git','-C',str(args.stage/'source'),'diff','HEAD','--name-only'],text=True).strip():
        raise ValueError('Upstream checkout changed after preparation')
    folder.mkdir(parents=True)
    environment,server_environment,client_environment=process_environments(args,folder)
    if 'requested_runtime' in protocol:
        server_environment['DOM_EXPECTED_SIM']=protocol['requested_runtime']['isaacsim']
        server_environment['DOM_EXPECTED_LAB']=protocol['requested_runtime']['isaaclab']
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True)
    write_json(folder/'runtime.json',{'gpu':gpu,'cuda_visible_devices':environment.get('CUDA_VISIBLE_DEVICES'),
               'simulation_cuda_visible_devices':server_environment.get('CUDA_VISIBLE_DEVICES'),
               'policy_cuda_visible_devices':client_environment.get('CUDA_VISIBLE_DEVICES'),
               'source_commit':actual_commit,'streaming':True,'trials_per_configuration':trials})
    write_json(folder/'commands.json',{'server':server_command,'client':client_command})
    if mode=='pilot':
        probe_script=getattr(args,'policy_probe',None) or SCRIPTS/'probe_policy.py'
        probe_source=getattr(args,'policy_source',None) or args.stage/'source'
        subprocess.run([str(args.policy_python),str(probe_script),'--source',str(probe_source),
                        '--checkpoint',str(args.stage/'checkpoint'),'--output',str(args.root/'policy_probe.json')],
                       cwd=args.stage/'source',env=client_environment,check=True,timeout=1200)
    server=client=None
    try:
        with (folder/'server.log').open('w') as slog, (folder/'client.log').open('w') as clog, (folder/'gpu_memory.csv').open('w') as memory:
            memory.write('wall_time_s,gpu_index,memory_used_MiB,memory_total_MiB\n')
            server=subprocess.Popen(server_command,cwd=args.stage/'source',env=server_environment,stdout=slog,stderr=subprocess.STDOUT,start_new_session=True)
            client=subprocess.Popen(client_command,cwd=args.stage/'source',env=client_environment,stdout=clog,stderr=subprocess.STDOUT,start_new_session=True)
            last_progress=time.monotonic();last_size=-1
            while client.poll() is None:
                if server.poll() is not None:raise RuntimeError(f'Simulator exited early ({server.returncode}); see server.log')
                stats=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
                for line in stats.splitlines():memory.write(f'{time.time():.3f},{line}\n')
                memory.flush()
                path=folder/'trial_results.jsonl';size=path.stat().st_size if path.exists() else 0
                if size!=last_size:last_progress=time.monotonic();last_size=size
                if time.monotonic()-last_progress>1200:raise RuntimeError('No completed trial for 20 minutes; inspect logs')
                time.sleep(5)
            if client.returncode:raise RuntimeError(f'Policy exited ({client.returncode}); see client.log')
        result=verify_folder(folder,args.root,env_list,trials)
        if mode=='pilot':write_json(args.root/'pilot_verified.json',result)
        print(json.dumps(result,indent=2),flush=True)
    except BaseException as error:
        write_json(folder/'failure.json',{'benchmark_complete':False,'error':str(error)});raise
    finally:
        stop_group(client);stop_group(server)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    f=sub.add_parser('fetch');f.add_argument('--stage',required=True,type=Path)
    f.add_argument('--stack',choices=SOURCE_COMMITS,default='current51')
    p=sub.add_parser('prepare')
    p.add_argument('--stage',required=True,type=Path);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--assets',required=True,type=Path);p.add_argument('--gpu-label',default='Local GPU')
    p.add_argument('--stack',choices=SOURCE_COMMITS,default='current51')
    r=sub.add_parser('run')
    for key in ('stage','root','assets','isaaclab'):r.add_argument('--'+key,required=True,type=Path)
    r.add_argument('--policy-python',type=Path,default=Path(sys.executable))
    r.add_argument('--mode',choices=('pilot','full'),required=True)
    r.add_argument('--dimension',choices=('CR','DA','LS','VU','SR','MP','VG','MG','DR'))
    r.add_argument('--port',type=int,default=26000);r.add_argument('--dry-run',action='store_true')
    r.add_argument('--sim-gpu',type=int,help='Physical CUDA GPU index for simulation')
    r.add_argument('--policy-gpu',type=int,help='Physical CUDA GPU index for policy inference')
    r.add_argument('--kit-args',default='',help='Isaac Kit arguments, including SSD portable cache paths')
    r.add_argument('--resume-verified',action='store_true',help='Skip fully verified dimensions; partial directories are refused')
    s=sub.add_parser('report');s.add_argument('--root',required=True,type=Path)
    args=parser.parse_args()
    for key in ('stage','root','assets','isaaclab','policy_python'):
        if hasattr(args,key):setattr(args,key,getattr(args,key).expanduser().resolve())
    if args.command=='fetch':fetch(args.stage,SOURCE_COMMITS[args.stack])
    elif args.command=='prepare':
        if (args.root/'protocol.json').exists():raise FileExistsError('Output already prepared; use a fresh --root')
        if sha256(args.stage/'checkpoint/model.safetensors')!=WEIGHTS_SHA256:raise ValueError('Official weight checksum mismatch')
        prepare(args.stage,args.root,args.assets,gpu_label=args.gpu_label,
                source_commit=SOURCE_COMMITS[args.stack])
    elif args.command=='report':summarize(args.root)
    else:
        if not 1024<=args.port<=65534:parser.error('--port must be 1024..65534')
        if args.mode=='pilot' and args.dimension:parser.error('--dimension applies only to full mode')
        def interrupted(signum,frame):raise KeyboardInterrupt(f'Interrupted by signal {signum}')
        signal.signal(signal.SIGTERM,interrupted)
        order=json.loads((args.root/'protocol.json').read_text())['dimension_order']
        indices=[order.index(args.dimension)] if args.dimension else list(range(9))
        if args.mode=='pilot':run_one(args,'pilot')
        else:
            for index in indices:run_one(args,'full',index)
            if not args.dimension and not args.dry_run:summarize(args.root)

if __name__=='__main__':main()
