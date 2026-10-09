#!/usr/bin/env python3
"""Prepare/run the four-model tracking stress benchmark on a Linux workstation."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'source'
SCRIPTS=SOURCE/'scripts'
sys.path.insert(0,str(SCRIPTS))


def stop(process):
    if process is None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()


def prepare(args):
    from tracking_checkpoint import checkpoint_files
    from tracking_online_common import sha256,write_json
    from prepare_benchmark import prepare as freeze
    base=json.loads((HERE/'templates/tracking_setup.json').read_text())
    base['layout']['scene_usd']=str(args.scenes/base['layout']['scene_usd'])
    base['layout']['object_usd']=str(args.objects/'apple'/base['layout']['object_usd'])
    base['models']={k:str(getattr(args,k)) for k in ('baseline','velocity','coordinate','coordinate_velocity')}
    base['checkpoint_metadata']={}
    for key,path in base['models'].items():
        _,config,weights,_=checkpoint_files(path,require_complete=True)
        base['checkpoint_metadata'][key]={'weights_sha256':sha256(weights),'config_sha256':sha256(config)}
    if args.robot_usd:base['config']['scene']['robot_usd']=str(args.robot_usd)
    # Keep the original room/object/layout hashes: changing the room is a new experiment.
    if args.root.exists():raise FileExistsError('Use a fresh --root; preparation never overwrites runs')
    template=args.root.parent/(args.root.name+'.input.json')
    if template.exists():raise FileExistsError(template)
    write_json(template,base)
    freeze(template,args.root,objects=args.objects)
    template.replace(args.root/'input.json')


def run(args):
    from tracking_online_common import write_json
    protocol=json.loads((args.root/'protocol.json').read_text())
    if args.mode=='full' and not args.dry_run:
        if not json.loads((args.root/'pilot_verified.json').read_text())['verified']:
            raise ValueError('Complete all 13 pilot shards before full evaluation')
    shards=protocol['pilot_shards'] if args.mode=='pilot' else list(range(90))
    if args.shard is not None:
        if args.shard not in shards:raise ValueError('Shard is not in this mode')
        shards=[args.shard]
    if args.dimension:
        index=protocol['dimension_order'].index(args.dimension)
        shards=[s for s in shards if s//10==index]
    model_args=[]
    for key,path in protocol['models'].items():model_args+=['--'+key.replace('_','-'),path]
    for shard in shards:
        folder=args.root/args.mode/f'{shard:03d}'
        manifest=json.loads((args.root/'shards'/f'{shard:03d}.json').read_text())
        if args.mode=='pilot':manifest['episodes']=manifest['episodes'][:1]
        endpoint=f'tcp://127.0.0.1:{args.port}'
        worker=[str(args.policy_python),'-u',str(SCRIPTS/'tracking_online_policy.py'),*model_args,'--endpoint',endpoint]
        sim=[str(args.isaaclab),'-p',str(SCRIPTS/'tracking_online_sim.py'),'--collector',str(HERE/'collector'),
             '--manifest',str(folder/'manifest.json'),'--output',str(folder),'--endpoint',endpoint,
             '--seconds',str(manifest['seconds'])]
        if args.mode=='pilot':sim+=['--all-videos']
        verify=[str(args.policy_python),str(SCRIPTS/'verify_benchmark.py'),'--manifest',str(folder/'manifest.json'),'--output',str(folder)]
        if args.dry_run:
            print(json.dumps({'shard':shard,'policy':worker,'simulator':sim,'verify':verify},indent=2));continue
        if not args.isaaclab.is_file() or not os.access(args.isaaclab,os.X_OK):raise ValueError('Invalid --isaaclab launcher')
        if folder.exists():
            if args.resume_verified:
                subprocess.run(verify,cwd=SOURCE,check=True);print('Skipped verified shard',shard,flush=True);continue
            raise FileExistsError(f'Refusing to overwrite {folder}')
        folder.mkdir(parents=True)
        write_json(folder/'manifest.json',manifest)
        write_json(folder/'commands.json',{'policy':worker,'simulator':sim,'verify':verify})
        env=dict(os.environ,PYTHONPATH=str(SOURCE)+os.pathsep+str(SCRIPTS)+os.pathsep+os.environ.get('PYTHONPATH',''),TOKENIZERS_PARALLELISM='false')
        process=simulation=None
        try:
            with (folder/'policy.log').open('w') as plog,(folder/'simulator.log').open('w') as slog:
                process=subprocess.Popen(worker,cwd=SOURCE,env=env,stdout=plog,stderr=subprocess.STDOUT,start_new_session=True)
                simulation=subprocess.Popen(sim,cwd=SOURCE,env=env,stdout=slog,stderr=subprocess.STDOUT,start_new_session=True)
                started=time.monotonic();last_progress=started;last_count=-1
                while simulation.poll() is None:
                    if process.poll() is not None and not (folder/'completion.json').exists():
                        # Simulator requests worker shutdown just before writing its marker.
                        if process.returncode:raise RuntimeError('Policy failed; inspect policy.log')
                    count=sum(1 for _ in folder.rglob('chunk_*.json'))
                    if count!=last_count:last_count=count;last_progress=time.monotonic()
                    if time.monotonic()-last_progress>1200:raise RuntimeError('No tracking progress for 20 minutes; inspect logs')
                    time.sleep(5)
                if simulation.returncode:raise RuntimeError('Simulator failed; inspect simulator.log')
                process.wait(timeout=30)
                if process.returncode:raise RuntimeError('Policy exited unsuccessfully')
            subprocess.run(verify,cwd=SOURCE,env=env,check=True)
        except BaseException as error:
            write_json(folder/'failure.json',{'error':str(error),'complete':False});raise
        finally:stop(simulation);stop(process)
    if args.mode=='pilot' and not args.dry_run:
        reports=[args.root/'pilot'/f'{s:03d}'/'verification.json' for s in protocol['pilot_shards']]
        if all(p.is_file() for p in reports):
            data=[json.loads(p.read_text()) for p in reports]
            if not all(p['verified'] and p['pairing_valid'] and p['rollouts']==4 for p in data):raise ValueError('Invalid pilot')
            write_json(args.root/'pilot_verified.json',{'verified':True,'shards':13,'rollouts':52})
    if args.mode=='full' and not args.dry_run:
        if all((args.root/'full'/f'{s:03d}'/'verification.json').exists() for s in range(90)):
            subprocess.run([str(args.policy_python),str(SCRIPTS/'summarize_benchmark.py'),'--root',str(args.root)],cwd=SOURCE,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare')
    for key in ('root','scenes','objects','baseline','velocity','coordinate','coordinate_velocity'):
        prep.add_argument('--'+key.replace('_','-'),required=True,type=Path)
    prep.add_argument('--robot-usd',type=Path)
    r=sub.add_parser('run')
    r.add_argument('--root',required=True,type=Path);r.add_argument('--isaaclab',required=True,type=Path)
    r.add_argument('--policy-python',type=Path,default=Path(sys.executable));r.add_argument('--mode',choices=('pilot','full'),required=True)
    r.add_argument('--shard',type=int);r.add_argument('--dimension',choices=('CR','DA','LS','VU','SR','MP','VG','MG','DR'))
    r.add_argument('--port',type=int,default=27000);r.add_argument('--dry-run',action='store_true');r.add_argument('--resume-verified',action='store_true')
    args=p.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.expanduser().resolve())
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    if args.command=='prepare':prepare(args)
    else:
        if not 1024<=args.port<=65535:p.error('Invalid port')
        run(args)

if __name__=='__main__':main()
