"""Scene-first evaluation with cached native streaming policies and safe fences."""
import argparse
import ast
from datetime import datetime,timezone
import importlib.util
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from resident_table1_worker import ResidentPool,minimum_retention

STOP=threading.Event()

def write(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def patch_run_tests(text):
    tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_tests')
    source=ast.get_source_segment(text,node)
    assert source.count('time.sleep(10)')==1
    return source.replace('time.sleep(10)','_resident_fence()')

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--img-port',type=int,default=27600);p.add_argument('--act-port',type=int,default=27601);args=p.parse_args()
    plan=json.loads(args.plan.read_text());args.output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='[%(levelname)s] %(asctime)s %(message)s',force=True)
    signal.signal(signal.SIGTERM,lambda *_:STOP.set());signal.signal(signal.SIGINT,lambda *_:STOP.set())
    pool=None;handler=None
    try:
        write(args.output/'status.json',{'stage':'loading_resident_policies','updated_utc':datetime.now(timezone.utc).isoformat()})
        pool=ResidentPool(plan['models'],args.output/'workers',profile=True)
        proof=pool.profile
        a,b=proof['single_resident'],proof['all_resident']
        proof['p50_ratio']=b['p50_ms']/a['p50_ms'];proof['p95_ratio']=b['p95_ms']/a['p95_ms']
        retention=proof.get('minimum_latency_retention',minimum_retention())
        proof['passed']=proof['p50_ratio']<=1/retention and proof['p95_ratio']<=1/retention
        proof['worker_receipts']=pool.receipts
        proof['per_model_probes']={}
        for key in pool.fronts:
            front=pool.select(key);before=front.command('probe',samples=3)
            front.command('place',device='cpu');front.command('place',device='cuda')
            after=front.command('probe',samples=3)
            assert before['action_sha256']==after['action_sha256'],'CPU/GPU cache changed actions'
            proof['per_model_probes'][key]={**after,'cpu_gpu_roundtrip_actions_exact':True}
        memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
        gpu=next([int(x.strip()) for x in line.split(',')] for line in memory.splitlines() if int(line.split(',')[0])==1)
        proof['policy_gpu_driver_memory_mib']={'used':gpu[1],'total':gpu[2]}
        proof['passed']=proof['passed'] and gpu[1]<.8*gpu[2]
        write(args.output/'resident_profile.json',proof)
        if not proof['passed']:raise RuntimeError(f'Resident idle-policy latency overhead exceeds {100*retention:g}% retention')
        import torch
        # The frontend only executes CPU streaming/queue code. Each GPU worker
        # imports its own frozen source, preserving DOMV's older implementation.
        source=Path(plan['models'][0]['source'])
        sys.path.insert(0,str(source))
        script=source/'scripts/inference.py'
        spec=importlib.util.spec_from_file_location('resident_upstream_inference',script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        exec(compile(patch_run_tests(script.read_text()),str(script),'exec'),module.__dict__)
        original_get=module.get_latest_observation
        def get(socket):
            if STOP.is_set():raise InterruptedError('Resident evaluation stopped')
            return original_get(socket)
        module.get_latest_observation=get
        obs,act=module.get_zmq_sockets('127.0.0.1',args.img_port,args.act_port)
        by_key={m['key']:m for m in plan['models']}
        for job in plan['jobs']:
            if STOP.is_set():raise InterruptedError('Resident evaluation stopped')
            folder=Path(job['folder']);folder.mkdir(parents=True,exist_ok=True)
            assert not (folder/'trial_results.jsonl').exists(),'Never overwrite resident trial records'
            model=by_key[job['model']];front=pool.select(job['model'])
            write(folder/'runtime.json',{'checkpoint_sha256':model['weight_sha256'],
                  'config_sha256':model['config_sha256'],'policy_source':model['source'],
                  'scored':job['scored'],'parallel_inference':False,'resident_checkpoints':len(pool.fronts),
                  'gpu_policy_capacity':pool.capacity,'all_policies_initialized_in_ram':True,
                  'gate':'causal_rules' if model['gated'] else 'constant',
                  'target_lead_s':0,'repeat_offset':job.get('repeat_offset',0),
                  'qualification':str(args.output/'resident_profile.json')})
            front.log=str(folder/'client.log')
            logger=logging.getLogger()
            if handler is not None:logger.removeHandler(handler);handler.close()
            handler=logging.FileHandler(folder/'client.log');handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s %(message)s'));logger.addHandler(handler)
            front.reset();module._resident_fence=front.idle
            cfg=json.loads((Path(model['checkpoint'])/'config.json').read_text())
            timestamps=cfg['delta_timestamps']['observation']
            while module.get_latest_observation(obs) is not None:pass
            tick=time.perf_counter()
            act.send_pyobj({'vla':job['model'],'epoch':0,'resident_job':job['id']})
            write(args.output/'status.json',{'stage':'evaluating','job':job,'updated_utc':datetime.now(timezone.utc).isoformat()})
            n,results=module.run_tests(obs,act,{'delta_timestamps':timestamps},front,model['checkpoint'],0,'euler',True,str(folder/'actions'))
            front.idle()
            assert n==job['repeats']
            logging.info('Evaluation completed. Total tests run: %d',n)
            module.save_test_results(results,str(folder/'actions'),job['model'])
            write(folder/'resident_client_completion.json',{'complete':True,'trials':n,'job':job,
                  'wall_seconds':time.perf_counter()-tick,'checkpoint':model['checkpoint'],
                  'checkpoint_sha256':model['weight_sha256'],'post_episode_wait':'worker completion fence',
                  'original_inference_dt_scale_sleep_preserved':True})
            write(args.output/'status.json',{'stage':'block_complete','job':job,'updated_utc':datetime.now(timezone.utc).isoformat()})
        act.send_pyobj({'resident_stop':True})
        write(args.output/'status.json',{'stage':'complete','updated_utc':datetime.now(timezone.utc).isoformat()})
    finally:
        if pool is not None:pool.close()

if __name__=='__main__':main()
