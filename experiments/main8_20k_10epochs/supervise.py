"""One allocation, disjoint GPUs/CPU affinity, per-model admission and resume."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--site',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--recipe')
    a=p.parse_args();r=a.root.resolve()
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(visible)!=4:raise RuntimeError('Exactly four allocated GPUs are required')
    cpus=sorted(os.sched_getaffinity(0))
    if len(cpus)<64:raise RuntimeError('Allocation must provide64 CPU cores')
    base='/scratch/aparame/Dynamic_VLA' if a.site=='palmetto' else '/work/nvme/bivv/aparameshwaran/dynamicvla'
    if a.site=='palmetto':
        vla=base+'/cross-vla-krfm-five-epochs-20261020/env/bin/python'
        native='/home/aparame/.conda/envs/dynamicvla/bin/python'
        if a.recipe not in ('smolvla_baseline','smolvla_kr','domv100'):
            raise RuntimeError('Palmetto requires one explicitly selected model per4-GPU node')
        roles=[(a.recipe,visible,cpus[:64],448,native if a.recipe=='domv100' else vla)]
    else:
        method='pi05' if a.site=='delta' else 'vlash'
        python=str(r/('env-delta-x86' if a.site=='delta' else 'env-deltaai-arm')/'bin/python')
        memory=256 if a.site=='delta' else 200
        roles=[(method+'_baseline',visible[:2],cpus[:32],memory,python),
               (method+'_kr',visible[2:],cpus[32:64],memory,python)]
    children={};stopping=[False];failed=[False]
    def request_stop(*_):
        stopping[0]=True
        for child in children.values():
            if child['process'].poll() is None:child['process'].send_signal(signal.SIGUSR1)
    signal.signal(signal.SIGUSR1,request_stop);signal.signal(signal.SIGTERM,request_stop)
    for name,gpus,cores,memory,python in roles:
        done=r/'runs'/name/'completed.json'
        if done.exists():continue
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(gpus),KRFM_SITE=a.site,
                 SLURM_CPUS_PER_TASK=str(len(cores)),SLURM_MEM_PER_NODE=str(memory*1024),
                 PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                 OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                 TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
        native_source=str(r/'delta-domv-f991015a')
        vlash=str(r/'assets/upstream-vlash') if a.site!='palmetto' else base+'/cross-vla-krfm-20261019/upstream-vlash'
        env['PYTHONPATH']=str(r)+':'+(native_source if name=='domv100' else vlash)
        log=(r/'logs'/(name+'-'+os.environ['SLURM_JOB_ID']+'.log')).open('a')
        command=['taskset','--cpu-list',','.join(map(str,cores)),'bash',str(r/'role.sh'),python,str(r),name]
        process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=str(r))
        children[name]={'process':process,'log':log,'checked':False}
    while any(c['process'].poll() is None for c in children.values()):
        for name,child in children.items():
            code=child['process'].poll()
            if code is None or child['checked']:continue
            child['checked']=True
            done=(r/'runs'/name/'completed.json').exists()
            stopped=(r/'runs'/name/'stopped.json').exists()
            if not done and not (stopping[0] and stopped):
                failed[0]=True;request_stop()
        time.sleep(2)
    for c in children.values():c['log'].close()
    complete=all((r/'runs'/name/'completed.json').exists() for name,*_ in roles)
    result={'complete':complete,'interrupted':stopping[0],'failed':failed[0],
            'models':{name:child['process'].returncode for name,child in children.items()}}
    (r/'logs'/('allocation-'+os.environ['SLURM_JOB_ID']+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if complete:return 0
    return 1 if failed[0] else 75


if __name__=='__main__':sys.exit(main())
