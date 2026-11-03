"""Admission then training in the same bounded allocation, independent outputs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from placements import layout, NAMES


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--placement',required=True)
    p.add_argument('--cpu-contract',action='store_true');a=p.parse_args();r=a.root
    spec=layout(a.placement);stopping=[False]
    signal.signal(signal.SIGUSR1,lambda *_:stopping.__setitem__(0,True))
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    if a.name not in NAMES:raise ValueError('Unknown replica')
    if int(os.environ['SLURM_NNODES'])!=spec['nodes']:raise ValueError('Wrong node count')
    hosts=subprocess.check_output(['scontrol','show','hostnames',os.environ['SLURM_JOB_NODELIST']],text=True).split()
    env=dict(os.environ,MASTER_ADDR=hosts[0],MASTER_PORT=str(20000+int(os.environ['SLURM_JOB_ID'])%30000),
             KRFM_SITE='palmetto',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    env.pop('GLOO_SOCKET_IFNAME',None)
    stages=['contract'] if a.cpu_contract else ['profile','train']
    for stage in stages:
        if stopping[0]:return 75
        if not a.cpu_contract:
            if (r/'runs'/a.name/'completed.json').exists():return 0
            if stage=='profile' and (r/'profiles'/(a.name+'.json')).exists():
                if not json.loads((r/'profiles'/(a.name+'.json')).read_text()).get('approved'):raise ValueError('Failed readiness; fresh attempt required')
                continue
        cmd=['srun','--kill-on-bad-exit=1','--nodes='+str(spec['nodes']),
             '--ntasks='+str(spec['nodes']),'--ntasks-per-node=1',
             '--cpus-per-task='+str(4 if a.cpu_contract else spec['cpus']),
             sys.executable,str(r/'node_role.py'),'--root',str(r),'--name',a.name,'--stage',stage]
        if a.cpu_contract:cmd.append('--cpu-contract')
        if a.cpu_contract:(r/'contract-signal-ready.json').unlink(missing_ok=True)
        child=subprocess.Popen(cmd,env=env,cwd=str(r));sent=False
        while child.poll() is None:
            if a.cpu_contract and (r/'contract-signal-ready.json').exists():stopping[0]=True
            if stopping[0] and not sent:
                # Slurm forwards USR1 from srun to its remote task wrappers.
                child.send_signal(signal.SIGUSR1);sent=True
            time.sleep(.5)
        if stopping[0]:return 75
        if child.returncode:return child.returncode
    return 0


if __name__=='__main__':raise SystemExit(main())
