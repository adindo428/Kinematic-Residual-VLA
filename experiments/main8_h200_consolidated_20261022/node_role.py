"""One elastic launcher per allocated node; forward early signals to ranks."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import psutil


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--stage',required=True)
    p.add_argument('--cpu-contract',action='store_true');a=p.parse_args()
    stopping=[False];signal.signal(signal.SIGUSR1,lambda *_:stopping.__setitem__(0,True))
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    nodes=int(os.environ['SLURM_NNODES']);local=2
    if not a.cpu_contract:
        devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
        if len(devices)!=local:raise RuntimeError('Exactly two GPUs per node required')
    command=[sys.executable,'-m','torch.distributed.run','--nnodes='+str(nodes),
             '--nproc-per-node='+str(local),'--node-rank='+os.environ['SLURM_PROCID'],
             '--master-addr='+os.environ['MASTER_ADDR'],'--master-port='+os.environ['MASTER_PORT']]
    if a.cpu_contract:
        command += ['launcher_contract.py','--root',str(a.root)]
    else:
        command += ['train.py',a.stage,'--root',str(a.root),'--recipe',a.name]
        if a.stage=='profile':command += ['--runtime',str(a.root/'runtimes'/(a.name+'.json'))]
    child=subprocess.Popen(command,cwd=str(a.root))
    signalled=set()
    while child.poll() is None:
        if stopping[0]:
            try:
                for worker in psutil.Process(child.pid).children():
                    if worker.pid not in signalled and any(x.endswith(('train.py','launcher_contract.py')) for x in worker.cmdline()):
                        worker.send_signal(signal.SIGUSR1);signalled.add(worker.pid)
            except psutil.NoSuchProcess:pass
        time.sleep(.2)
    return child.returncode


if __name__=='__main__':raise SystemExit(main())
