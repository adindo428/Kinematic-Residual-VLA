"""Forward graceful stop only to native training workers; preserve torchrun."""
import subprocess,signal,os,sys,time
from pathlib import Path
import psutil
R=Path('/scratch/aparame/Dynamic_VLA/krv100-fresh20k-h200-20261029');mode=sys.argv[1]
proc=subprocess.Popen([sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=2',str(R/'source/scripts/fresh20k_krv.py'),mode])
def stop(sig,frame):
 for child in psutil.Process(proc.pid).children():
  try:
   if any('fresh20k_krv.py' in a for a in child.cmdline()):os.kill(child.pid,signal.SIGUSR1)
  except psutil.Error:pass
for sig in (signal.SIGUSR1,signal.SIGTERM):signal.signal(sig,stop)
status=proc.wait()
continuation=R/'family/runs/krv100'/f"continuation-{os.environ['SLURM_JOB_ID']}.json"
if mode=='train' and continuation.exists():
 import json
 receipt=json.load(open(continuation));assert receipt['checkpoint_saved']
 status=75
sys.exit(status)
