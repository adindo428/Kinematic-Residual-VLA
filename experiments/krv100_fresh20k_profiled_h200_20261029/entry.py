"""Isolated exact20k KRV100 readiness, production and checkpoint resume."""
import argparse,json,logging,os,sys,threading,time
from pathlib import Path
from datetime import timedelta
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,torch.distributed as dist
from core.dom_distributed import initialize_control_group,leader_call
from core.dom_distributed_training import distributed_smoke,train_distributed
from core.dom_training import load_family
from utils.dom_experiment import digest,atomic_json
ROOT=Path('/scratch/aparame/Dynamic_VLA/krv100-fresh20k-profiled-h200-20261029')
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['profile','train','benchmark']);a=p.parse_args()
 logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
 local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);assert world==2
 torch.cuda.set_device(local);torch.set_num_threads(int(os.environ.get('KRV_MAIN_THREADS','2')));device=torch.device('cuda',local)
 prop=torch.cuda.get_device_properties(device);assert prop.name=='NVIDIA H200',prop.name
 dist.init_process_group('nccl',timeout=timedelta(minutes=30));initialize_control_group()
 # Peak process+worker RSS, conservatively summed across ranks, including load/reload.
 import psutil
 stop=threading.Event()
 def monitor():
  process=psutil.Process();peak=0
  while not stop.wait(.1):
   try:peak=max(peak,process.memory_info().rss+sum(c.memory_info().rss for c in process.children(recursive=True)));os.environ['KRV_PEAK_RSS']=str(peak)
   except psutil.Error:pass
 thread=threading.Thread(target=monitor,daemon=True);thread.start()
 try:
  family=ROOT/'family';manifest=load_family(family);runtime=json.load(open(os.environ.get('KRV_RUNTIME_FILE',str(ROOT/'runtime.json'))));recipe=json.load(open(family/'recipe.json'))
  torch.set_num_threads(runtime['main_threads'])
  check=dict(recipe);assert digest({k:v for k,v in check.items() if k!='fingerprint'})==check['fingerprint'];assert recipe['manifest']==manifest['fingerprint']
  assert recipe['epochs']==10 and not recipe['early_stopping']['enabled'] and recipe['effective_batch']==352
  hardware={'name':prop.name,'total_memory':prop.total_memory,'torch':torch.__version__,'cuda':torch.version.cuda,'capability':[prop.major,prop.minor]}
  key=digest({'hardware':hardware,'world_size':world,'runtime':runtime,'variant':'krv100'})[:16]
  if a.mode=='benchmark':
   from scripts.krv_benchmark import benchmark
   benchmark(family,runtime,device)
  elif a.mode=='profile':
   report=distributed_smoke(family,'krv100',runtime,device)
   assert report['optimizer_updates']==20 and report['samples']==6689
   leader_call(lambda:atomic_json(ROOT/'readiness-complete.json',{'profile':key,'fingerprint':report['fingerprint'],'source':manifest['source_sha256'],'manifest':manifest['fingerprint'],'job_id':os.environ['SLURM_JOB_ID'],'selected_runtime_sha256':__import__('hashlib').sha256((ROOT/'runtime.json').read_bytes()).hexdigest()}))
  else:
   approval=json.load(open(ROOT/'readiness-complete.json'));profile=json.load(open(family/'distributed-profiles'/f'{key}.json'))
   claimed=profile.pop('fingerprint');assert digest(profile)==claimed==approval['fingerprint']
   assert profile['manifest']==manifest['fingerprint'] and profile['complete'] and profile['safe'] and profile['optimizer_updates']==20
   assert approval['source']==manifest['source_sha256'] and approval['profile']==key
   assert approval['selected_runtime_sha256']==__import__('hashlib').sha256((ROOT/'runtime.json').read_bytes()).hexdigest()
   selection=json.load(open(ROOT/'tuning/selection.json'));assert selection['runtime']==runtime and selection['readiness_passed']
   continuation=family/'runs/krv100'/f"continuation-{os.environ['SLURM_JOB_ID']}.json"
   leader_call(lambda:continuation.unlink(missing_ok=True))
   # Native trainer loads its own complete generation if one exists; otherwise released initialization.
   destination=family/'runs/krv100'
   if (destination/'provenance.json').exists() and not (destination/'latest.json').exists():raise RuntimeError('Expected own checkpoint missing; reconcile before restarting')
   result=train_distributed(family,'krv100',runtime,recipe,device)
   if result.get('interrupted'):
    leader_call(lambda:atomic_json(continuation,{'checkpoint_saved':True,'progress':result,'manifest':manifest['fingerprint']}))
    return 75
   assert result['epoch']==10 and result['samples']==24067730 and result['updates']==68380
   leader_call(lambda:atomic_json(ROOT/'completed.json',result))
  return 0
 finally:
  stop.set();thread.join(timeout=1);dist.destroy_process_group()
if __name__=='__main__':sys.exit(main())
