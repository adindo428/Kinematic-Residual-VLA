"""Bounded batch/worker/thread sweep, followed by full native readiness."""
import hashlib,json,os,signal,subprocess,sys,time,threading
from pathlib import Path
import psutil
R=Path('/scratch/aparame/Dynamic_VLA/krv100-fresh20k-profiled-h200-20261029');sys.path.insert(0,str(R/'source'))
from utils.dom_experiment import atomic_json,digest
HERE=R/'tuning';HERE.mkdir(exist_ok=True)
def runtime(mb=32,workers=8,threads=2):
 assert 3*workers+2*threads<=32
 return {'microbatch':mb,'workers':workers,'main_threads':threads,'prefetch':2,'precision':'bf16','pin_memory':True,'fused':True}
class Telemetry:
 def __init__(self,proc):self.proc=psutil.Process(proc.pid);self.halt=threading.Event();self.samples=[];self.thread=threading.Thread(target=self.collect,daemon=True)
 def collect(self):
  previous={};last=time.monotonic()
  while not self.halt.wait(2):
   now=time.monotonic();cpu=0;current={}
   try:processes=[self.proc,*self.proc.children(recursive=True)]
   except psutil.Error:break
   for process in processes:
    try:
     times=process.cpu_times();v=times.user+times.system;current[process.pid]=v
     if process.pid in previous:cpu+=max(0,v-previous[process.pid])
    except psutil.Error:pass
   try:
    p=subprocess.run(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
    gpu=p.stdout.strip().splitlines() if p.returncode==0 else []
   except (subprocess.TimeoutExpired,OSError,KeyError):gpu=[]
   self.samples.append({'monotonic':now,'cpu_busy_cores':cpu/(now-last),'gpu_index_utilization_percent_memory_mib':gpu});previous=current;last=now
 def __enter__(self):self.thread.start();return self
 def __exit__(self,*_):self.halt.set();self.thread.join(timeout=7)
def run(rt,mode='benchmark'):
 key=digest(rt)[:12];case=HERE/(key+'.json');atomic_json(case,rt);before={p.name for p in (HERE/'benchmarks').glob('*.json')} if (HERE/'benchmarks').exists() else set()
 env=dict(os.environ,KRV_RUNTIME_FILE=str(case),KRV_MAIN_THREADS=str(rt['main_threads']),WANDB_MODE='disabled')
 label=mode+'-'+key;log=HERE/(label+'.log');started=time.monotonic()
 with log.open('ab',buffering=0) as stream:
  proc=subprocess.Popen([sys.executable,str(R/'launch.py'),mode],stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  with Telemetry(proc) as telemetry:
   try:status=proc.wait(timeout=900)
   except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=60)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    status=124
 atomic_json(HERE/(label+'-telemetry.json'),{'samples':telemetry.samples,'interval':'entire fresh process, including startup; not benchmark-only','runtime':rt})
 result={'runtime':rt,'mode':mode,'status':status,'log':str(log),'elapsed_with_startup_seconds':time.monotonic()-started,'telemetry':str(HERE/(label+'-telemetry.json'))}
 if status:
  text=log.read_text(errors='replace');result['oom']=('out of memory' in text.lower() or 'OutOfMemoryError' in text)
  if not result['oom']:raise RuntimeError('Non-OOM candidate failure; inspect '+str(log))
 elif mode=='benchmark':
  matches=[json.loads(p.read_text()) for p in (HERE/'benchmarks').glob('*.json')];receipt=next(x for x in matches if x['runtime']==rt)
  fp=receipt.pop('fingerprint');assert digest(receipt)==fp;result['measurement']=receipt;result['receipt_sha256']=fp
  window=[x for x in telemetry.samples if receipt['training_window_monotonic_start']+2<=x['monotonic']<=receipt['training_window_monotonic_end']]
  result['measured_window_utilization']={'cpu_busy_cores_mean':sum(x['cpu_busy_cores'] for x in window)/len(window) if window else None,'cpu_allocation_cores':32,'samples':window,'note':'Allocated GPU indices only; CPU includes ranks plus decoder workers; entire-process telemetry kept separately'}
 return result
def choose(cases):
 safe=[c for c in cases if c.get('status')==0 and c.get('measurement',{}).get('safe')]
 if not safe:raise RuntimeError('No safe decoded runtime')
 return max(safe,key=lambda c:c['measurement']['samples_per_second'])
def main():
 assert not (R/'readiness-complete.json').exists(),'Recorded readiness; reconcile before rerunning'
 cases=[];seen=set();deadline=time.monotonic()+100*60
 def trial(rt):
  key=digest(rt)
  if key in seen:return
  assert time.monotonic()<deadline,'Bounded sweep deadline exceeded; no production release'
  seen.add(key);case=run(rt);cases.append(case);atomic_json(HERE/'attempts.json',{'cases':cases,'complete':False})
 trial(runtime())
 for mb in (16,44,64,88,176):trial(runtime(mb))
 best=choose(cases)['runtime']['microbatch']
 for workers,threads in ((0,2),(2,2),(4,2),(8,1),(8,4),(4,4),(4,8)):
  trial(runtime(best,workers,threads))
 # Repeat baseline and winner to reduce a one-window ranking artifact.
 winner=choose(cases);repeat=run(winner['runtime']);baseline_repeat=run(runtime())
 cases.extend([repeat,baseline_repeat]);atomic_json(HERE/'attempts.json',{'cases':cases,'complete':True})
 assert repeat.get('measurement',{}).get('safe') and baseline_repeat.get('measurement',{}).get('safe')
 # Conservatively use the original fixed runtime when its repeat is faster.
 selected=repeat if repeat['measurement']['samples_per_second']>=baseline_repeat['measurement']['samples_per_second'] else baseline_repeat
 atomic_json(R/'runtime.json',selected['runtime'])
 selection={'runtime':selected['runtime'],'measurement':selected['measurement'],'baseline_repeat':baseline_repeat['measurement'],'repeat_rate_ratio':selected['measurement']['samples_per_second']/baseline_repeat['measurement']['samples_per_second'],'criterion':'highest decoded safe throughput; repeat winner against fixed MB32/W8/T2 control','readiness_passed':False,'benchmark_is_not_production':True,'job_id':os.environ['SLURM_JOB_ID']}
 atomic_json(HERE/'selection.json',selection)
 readiness=run(selected['runtime'],'profile');assert readiness['status']==0
 selection['readiness_passed']=True;selection['readiness']=json.load(open(R/'readiness-complete.json'));atomic_json(HERE/'selection.json',selection)
 print(json.dumps(selection),flush=True)
if __name__=='__main__':main()
