"""Bounded H100 microbatch/worker sweep; no production or W&B writes."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
import psutil
from common import atomic
from placements import NAMES

def execute(command,env,log,timeout=900):
 with log.open('w') as f:
  p=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:return p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   children=psutil.Process(p.pid).children(recursive=True)
   for proc in reversed(children):
    try:proc.terminate()
    except psutil.NoSuchProcess:pass
   p.terminate();_,alive=psutil.wait_procs(children,timeout=10)
   for proc in alive:
    try:proc.kill()
    except psutil.NoSuchProcess:pass
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:p.kill();p.wait()
   return 124

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
 started=time.monotonic();summary={};env=dict(os.environ,KRFM_SITE='palmetto',SLURM_CPUS_PER_TASK='32',
  PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
 env['PYTHONPATH']=str(r)+':/scratch/aparame/Dynamic_VLA/cross-vla-krfm-20261019/upstream-vlash'
 for name in NAMES:
  if time.monotonic()-started>3.6*3600:break
  dom=name=='domv100';method='dom' if dom else name.split('_')[0]
  python='/home/aparame/.conda/envs/dynamicvla/bin/python' if dom else sys.executable
  local_env=dict(env)
  if dom:local_env['PYTHONPATH']=str(r)+':/scratch/aparame/Dynamic_VLA/main8-20k-10epochs-20261021-r2/delta-domv-f991015a'
  base=[python,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',str(r/'train.py')]
  folder=r/'tuning'/name;folder.mkdir(exist_ok=True);attempts=[]
  micros=[11,22,44] if dom else [8,16,32] if method=='smolvla' else [2,4,8] if method=='pi05' else [4,8,16]
  def benchmark(micro,workers):
   runtime={'microbatch':micro,'workers':workers,'main_threads':2,'prefetch':2,'autocast_cache_enabled':False,'allocator':'expandable_segments:True'}
   label=f'mb{micro}-w{workers}';config=folder/(label+'-runtime.json');result=folder/(label+'.json');atomic(config,runtime)
   code=execute(base+['benchmark','--root',str(r),'--recipe',name,'--runtime',str(config),'--result',str(result)],local_env,folder/(label+'.log'))
   receipt=json.loads(result.read_text()) if result.exists() else {}
   eligible=code==0 and receipt.get('peak_gpu_reserved_bytes',float('inf'))<=.85*receipt.get('total_gpu_bytes',0) and receipt.get('conservative_node_rss_bytes',float('inf'))<=.85*256*1024**3
   attempts.append({'runtime':runtime,'returncode':code,'eligible':eligible,'receipt':receipt,'config':str(config),'log':str(folder/(label+'.log'))});atomic(folder/'attempts.json',attempts)
  for micro in micros:benchmark(micro,4)
  eligible=[x for x in attempts if x['eligible']]
  if eligible:
   best=max(eligible,key=lambda x:x['receipt']['samples_per_second'])
   for workers in [8,12]:benchmark(best['runtime']['microbatch'],workers)
   selected=sorted([x for x in attempts if x['eligible']],key=lambda x:x['receipt']['samples_per_second'],reverse=True)
   approved=False
   for attempt,best in enumerate(selected[:2]):
    code=execute(base+['profile','--root',str(r),'--recipe',name,'--runtime',best['config']],local_env,folder/f'readiness-{attempt}.log',timeout=1800)
    file=r/'profiles'/(name+'.json');receipt=json.loads(file.read_text()) if file.exists() else {}
    atomic(folder/f'readiness-{attempt}.json',{'returncode':code,'receipt':receipt,'runtime':best['runtime']})
    if code==0 and receipt.get('approved'):
     atomic(folder/'selection.json',{'runtime':best['runtime'],'short_benchmark':best['receipt'],'full_readiness':receipt,'diagnostic_only':True,'production_forbidden':True});approved=True;break
   summary[name]={'approved':approved,'attempts':len(attempts)}
  else:summary[name]={'approved':False,'attempts':len(attempts),'reason':'No candidate retained15percent headroom'}
  atomic(r/'diagnostic-results.json',summary)
 atomic(r/'diagnostic-complete.json',{'models':summary,'all_requested_measured':set(summary)==set(NAMES),'seconds':time.monotonic()-started,'production_forbidden':True})
if __name__=='__main__':main()
