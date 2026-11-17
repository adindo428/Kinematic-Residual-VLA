"""Resume local20k VPD training only after the complete100trial-per-arm study."""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'scratch/dom45/training/krv-vpd-20k-resume-20261115'
EVAL=ROOT/'scratch/dom45/evaluations/krv-cr50-checkpoint-ablation-20261114'
STOP=False

def read(path):return json.loads(path.read_text())
def write(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def ready():
 try:
  plan=read(EVAL/'plan.json');done=read(EVAL/'completion.json');extension=read(EVAL/'extension-status.json')
 except FileNotFoundError:return False
 if plan['protocol']['trials_per_arm']!=100 or extension.get('stage')!='complete' or not done.get('verified'):return False
 scores=done.get('scores',{})
 return len(scores)==6 and all(value[1]==100 for value in scores.values())
def stop(*_):
 global STOP
 STOP=True

def main():
 if (STAGE/'NO_MORE_20K_TRAINING.json').exists():
  write(STAGE/'queue-status.json',{'stage':'cancelled_by_user','reason':'No more20k training; do not restart'})
  return
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 with (STAGE/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert read(STAGE/'migration.json')['verified'] and read(STAGE/'cpu-tests.json')['passed']
  while not STOP and not ready():
   if (STAGE/'NO_MORE_20K_TRAINING.json').exists():return
   write(STAGE/'queue-status.json',{'stage':'waiting_for_evaluation','dependency':'Allsix KRV arms complete100trials each with verified results','will_resume_update':26959,'VPD_training_and_native_restoration':True,'VI':False,'pid':os.getpid()})
   # A stopped/failed evaluation requires attention; never take its GPUs or skip its trials.
   time.sleep(10)
  if STOP:
   write(STAGE/'queue-status.json',{'stage':'queue_cancelled_without_starting_training'});return
  time.sleep(5)
  assert ready()
  if (STAGE/'NO_MORE_20K_TRAINING.json').exists():return
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',HF_HOME=str(ROOT/'scratch/dom45/cache/huggingface'),HF_HUB_CACHE=str(ROOT/'scratch/dom45/cache/huggingface/hub'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  with (STAGE/'coordinator.log').open('a') as log:
   child=subprocess.Popen([sys.executable,'-u',str(STAGE/'train.py'),'--mode','coordinate'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
  write(STAGE/'queue-status.json',{'stage':'training_coordinator_started','pid':child.pid,'dependency_verified':True,'readiness_required_before_production':True})
  (STAGE/'coordinator.pid').write_text(str(child.pid)+'\n')
  code=child.wait()
  write(STAGE/'queue-status.json',{'stage':'complete' if code==0 else 'coordinator_stopped','exit_code':code,'pid':child.pid})

if __name__=='__main__':main()
