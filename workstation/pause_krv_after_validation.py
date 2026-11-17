"""Pin a validated local KRV epoch, then checkpoint-stop the recorded workers."""
import argparse,json,os,signal,time
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'scratch/dom45/training/krv-distance-gate-history-20261113'
ENTRY=ROOT/'workstation/run_velocity_history_training.py'

def read(path):return json.loads(path.read_text())
def write(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def validated_pointer(run,epoch):
 try:
  pointer=read(run/'latest.json');p=pointer['progress']
  if p['epoch']!=epoch or p['cursor']!=0 or p.get('pending_validation'):return None
  metrics=read(run/f'validation-epoch-{epoch:03d}.json')
  assert 'h20/action_mse' in metrics
  generation=run/'generations'/pointer['generation']
  if not all((generation/name).is_file() for name in ('model.safetensors','training_state.pt','config.json','provenance.json')):return None
  return pointer,metrics
 except FileNotFoundError:return None

def pin(run,pointer,destination):
 generation=run/'generations'/pointer['generation'];destination.mkdir(parents=True,exist_ok=True)
 for name in ('model.safetensors','training_state.pt','config.json','provenance.json'):
  source=generation/name;target=destination/name
  if target.exists():assert source.stat().st_ino==target.stat().st_ino
  else:os.link(source,target)
 write(destination/'pointer.json',pointer)

def matches(process,record):
 try:
  cmd=process.cmdline()
  return process.create_time()==record['created'] and str(ENTRY) in cmd and '--mode' in cmd and cmd[cmd.index('--mode')+1]=='train' and 'torch.distributed.run' not in cmd
 except (psutil.NoSuchProcess,psutil.AccessDenied):return False

def main():
 request=read(STAGE/'pause-request.json');epoch=request['after_validated_epoch'];workers=request['workers'];run=STAGE/'family/runs/krv100'
 status=STAGE/'pause-status.json';deadline=time.monotonic()+4*3600
 while True:
  if not read(STAGE/'pause-request.json').get('enabled'):return
  ready=validated_pointer(run,epoch)
  if ready:break
  living=[p for record in workers if matches(p:=psutil.Process(record['pid']),record)]
  if len(living)!=len(workers):
   write(status,{'stage':'stopped_before_target','target_epoch':epoch,'detail':'A recorded training worker exited before target validation'});return
  progress=read(run/'latest.json')['progress']
  assert progress['epoch']<=epoch,'Target epoch was missed; review checkpoint pointers before proceeding'
  assert time.monotonic()<deadline,'Target validation did not complete within the bounded four-hour watcher window'
  write(status,{'stage':'waiting_for_validation','target_epoch':epoch,'latest_checkpoint_progress':progress,'watcher_pid':os.getpid()})
  time.sleep(2)
 pointer,metrics=ready;archive=STAGE/f'paused-after-epoch{epoch:03d}'
 pin(run,pointer,archive/'checkpoint');write(archive/'validation.json',metrics)
 best=read(run/'best.json');pin(run,best,archive/'validation-best')
 assert validated_pointer(run,epoch)[0]==pointer,'Checkpoint changed while pinning target epoch'
 processes=[]
 for record in workers:
  process=psutil.Process(record['pid']);assert matches(process,record),'Refuse to signal an unverified PID';processes.append(process)
 write(status,{'stage':'requesting_graceful_pause','target_epoch':epoch,'validated_checkpoint':pointer,'validation':metrics,'archive':str(archive),'worker_pids':[p.pid for p in processes]})
 for process in processes:process.send_signal(signal.SIGUSR1)
 _,alive=psutil.wait_procs(processes,timeout=300)
 assert not [p for p in alive if p.status()!=psutil.STATUS_ZOMBIE],'Training did not exit after checkpoint-stop request'
 final=read(run/'latest.json');assert final['progress']['epoch']>=epoch and not final['progress'].get('pending_validation')
 write(status,{'stage':'paused','target_epoch':epoch,'epoch_checkpoint':pointer,'validation':metrics,'archive':str(archive),'latest_checkpoint':final,'workers_exited':True,'resume_preserves_optimizer_scheduler_rng':True})
 print(json.dumps(read(status)),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as error:
  write(STAGE/'pause-status.json',{'stage':'watcher_failed','error':str(error)});raise
