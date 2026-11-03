import json,subprocess,shlex
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
SSH=['ssh','-S','/tmp/deltaai-kinematic.sock','-o','BatchMode=yes','aparameshwaran@gh-login01.delta.ncsa.illinois.edu']
REMOTE='/work/nvme/bivv/aparameshwaran/dynamicvla/setup-domv-resume-20261102'
receipt=HERE/'submission.json'
assert not receipt.exists(),'Reconcile existing submission before retrying'
r=subprocess.run(SSH+['sbatch --hold --parsable '+REMOTE+'/production.sbatch'],text=True,capture_output=True)
s={'submitted_at':datetime.now(timezone.utc).isoformat(),'stdout':r.stdout,'stderr':r.stderr,'returncode':r.returncode}
receipt.write_text(json.dumps(s,indent=2)+'\n')
assert r.returncode==0 and r.stdout.strip().split(';')[0].isdigit(),s
job=r.stdout.strip().split(';')[0];s['job_id']=job;receipt.write_text(json.dumps(s,indent=2)+'\n')
code='''import json,hashlib,subprocess,shutil
from pathlib import Path
job=JOB
launch=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/setup-domv-resume-20261102')
root=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/domv-scale-ddp-20261014')
source=Path('/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261012-video-clean/dom100')
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
control=json.loads(Path('/work/nvme/bivv/aparameshwaran/dynamicvla/setup-epoch-cap-20261019/control.json').read_text())
control.pop('fingerprint');control['jobs']={job:control['jobs']['3381947']}
control['authorization']='2026-11-02 user requested full-dataset DOM+velocity continuation on two nodes x four GH200; preserve prior ten-total-epoch cap and original scheduler'
control['bootstrap_sha256']=sha(launch/'sitecustomize.py');control['fingerprint']=digest(control)
(launch/'control.json').write_text(json.dumps(control,indent=2)+'\\n')
expected={'core/dom_checkpoint.py':'2f187f78b37bbb9431d33b2139362ecfdbefe0de38e2a6230546e609ea8adf18','core/dom_distributed_training.py':'336c1ecd5c7f2883cb1dc8d5cbb9934cda983d72fff62ef783e1f91d2263cd73','core/dom_distributed.py':'cbcb2c77602c278978be6027929133186cb89023ad10cf51eaeb6d4795cd81f7'}
for name,h in expected.items():assert sha(source/name)==h,name
run=root/'runs/domv100';backup=launch/'starting-checkpoints';backup.mkdir()
for pointer in ['latest.json','best.json']:
 value=json.loads((run/pointer).read_text());generation=value['generation'];dest=backup/generation
 if not dest.exists():subprocess.check_call(['cp','-al',str(run/'generations'/generation),str(dest)])
 shutil.copy2(run/pointer,backup/pointer)
manifest=json.loads((root/'manifest.json').read_text());recipe=json.loads((root/'recipe.json').read_text());runtime=json.loads((root/'selected-runtime.json').read_text())
lock={'files':{n:sha(launch/n) for n in ['production.sbatch','preflight.py','sitecustomize.py','control.json']},'manifest':manifest['fingerprint'],'recipe':recipe['fingerprint'],'runtime':runtime['fingerprint'],'checkpoint_sources':expected,'retention_verified':{'latest':True,'best_validation':True,'finite_metric':True,'ties_keep_previous':True,'immutable_generations':True,'pruning_protects_latest_and_best':True,'exact_latest_resume':True},'initial_checkpoints_preserved':True,'world_size':8,'node_count':2,'operational_total_epochs':10,'scheduler_horizon_epochs':100}
(launch/'launch-lock.json').write_text(json.dumps(lock,indent=2)+'\\n')
subprocess.check_call(['scontrol','release',job])
queue=subprocess.check_output(['squeue','-h','-j',job,'-o','%i|%u|%a|%j|%T|%R'],universal_newlines=True)
print(json.dumps({'job_id':job,'released':True,'queue':queue.strip(),'lock':lock}))
'''.replace('JOB',repr(job))
r=subprocess.run(SSH+['python3 -c '+shlex.quote(code)],text=True,capture_output=True)
(HERE/'freeze-release.log').write_text(r.stdout+r.stderr)
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
v=json.loads(r.stdout);s.update(released=True,queue=v['queue']);receipt.write_text(json.dumps(s,indent=2)+'\n')
(HERE/'launch-lock.json').write_text(json.dumps(v['lock'],indent=2)+'\n')
print(json.dumps(s,indent=2))
