from pathlib import Path
import subprocess
GATE=r'''"""Revalidate an already completed actual-GPU readiness receipt for Slurm retention."""
import sys,json,hashlib,subprocess,os
from pathlib import Path
import torch
config=json.loads(Path(sys.argv[1]).read_text());r=Path(config['root']);name=config['recipe']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fingerprint(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
p=r/'profiles'/f'{name}.json';assert sha(p)==config['profile_sha256'],'GPU receipt changed'
receipt=json.loads(p.read_text());selection=json.loads((r/'tuning'/name/'selection.json').read_text());manifest=json.loads((r/'source-manifest.json').read_text());prepared=json.loads((r/'prepared.json').read_text());recipe=json.loads((r/'recipes'/f'{name}.json').read_text())
source={p.name:sha(p) for p in r.glob('*.py')};assert source==manifest['files']==receipt['source'];assert fingerprint(source)==manifest['source_sha256']==config['source_sha256']
assert prepared['complete']
for key,namefile in [('statistics_sha256','statistics.json'),('split_sha256','split.json'),('checkpoints_sha256','checkpoints.json'),('implementation_sha256','implementation-lock.json')]:assert sha(r/namefile)==prepared[key]
lock=json.loads((r/'implementation-lock.json').read_text())
for filename,expected in lock['files'].items():assert sha(filename)==expected
assert fingerprint(recipe)==config['recipe_fingerprint'];assert receipt['runtime']==selection['selected']['runtime'];assert receipt['identity']==selection['selected']['identity']
assert receipt['hardware']=='NVIDIA A100 80GB PCIe' and receipt['runtime']['autocast_cache_enabled'] is False and receipt['runtime']['allocator']=='expandable_segments:True'
expected=fingerprint({'recipe':recipe,'prepared':prepared,'source':source,'hardware':receipt['hardware'],'world':2,'torch':torch.__version__,'runtime':receipt['runtime']});assert expected==receipt['identity']
assert receipt['approved'] and receipt['decoded_optimizer_updates']==20
assert receipt['strict_checkpoint_rng_reload'] and receipt['strict_inference_reload'] and receipt['coordinated_checkpoint_test']
assert receipt['peak_gpu_reserved_bytes']<=.85*selection['selected']['total_gpu_bytes'] and receipt['node_process_peak_rss_bytes']<=.85*224*1024**3
account=subprocess.check_output(['sacct','-X','-j',config['gpu_profile_job_id'],'-n','-P','--format=JobIDRaw,State,ExitCode'],text=True)
rows=[line.split('|') for line in account.splitlines() if line.strip()];assert any(v[0]==config['gpu_profile_job_id'] and v[1]=='COMPLETED' and v[2]=='0:0' for v in rows)
result={'approved':True,'gpu_profile_job_id':config['gpu_profile_job_id'],'gpu_profile_state':'COMPLETED','gpu_profile_sha256':config['profile_sha256'],'identity':expected,'source_sha256':manifest['source_sha256'],'recipe':name,'actual_gpu_readiness_revalidated':True,'new_gpu_test_performed':False,'job_id':os.environ.get('SLURM_JOB_ID'),'retention_recovery':True}
if '--check-only' not in sys.argv:
 p=Path(config['receipt']);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));os.replace(tmp,p)
print(json.dumps(result),flush=True)
'''
code='GATE='+repr(GATE)+'\n'+r'''
from pathlib import Path
import json,subprocess,hashlib,os,fcntl,sys
from datetime import datetime,timezone
root=Path('/scratch/aparame/Dynamic_VLA/cross-vla-krfm-a100-pi05-cache-20261020');control=Path('/scratch/aparame/Dynamic_VLA/cross-vla-krfm-pi05-release-20261020');name='pi05_kr'
def atomic(p,v):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');os.replace(tmp,p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fingerprint(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
control.mkdir(exist_ok=True);(control/'logs').mkdir(exist_ok=True)
path=control/'verify_ready.py'
if path.exists():assert path.read_text()==GATE,'Frozen release verifier changed'
else:path.write_text(GATE)
with (root/'.submit.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);state=json.loads((root/'submissions.json').read_text());releases=control/'submissions.json';release=json.loads(releases.read_text()) if releases.exists() else {}
 assert not state.get('pending_submission') and not release.get('pending_submission'),'Uncertain submission must be reconciled'
 key=name+'_train'
 if key in state:print(json.dumps({'already_recorded':state[key]}));raise SystemExit(0)
 script=root/(key+'.sbatch');jobname='a100cache_'+name+'-trai'
 jobs=subprocess.check_output(['squeue','-u','aparame','-h','-n',jobname,'-o','%i'],text=True).split()
 for j in jobs:
  details=subprocess.check_output(['scontrol','show','job',j,'-o'],text=True);assert 'Command='+str(script) not in details,'Unrecorded production exists'
 profile=state[name+'_profile']['job_id'];p=root/'profiles'/f'{name}.json';receipt=json.loads(p.read_text());assert receipt['approved'];assert not (root/'runs'/name/'latest.pt').exists(),'Unexpected existing production checkpoint: reconcile before release'
 config={'root':str(root),'recipe':name,'gpu_profile_job_id':profile,'profile_sha256':sha(p),'source_sha256':json.loads((root/'source-manifest.json').read_text())['source_sha256'],'recipe_fingerprint':fingerprint(json.loads((root/'recipes'/f'{name}.json').read_text())),'receipt':str(control/(name+'-verified.json'))}
 configpath=control/(name+'-config.json');atomic(configpath,config)
 test=subprocess.run(['sbatch','--test-only','--dependency=afterok:'+profile,str(script)],text=True,capture_output=True)
 atomic(control/(name+'-expired-dependency.json'),{'command':['sbatch','--test-only','--dependency=afterok:'+profile,str(script)],'returncode':test.returncode,'stdout':test.stdout,'stderr':test.stderr,'checked_at':datetime.now(timezone.utc).isoformat()})
 assert test.returncode and 'dependency' in (test.stderr+test.stdout).lower(),'Expected retention-only dependency rejection'
 env=dict(os.environ,OMP_NUM_THREADS='2');env['LD_LIBRARY_PATH']='/home/aparame/.conda/envs/dynamicvla/lib:'+env.get('LD_LIBRARY_PATH','')
 check=subprocess.run([str(root/'env/bin/python'),str(path),str(configpath),'--check-only'],env=env,text=True,capture_output=True,timeout=120);atomic(control/(name+'-preflight.json'),{'returncode':check.returncode,'stdout':check.stdout,'stderr':check.stderr});assert check.returncode==0,check.stderr
 gate=control/(name+'.sbatch');gate.write_text('#!/bin/bash -l\n#SBATCH --job-name=pi05-ready-release\n#SBATCH --account=cuuser_yue6_vae_verification\n#SBATCH --partition=work1\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=2\n#SBATCH --mem=16G\n#SBATCH --time=00:12:00\n#SBATCH --output='+str(control)+'/logs/'+name+'-%j.out\n#SBATCH --error='+str(control)+'/logs/'+name+'-%j.err\nset -euo pipefail\nexport OMP_NUM_THREADS=2\nexport LD_LIBRARY_PATH=/home/aparame/.conda/envs/dynamicvla/lib:${LD_LIBRARY_PATH:-}\n'+str(root/'env/bin/python')+' '+str(path)+' '+str(configpath)+'\n')
 pre=subprocess.run(['sbatch','--test-only',str(gate)],text=True,capture_output=True);assert pre.returncode==0,pre.stderr
 release['pending_submission']={'recipe':name,'stage':'readiness_verification','script':str(gate),'attempted_at':datetime.now(timezone.utc).isoformat()};atomic(releases,release)
 result=subprocess.run(['sbatch','--parsable',str(gate)],text=True,capture_output=True,timeout=45)
 if result.returncode:
  release.setdefault('errors',[]).append({**release.pop('pending_submission'),'stderr':result.stderr});atomic(releases,release);raise RuntimeError(result.stderr)
 gatejob=result.stdout.strip().split(';')[0];assert gatejob.isdigit(),'Uncertain gate submission';release[name]={'job_id':gatejob,'gpu_profile_job_id':profile,'config':config,'script':str(gate)};release.pop('pending_submission');atomic(releases,release);print('readiness_release',gatejob,flush=True)
 state['pending_submission']={'key':key,'script':str(script),'dependency':gatejob,'gpu_profile_job_id':profile,'attempted_at':datetime.now(timezone.utc).isoformat()};atomic(root/'submissions.json',state)
 result=subprocess.run(['sbatch','--parsable','--dependency=afterok:'+gatejob,str(script)],text=True,capture_output=True,timeout=45)
 if result.returncode:
  state.setdefault('submission_errors',[]).append({**state.pop('pending_submission'),'stderr':result.stderr});atomic(root/'submissions.json',state);raise RuntimeError(result.stderr)
 job=result.stdout.strip().split(';')[0];assert job.isdigit(),'Uncertain production submission'
 record={'job_id':job,'recipe':name,'command':'train','dependency':gatejob,'gpu_profile_job_id':profile,'readiness_release_job_id':gatejob,'readiness_release_receipt':config['receipt'],'script':str(script),'gpus':2,'gpu':'A100 80 GB','submitted_at':datetime.now(timezone.utc).isoformat(),'dependency_retention_recovery':True}
 state[key]=record;state.pop('pending_submission');atomic(root/'submissions.json',state)
 recipe=json.loads((root/'recipes'/f'{name}.json').read_text());p=root/'submissions/hardware_candidates.json';candidates=json.loads(p.read_text()) if p.exists() else {};candidates[name]={**record,'world_size':2,'variant':name,'name':name+'-a100-pi05-cache','method':'pi05 + KRFM','lr':recipe['lr'],'target_epochs':None,'target_updates':recipe['steps'],'recipe_fingerprint':fingerprint(recipe)};atomic(p,candidates)
 release[name]['production_job_id']=job;atomic(releases,release);print('production',job,flush=True)
'''
r=subprocess.run(['ssh','-S','/private/tmp/vlcm-palmetto-k1vtw3i7/socket','-o','BatchMode=yes','-o','ConnectTimeout=12','aparame@slogin.palmetto.clemson.edu','python3 -'],input=code,text=True,capture_output=True,timeout=240)
print(r.stdout);print(r.stderr);print('exit',r.returncode)
Path('/private/tmp/pi05-kr-release-receipt.txt').write_text(r.stdout+'\n'+r.stderr)
