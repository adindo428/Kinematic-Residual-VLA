from pathlib import Path
import json,subprocess,hashlib
H=Path(__file__).resolve().parent;R='/scratch/aparame/Dynamic_VLA/h200-sxm-standbys-20261029';OLD='/scratch/aparame/Dynamic_VLA/h200-continuations-20261022'
SSH=['ssh','-F','/dev/null','-S','/private/tmp/vlcm-palmetto-k1vtw3i7/socket','-o','BatchMode=yes','-o','ConnectTimeout=5','aparame@slogin.palmetto.clemson.edu']
def remote(s,data=None):return subprocess.run(SSH+[s],input=data,capture_output=True,check=True,timeout=120).stdout.decode()
def persist(v):
 p=H/'standby-submissions.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
assert not (H/'standby-submissions.json').exists()
remote('test ! -e '+R+' && mkdir -p '+R+'/logs')
plan=[('pi05_baseline','17931273','17859073'),('vlash_kr','17931276','17859078')];scripts={}
for n,old,parent in plan:
 state=remote('squeue -h -j '+old+' -o "%T|%E"').strip();assert state=='PENDING|afterany:'+parent+'(unfulfilled)',state
 s=remote('cat '+OLD+'/scripts/'+n+'.sbatch');s=s.replace('#SBATCH --job-name=h2-cont-'+n,'#SBATCH --job-name=sxm-cont-'+n).replace('#SBATCH --constraint=gpu_h200','#SBATCH --constraint=gpu_h200&chip_manufacturer_intel')
 s=s.replace(OLD+'/logs/'+n+'-%j',R+'/logs/'+n+'-%j')
 if n=='vlash_kr':
  expected='exec bash /scratch/aparame/Dynamic_VLA/main8-replicas-h200-20261021/scripts/vlash_kr.sbatch'
  assert expected in s;s=s.replace(expected,'exec bash /scratch/aparame/Dynamic_VLA/h200-microbatch-20261022/vlash_kr/scripts/vlash_kr.sbatch')
 scripts[n]=s;(H/('standby-'+n+'.sbatch')).write_text(s);remote('cat > '+R+'/'+n+'.sbatch',s.encode());pre=remote('bash -n '+R+'/'+n+'.sbatch && sbatch --test-only --dependency=afterany:'+parent+' '+R+'/'+n+'.sbatch 2>&1');(H/('standby-'+n+'-preflight.txt')).write_text(pre)
state={}
for n,old,parent in plan:
 q=remote('squeue -h -j '+old+' -o "%T|%E"').strip();assert q=='PENDING|afterany:'+parent+'(unfulfilled)',q
 state[n]={'old_job_id':old,'parent_job_id':parent,'dependency':'afterany:'+parent,'script':R+'/'+n+'.sbatch','script_sha256':hashlib.sha256(scripts[n].encode()).hexdigest(),'state':'prepared'};persist(state)
 remote('scancel '+old);gone=remote('squeue -h -j '+old+' -o "%T" 2>/dev/null || true').strip();assert gone in ('','CANCELLED','COMPLETING'),gone
 state[n]['state']='old_pending_cancelled';persist(state)
 state[n]['state']='submission_intent';persist(state);remote('cat > '+R+'/submissions.json',json.dumps(state).encode())
 out=remote('sbatch --parsable --dependency=afterany:'+parent+' '+R+'/'+n+'.sbatch').strip();j=out.split(';')[0];assert j.isdigit(),'Uncertain submission '+out
 state[n].update(state='submitted',job_id=j);persist(state);remote('cat > '+R+'/submissions.json',json.dumps(state).encode());print(n,j,flush=True)
