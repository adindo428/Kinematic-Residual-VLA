from pathlib import Path
import subprocess,json,hashlib,shlex,sys
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
CONTROL='/scratch/aparame/Dynamic_VLA/h200-continuations-20261022'
PY='/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/env/bin/python'
SSH=['ssh','-F','/dev/null','-S','/private/tmp/vlcm-palmetto-k1vtw3i7/socket','-o','BatchMode=yes','-o','ConnectTimeout=5','aparame@slogin.palmetto.clemson.edu']
JOBS=[('pi05_baseline','17859073'),('pi05_kr','17859076'),('vlash_baseline','17859077'),('vlash_kr','17859078'),('smolvla_baseline','17912900'),('smolvla_kr','17912901'),('domv100','17912902')]
def remote(cmd,data=None):return subprocess.run(SSH+[cmd],input=data,capture_output=True,check=True,timeout=180).stdout.decode()
def persist(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def prepare():
 assert not (HERE/'prepared.json').exists()
 q=remote('squeue -u aparame -h -o "%i|%j|%T|%E"');assert 'h2-cont-' not in q
 (HERE/'queue-before.txt').write_text(q)
 remote('test ! -e '+CONTROL+' && mkdir -p '+CONTROL+'/scripts '+CONTROL+'/specs '+CONTROL+'/logs '+CONTROL+'/receipts')
 for f in ('guard.py','test_guard.py'):remote('cat > '+CONTROL+'/'+f,(HERE/f).read_bytes())
 result=remote('cd '+CONTROL+' && '+PY+' -m unittest -v 2>&1');(HERE/'remote-tests.txt').write_text(result)
 specs={}
 for n,j in JOBS:
  root='/scratch/aparame/Dynamic_VLA/'+('main8-replicas-h200-20261021' if n.startswith(('pi05','vlash')) else 'main8-h200-only-20261022')
  original=root+'/scripts/'+n+'.sbatch';body=remote('cat '+original)
  snapshot=remote("python3 - "+shlex.quote(root)+' '+shlex.quote(original)+" <<'SNAP'\nimport sys,json,hashlib\nfrom pathlib import Path\nr=Path(sys.argv[1]);ps=list(r.glob('*.py'))+[Path(sys.argv[2]),r/'recipes'/('"+n+".json'),r/'profiles'/('"+n+".json')]\nprint(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in ps}))\nSNAP")
  hashes=json.loads(snapshot);hashes[CONTROL+'/guard.py']=hashlib.sha256((HERE/'guard.py').read_bytes()).hexdigest()
  pname=[l.split('=',1)[1] for l in body.splitlines() if l.startswith('#SBATCH --job-name=')][0]
  spec={'name':n,'parent':j,'root':root,'control':CONTROL,'parent_job_name':pname,'job_name':'h2-cont-'+n,'hashes':hashes,'original_script':original}
  specs[n]=spec;persist(HERE/(n+'.json'),spec);remote('cat > '+CONTROL+'/specs/'+n+'.json',json.dumps(spec).encode())
  headers=[l for l in body.splitlines() if l.startswith('#SBATCH')]
  headers=[('#SBATCH --job-name='+spec['job_name']) if l.startswith('#SBATCH --job-name=') else ('#SBATCH --output='+CONTROL+'/logs/'+n+'-%j.out') if l.startswith('#SBATCH --output=') else ('#SBATCH --error='+CONTROL+'/logs/'+n+'-%j.err') if l.startswith('#SBATCH --error=') else l for l in headers]
  python='/home/aparame/.conda/envs/dynamicvla/bin/python' if n=='domv100' else PY
  wrapper='#!/usr/bin/env bash\n'+'\n'.join(headers)+'\nset -euo pipefail\nexport OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\nexec 9>'+CONTROL+'/'+n+'.lock\nflock -n 9 || exit 1\nset +e\n'+python+' '+CONTROL+'/guard.py --spec '+CONTROL+'/specs/'+n+'.json\nstatus=$?\nset -e\nif [[ $status -eq 10 ]]; then exit 0; fi\n[[ $status -eq 0 ]] || exit "$status"\nexec bash '+original+'\n'
  (HERE/(n+'.sbatch')).write_text(wrapper);remote('cat > '+CONTROL+'/scripts/'+n+'.sbatch',wrapper.encode())
  out=remote('bash -n '+CONTROL+'/scripts/'+n+'.sbatch && sbatch --test-only --dependency=afterany:'+j+' '+CONTROL+'/scripts/'+n+'.sbatch 2>&1');(HERE/(n+'-preflight.txt')).write_text(out)
 persist(HERE/'prepared.json',specs)
 # Inspect every current own checkpoint without modifying training state.
 code="""import sys,json,gc
from pathlib import Path
import torch
from guard import validate
for f in sorted(Path('specs').glob('*.json')):
 s=json.loads(f.read_text());r=Path(s['root']);o=r/'runs'/s['name'];m=json.load(open(o/'latest.json'));c=torch.load(o/'latest.pt',map_location='cpu',mmap=True,weights_only=False)
 done=validate(c,m,json.load(open(r/'recipes'/(s['name']+'.json'))),json.load(open(o/'provenance.json')))
 print(json.dumps({'name':s['name'],'valid':True,'completed':done,'progress':c['progress']}),flush=True)
 del c;gc.collect()
"""
 out=remote('cd '+CONTROL+' && '+PY+' -',code.encode());(HERE/'checkpoint-preflight.jsonl').write_text(out);print(out)
def submit():
 specs=json.loads((HERE/'prepared.json').read_text());assert (HERE/'checkpoint-preflight.jsonl').exists()
 statepath=HERE/'submissions.json';state=json.loads(statepath.read_text()) if statepath.exists() else {}
 for n,j in JOBS:
  if n in state:assert state[n].get('job_id'),'Uncertain submission; reconcile';continue
  q=remote('squeue -u aparame -h -o "%i|%j|%T|%E"');assert '|h2-cont-'+n+'|' not in q,'Unrecorded existing continuation'
  script=CONTROL+'/scripts/'+n+'.sbatch';actual=remote('sha256sum '+script).split()[0];assert actual==hashlib.sha256((HERE/(n+'.sbatch')).read_bytes()).hexdigest()
  state[n]={'parent_job_id':j,'dependency':'afterany:'+j,'script':script,'script_sha256':actual,'state':'intent','at':datetime.now(timezone.utc).isoformat()};persist(statepath,state)
  remote('cat > '+CONTROL+'/submissions.json',json.dumps(state).encode())
  response=remote('sbatch --parsable --dependency=afterany:'+j+' '+script).strip();job=response.split(';')[0];assert job.isdigit(),'Uncertain submission '+response
  state[n].update(job_id=job,state='submitted');persist(statepath,state)
  remote('cat > '+CONTROL+'/submissions.json',json.dumps(state).encode());print(n,job,flush=True)
if __name__=='__main__':{'prepare':prepare,'submit':submit}[sys.argv[1]]()
