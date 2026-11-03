"""Requeue two terminal failed continuations on the exact approved NVL pool."""
import json,subprocess,hashlib,shlex
from pathlib import Path
from datetime import datetime,timezone
H=Path(__file__).resolve().parent;R='/scratch/aparame/Dynamic_VLA/restart-repair-20261029-r2';P='/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/env/bin/python';B='/scratch/aparame/Dynamic_VLA/'
SSH=['ssh','-F','/dev/null','-S','/private/tmp/vlcm-palmetto-k1vtw3i7/socket','-o','BatchMode=yes','-o','ConnectTimeout=5','aparame@slogin.palmetto.clemson.edu']
def remote(cmd,data=None):return subprocess.run(SSH+[cmd],input=data,capture_output=True,check=True,timeout=120).stdout.decode()
def persist(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def main():
 assert not (H/'submissions.json').exists(),'Recorded/uncertain submissions; reconcile first'
 queue=remote('squeue -u aparame -h -o "%i|%j|%T"');assert 'nvlfix-' not in queue
 for name in ('main8-rep-h200-pi05_kr','h2-cont-pi05_kr','main8-rep-h200-vlash_baseline','h2-cont-vlash_baseline'):assert '|'+name+'|' not in queue,'Affected run still active or queued'
 (H/'queue-before.txt').write_text(queue)
 remote('test ! -e '+R+' && mkdir -p '+R+'/scripts '+R+'/logs')
 for f in ('checkpoint_contract.py','verify.py','test_guard.py'):remote('cat > '+R+'/'+f,(H/f).read_bytes())
 audit=json.load(open(H/'identity-audit.json'));specs={};scripts={}
 for n,a in audit.items():
  root=a['root'];original=root+'/scripts/'+n+'.sbatch';body=remote('cat '+original)
  data=remote("python3 - "+shlex.quote(root)+' '+shlex.quote(original)+" <<'SNAP'\nfrom pathlib import Path\nimport hashlib,json,sys\nr=Path(sys.argv[1]);files=[*r.glob('*.py'),r/'recipes'/'"+n+".json',r/'profiles'/'"+n+".json',r/'prepared-palmetto.json',r/'source-manifest.json',r/'split.json',Path(sys.argv[2])]\nprint(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))\nSNAP")
  specs[n]={'root':root,'name':n,'identity':a['approval_identity'],'saved_updates':a['saved']['progress']['updates'],'files':json.loads(data),'conflicting_names':['main8-rep-h200-'+n,'h2-cont-'+n,'nvlfix-'+n]}
  headers=[line for line in body.splitlines() if line.startswith('#SBATCH')]
  headers=[('#SBATCH --job-name=nvlfix-'+n) if l.startswith('#SBATCH --job-name=') else ('#SBATCH --constraint=gpu_h200&graphics') if l.startswith('#SBATCH --constraint=') else ('#SBATCH --output='+R+'/logs/'+n+'-%j.out') if l.startswith('#SBATCH --output=') else ('#SBATCH --error='+R+'/logs/'+n+'-%j.err') if l.startswith('#SBATCH --error=') else l for l in headers]
  headers.append('#SBATCH --exclude=node0102')
  wrapper='#!/usr/bin/env bash\n'+'\n'.join(headers)+'\nset -euo pipefail\nexport OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\nexport LD_LIBRARY_PATH=/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/lib64:/home/aparame/.conda/envs/dynamicvla/lib:${LD_LIBRARY_PATH:-}\nset +e\n'+P+' '+R+'/verify.py --name '+n+' --gpu\nstatus=$?\nset -e\n[[ $status -eq 10 ]] && exit 0\n[[ $status -eq 0 ]] || exit "$status"\nexec bash '+original+'\n'
  scripts[n]=wrapper
 persist(H/'specs.json',specs);remote('cat > '+R+'/specs.json',json.dumps(specs).encode())
 tests=remote('cd '+R+' && '+P+' -m unittest -v 2>&1');(H/'remote-contracts.txt').write_text(tests)
 cpu='''#!/usr/bin/env bash
#SBATCH --job-name=nvlfix-cpu-verify
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output='''+R+'''/logs/verify-%j.out
#SBATCH --error='''+R+'''/logs/verify-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd '''+R+'\n'+P+' verify.py\n'
 scripts={'verification':cpu,**scripts}
 for n,body in scripts.items():
  (H/(n+'.sbatch')).write_text(body);remote('cat > '+R+'/scripts/'+n+'.sbatch',body.encode());pre=remote('bash -n '+R+'/scripts/'+n+'.sbatch && sbatch --test-only '+R+'/scripts/'+n+'.sbatch 2>&1');(H/(n+'-preflight.txt')).write_text(pre)
 manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in H.iterdir() if f.name in ('verify.py','checkpoint_contract.py','specs.json','test_guard.py') or f.suffix=='.sbatch'};persist(H/'control-manifest.json',manifest);remote('cat > '+R+'/control-manifest.json',json.dumps(manifest).encode())
 state={}
 def submit(n,parent=None):
  state[n]={'state':'intent','script':R+'/scripts/'+n+'.sbatch','script_sha256':hashlib.sha256(scripts[n].encode()).hexdigest(),'dependency':'afterok:'+parent if parent else None,'at':datetime.now(timezone.utc).isoformat()};persist(H/'submissions.json',state);remote('cat > '+R+'/submissions.json',json.dumps(state).encode())
  raw=remote('sbatch --parsable '+('--dependency=afterok:'+parent+' ' if parent else '')+state[n]['script']).strip();j=raw.split(';')[0];assert j.isdigit(),'Uncertain submission: '+raw
  state[n].update(state='submitted',job_id=j);persist(H/'submissions.json',state);remote('cat > '+R+'/submissions.json',json.dumps(state).encode());print(n,j,flush=True);return j
 gate=submit('verification')
 for n in specs:submit(n,gate)
 candidates={n:{**state[n],'cluster':'Palmetto','variant':n,'root':s['root'],'family':s['root'],'run_directory':s['root']+'/runs/'+n,'gpu':'H200 NVL','gpu_count':2,'world_size':2,'resume_from_updates':s['saved_updates'],'conditional_no_new_budget':True} for n,s in specs.items()};persist(H/'hardware_candidates.json',candidates);remote('mkdir -p '+R+'/submissions && cat > '+R+'/submissions/hardware_candidates.json',json.dumps(candidates).encode())
if __name__=='__main__':main()
