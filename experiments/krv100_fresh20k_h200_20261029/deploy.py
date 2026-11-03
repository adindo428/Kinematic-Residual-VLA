"""Deploy immutable fresh KRV source, preflight, then persist gated job IDs."""
import subprocess,json,hashlib,shlex
from pathlib import Path
from datetime import datetime,timezone
H=Path(__file__).resolve().parent;R='/scratch/aparame/Dynamic_VLA/krv100-fresh20k-h200-20261029';P='/home/aparame/.conda/envs/dynamicvla/bin/python'
SSH=['ssh','-F','/dev/null','-S','/private/tmp/vlcm-palmetto-k1vtw3i7/socket','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=1','aparame@slogin.palmetto.clemson.edu']
def remote(cmd,data=None,timeout=180):return subprocess.run(SSH+[cmd],input=data,capture_output=True,check=True,timeout=timeout).stdout.decode()
def persist(p,v):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def main():
 assert not (H/'submissions.json').exists(),'Existing or uncertain submission; reconcile'
 queue=remote('squeue -u aparame -h -o "%i|%j|%T"');assert 'fresh20k-krv-' not in queue
 (H/'queue-before.txt').write_text(queue)
 remote('test ! -e '+R+' && mkdir -p '+R+'/logs '+R+'/scripts')
 for name in ('stage.py','entry.py','cpu_preflight.py','launch.py'):
  remote('cat > '+R+'/'+name,(H/name).read_bytes())
 remote('cat > '+R+'/pc-reference.json',(H.parents[1]/'maintenance/restart_repair_20261029/new-workstation-wandb.json').read_bytes())
 staged=remote('OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+P+' '+R+'/stage.py',timeout=240);(H/'staged.json').write_text(staged)
 (H/'source-manifest.json').write_text(remote('cat '+R+'/source-manifest.json'))
 base='''#!/usr/bin/env bash
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
'''
 env='''set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH=/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/lib64:/home/aparame/.conda/envs/dynamicvla/lib:${LD_LIBRARY_PATH:-}
cd '''+R+'''
export PYTHONPATH="$PWD/source"
export WANDB_CONFIG_DIR="$PWD/wandb-config"
mkdir -p "$WANDB_CONFIG_DIR"
secret=/home/aparame/.config/kinematic-residual-vla/secrets.env
if [[ -f "$secret" && ! -L "$secret" && $(stat -c %a "$secret") == 600 ]]; then set -a; source "$secret"; set +a; fi
'''
 scripts={}
 for mode in ('setup','profile','train'):
  body=base+'#SBATCH --job-name=fresh20k-krv-'+mode+'\n#SBATCH --cpus-per-task='+('4' if mode=='setup' else '32')+'\n#SBATCH --mem='+('64G' if mode=='setup' else '256G')+'\n#SBATCH --time='+('01:00:00' if mode=='setup' else '02:00:00' if mode=='profile' else '48:00:00')+'\n#SBATCH --output='+R+'/logs/'+mode+'-%j.out\n#SBATCH --error='+R+'/logs/'+mode+'-%j.err\n'
  if mode!='setup':body+='#SBATCH --gpus-per-node=h200:2\n#SBATCH --constraint=gpu_h200&chip_manufacturer_intel\n'
  if mode=='train':body+='#SBATCH --signal=B:USR1@1200\n#SBATCH --requeue\n'
  body+=env
  if mode=='setup':body+=P+' cpu_preflight.py\n'
  else:
   body+='runner=\nforward_stop() { [[ -z "$runner" ]] || kill -USR1 "$runner" || true; }\ntrap forward_stop USR1 TERM\n'+P+' launch.py '+mode+' &\nrunner=$!\nset +e\nwhile true; do wait "$runner"; status=$?; if ! kill -0 "$runner" 2>/dev/null; then break; fi; done\nset -e\n'
   if mode=='train':body+='if [[ $status -eq 75 ]]; then scontrol requeue "$SLURM_JOB_ID"; exit 0; fi\n'
   body+='exit "$status"\n'
  scripts[mode]=body;(H/(mode+'.sbatch')).write_text(body);remote('cat > '+R+'/scripts/'+mode+'.sbatch',body.encode())
  pre=remote('bash -n '+R+'/scripts/'+mode+'.sbatch && sbatch --test-only '+R+'/scripts/'+mode+'.sbatch 2>&1');(H/(mode+'-preflight.txt')).write_text(pre)
 # Freeze control files before actual submission. Readiness identity separately hashes every native Python module.
 manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in H.iterdir() if f.suffix in ('.py','.sbatch')};persist(H/'control-manifest.json',manifest);remote('cat > '+R+'/control-manifest.json',json.dumps(manifest).encode())
 state={}
 def submit(mode,parent=None):
  state[mode]={'state':'intent','script':R+'/scripts/'+mode+'.sbatch','script_sha256':hashlib.sha256(scripts[mode].encode()).hexdigest(),'dependency':('afterok:'+parent) if parent else None,'at':datetime.now(timezone.utc).isoformat()};persist(H/'submissions.json',state);remote('cat > '+R+'/submissions.json',json.dumps(state).encode())
  raw=remote('sbatch --parsable '+('--dependency=afterok:'+parent+' ' if parent else '')+state[mode]['script']).strip();jid=raw.split(';')[0];assert jid.isdigit(),'Uncertain submission: '+raw
  state[mode].update(state='submitted',job_id=jid);persist(H/'submissions.json',state);remote('cat > '+R+'/submissions.json',json.dumps(state).encode());print(mode,jid,flush=True);return jid
 setup=submit('setup');profile=submit('profile',setup);train=submit('train',profile)
 candidates={'krv100':{'cluster':'Palmetto','family':R+'/family','root':R,'job_id':train,'profile_job_id':profile,'setup_job_id':setup,'gpu':'H200 SXM','gpu_count':2,'variant':'krv100','fresh_20k':True,'epochs':10,'total_updates':68380,'run_directory':R+'/family/runs/krv100','wandb_id':None}}
 persist(H/'hardware_candidates.json',candidates);remote('mkdir -p '+R+'/submissions && cat > '+R+'/submissions/hardware_candidates.json',json.dumps(candidates).encode())
 (H/'launch-verification.txt').write_text(remote('squeue -j '+','.join([setup,profile,train])+' -o "%.18i %.30j %.15T %.30R"; sacct -X -j '+','.join([setup,profile,train])+' --format=JobID,State,ExitCode -P'))
if __name__=='__main__':main()
