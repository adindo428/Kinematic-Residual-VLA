"""Install isolated, memory-admitted continuations; preserve live trainers."""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
BASE=Path('/scratch/aparame/Dynamic_VLA')
CONTROL=BASE/'h200-continuations-20261022'
DEST=BASE/'h200-microbatch-20261022'
PLAN={'vlash_baseline':('main8-replicas-h200-20261021',16,'17931275'),'vlash_kr':('main8-replicas-h200-20261021',16,'17931276'),'domv100':('main8-h200-only-20261022',44,'17931280')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,obj):
 p=Path(p);tmp=p.with_name(p.name+'.install-tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,p)
def replace_once(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
assert not DEST.exists(),'Destination already exists; inspect before retrying'
# Stage everything before touching launch points.
DEST.mkdir()
receipts={}
for name,(family,micro,job) in PLAN.items():
 old=BASE/family;new=DEST/name;new.mkdir()
 specpath=CONTROL/'specs'/f'{name}.json';spec=json.loads(specpath.read_text())
 for p,h in spec['hashes'].items():assert sha(p)==h,p
 for p in old.iterdir():
  if p.name in ['__pycache__','runs','profiles','logs','scripts','submissions','wandb-config']:continue
  q=new/p.name
  if p.is_dir():
   if p.name=='runtimes':shutil.copytree(p,q)
   else:q.symlink_to(p,target_is_directory=True)
  else:shutil.copy2(p,q)
 for d in ['runs','profiles','logs','scripts','wandb-config']:(new/d).mkdir()
 out=new/'runs'/name;out.mkdir()
 for fn in ['latest.pt','latest.json','wandb.json']:
  assert (old/'runs'/name/fn).exists(),fn
  (out/fn).symlink_to(old/'runs'/name/fn)
 origin={'root':str(old),'name':name,'identity':json.loads((old/'runs'/name/'latest.json').read_text())['identity'],'checkpoint':str(old/'runs'/name/'latest.pt'),'pending_job':job,'microbatch_before':json.loads((old/'profiles'/f'{name}.json').read_text())['runtime']['microbatch'],'microbatch_after':micro}
 atomic(new/'continuation-origin.json',origin)
 runtime=json.loads((new/'runtimes'/f'{name}.json').read_text());runtime['microbatch']=micro;atomic(new/'runtimes'/f'{name}.json',runtime)
 train=(new/'train.py').read_text()
 helper='''def continuation_parent_identity(checkpoint, identity):
    """Only the exact terminal parent's symlink may use the prior identity."""
    origin = json.loads((Path(__file__).parent/'continuation-origin.json').read_text())
    if checkpoint.is_symlink():
        require(checkpoint.resolve() == Path(origin['checkpoint']).resolve(), 'Wrong continuation checkpoint')
        metadata = json.loads(checkpoint.with_suffix('.json').read_text())
        require(metadata['identity'] == origin['identity'], 'Parent identity changed')
        return origin['identity']
    return identity


'''
 train=replace_once(train,'def main():\n',helper+'def main():\n')
 train=replace_once(train,"        progress = restore_parent(root, recipe, policy, optimizer, scheduler, rank, restore_rng)\n        # Compare decoded", "        progress = load(out/'latest.pt', policy, optimizer, scheduler, identity, rank,\n                        parent_identity=continuation_parent_identity(out/'latest.pt', identity))\n        validate_progress(progress, recipe)\n        # Compare decoded")
 train=replace_once(train,'        progress = load(checkpoint, policy, optimizer, scheduler, identity, rank)\n', '        progress = load(checkpoint, policy, optimizer, scheduler, identity, rank,\n                        parent_identity=continuation_parent_identity(checkpoint, identity))\n')
 # If a candidate run has lost its checkpoint, never restart from the old budget origin.
 train=replace_once(train,"    else:\n        require(not any((out/name).exists()", "    else:\n        raise ValueError('Continuation checkpoint missing; refusing parent reset')\n        require(not any((out/name).exists()")
 (new/'train.py').write_text(train)
 compile(train,str(new/'train.py'),'exec')
 manifest=json.loads((new/'source-manifest.json').read_text());manifest['files']={p.name:sha(p) for p in new.glob('*.py') if not p.name.startswith('._')};atomic(new/'source-manifest.json',manifest)
 prepared=json.loads((new/'prepared-palmetto.json').read_text())
 prepared['files']={p:sha(new/p) for p in prepared['files']}
 prepared['files']['continuation-origin.json']=sha(new/'continuation-origin.json')
 prepared['source_sha256']=sha(new/'source-manifest.json');atomic(new/'prepared-palmetto.json',prepared)
 oldscript=Path(spec['original_script']);backup=new/'scripts'/'original.sbatch';shutil.copy2(oldscript,backup)
 launcher=backup.read_text().replace(str(old),str(new))
 # A clean profile failure may fall back; never fall back after new training is admitted.
 fallback=f'''if [[ $status -ne 0 ]] && [[ ! -f "{new}/profiles/{name}.json" ]] && [[ ! -f "{new}/runs/{name}/provenance.json" ]]; then
  echo "Candidate readiness failed; resuming unchanged approved runtime" >&2
  exec bash "{backup}"
fi
if [[ $status -ne 0 ]] && [[ -f "{new}/profiles/{name}.json" ]] && [[ ! -f "{new}/runs/{name}/provenance.json" ]]; then
  if /scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/env/bin/python -c 'import json,sys; sys.exit(0 if not json.load(open(sys.argv[1])).get("approved") else 1)' "{new}/profiles/{name}.json"; then
    echo "Candidate exceeded readiness limits; resuming unchanged approved runtime" >&2
    exec bash "{backup}"
  fi
fi
'''
 launcher=replace_once(launcher,'exit "$status"\n',fallback+'exit "$status"\n')
 candidate=new/'scripts'/f'{name}.sbatch';candidate.write_text(launcher)
 subprocess.run(['bash','-n',str(candidate)],check=True)
 dispatch=f'''# Only the queued continuation uses the isolated microbatch candidate.
if [[ "${{SLURM_JOB_ID:-}}" == "{job}" ]]; then
  exec bash "{candidate}"
fi
'''
 dispatched=replace_once(backup.read_text(),'set -euo pipefail\n','set -euo pipefail\n'+dispatch)
 staged=new/'scripts'/'dispatch.sbatch';staged.write_text(dispatched)
 subprocess.run(['bash','-n',str(staged)],check=True)
 receipts[name]={'job':job,'root':str(new),'old_script':str(oldscript),'spec':str(specpath),'old_hash':sha(oldscript),'dispatch':str(staged),'microbatch':micro,'accumulation':json.loads((new/'recipes'/f'{name}.json').read_text())['global_batch']//(2*micro),'global_batch':json.loads((new/'recipes'/f'{name}.json').read_text())['global_batch']}
# Hold only the three dependent jobs during atomic launch-point updates.
jobs=[v['job'] for v in receipts.values()]
for job in jobs:
 state=subprocess.check_output(['squeue','-h','-j',job,'-o','%T'],text=True).strip();assert state=='PENDING',(job,state)
subprocess.run(['scontrol','hold',','.join(jobs)],check=True)
for name,r in receipts.items():
 p=Path(r['old_script']);assert sha(p)==r['old_hash']
 specpath=Path(r['spec']);spec=json.loads(specpath.read_text());shutil.copy2(specpath,DEST/f'{name}-original-spec.json')
 tmp=p.with_name(p.name+'.install-tmp');shutil.copy2(r['dispatch'],tmp);os.replace(tmp,p)
 spec['hashes'][str(p)]=sha(p)
 # Include staged runtime/source/config hashes in the existing admission guard.
 new=Path(r['root'])
 for f in [*new.glob('*.py'),new/'prepared-palmetto.json',new/'source-manifest.json',new/'continuation-origin.json',new/'runtimes'/f'{name}.json',* (new/'scripts').glob('*.sbatch')]:spec['hashes'][str(f)]=sha(f)
 atomic(specpath,spec)
 for f,h in spec['hashes'].items():assert sha(f)==h,f
atomic(DEST/'deployment.json',receipts)
subprocess.run(['scontrol','release',','.join(jobs)],check=True)
print(json.dumps(receipts,indent=2))
