import json,hashlib,subprocess,os,shutil
from pathlib import Path
B=Path('/scratch/aparame/Dynamic_VLA');D=B/'h200-microbatch-20261022';C=B/'h200-continuations-20261022'
plan={'vlash_baseline':('17931275','17859077'),'vlash_kr':('17931276','17859078'),'domv100':('17931280','17912902')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,v):
 t=p.with_name(p.name+'.install-tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def check(job,parent):
 s=subprocess.check_output(['squeue','-h','-j',job,'-o','%T|%E'],text=True).strip()
 assert s=='PENDING|afterany:'+parent+'(unfulfilled)',s
 assert subprocess.check_output(['squeue','-h','-j',parent,'-o','%T'],text=True).strip()=='RUNNING'
receipts={}
for name,(job,parent) in plan.items():
 check(job,parent);new=D/name;p=C/'specs'/f'{name}.json';s=json.loads(p.read_text())
 for f,h in s['hashes'].items():assert sha(f)==h,f
 oldscript=Path(s['original_script']);assert sha(oldscript)==sha(new/'scripts'/'original.sbatch')
 shutil.copy2(p,D/f'{name}-original-spec.json')
 s['hashes'][str(oldscript)]=sha(new/'scripts'/'dispatch.sbatch')
 for f in [*new.glob('*.py'),new/'prepared-palmetto.json',new/'source-manifest.json',new/'continuation-origin.json',new/'runtimes'/f'{name}.json',*(new/'scripts').glob('*.sbatch')]:s['hashes'][str(f)]=sha(f)
 check(job,parent)
 tmp=oldscript.with_name(oldscript.name+'.install-tmp');shutil.copy2(new/'scripts'/'dispatch.sbatch',tmp);os.replace(tmp,oldscript)
 atomic(p,s)
 for f,h in s['hashes'].items():assert sha(f)==h,f
 check(job,parent)
 rt=json.loads((new/'runtimes'/f'{name}.json').read_text());recipe=json.loads((new/'recipes'/f'{name}.json').read_text())
 receipts[name]={'job':job,'parent':parent,'root':str(new),'microbatch':rt['microbatch'],'accumulation':recipe['global_batch']//(2*rt['microbatch']),'global_batch':recipe['global_batch'],'source_and_guard_hashes_verified':True,'dependency':'afterany:'+parent,'active_parent_unmodified':True}
atomic(D/'deployment.json',receipts)
print(json.dumps(receipts,indent=2))
