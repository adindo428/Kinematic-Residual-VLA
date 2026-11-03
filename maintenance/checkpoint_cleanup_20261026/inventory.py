from pathlib import Path
import json,os,subprocess
B=Path('/scratch/aparame/Dynamic_VLA');rows=[];links=[]
for family in sorted(B.iterdir()):
 if not family.is_dir() or family.is_symlink() or not family.name.startswith(('main8-','h200-microbatch-','cross-vla-krfm-')):continue
 def walk(d,depth=0):
  if depth>6:return
  for p in d.iterdir():
   if p.is_symlink():
    if p.suffix in ('.pt','.safetensors') or p.name=='parents':links.append({'path':str(p),'target':str(p.resolve())})
    continue
   if p.is_file() and p.suffix in ('.pt','.safetensors'):
    s=p.stat();m=p.with_suffix('.json');meta=json.loads(m.read_text()) if m.exists() else None
    rows.append({'path':str(p),'bytes':s.st_size,'inode':s.st_ino,'mtime':s.st_mtime,'metadata':meta,'family':str(family)})
   elif p.is_dir() and p.name not in ('env','upstream-vlash','upstream','wandb','wandb-config','videos','data','.git','__pycache__','checkpoint','checkpoints') and not p.name.startswith('.'):walk(p,depth+1)
 for folder in ('runs','parents','profiles','tuning','readiness','benchmarks'):
  d=family/folder
  if d.is_dir() and not d.is_symlink():walk(d)
 if family.name.startswith('h200-microbatch-'):
  for sub in family.iterdir():
   if sub.is_dir() and not sub.is_symlink():
    for folder in ('runs','parents','profiles','tuning'):
     d=sub/folder
     if d.is_dir() and not d.is_symlink():walk(d)
print(json.dumps({'files':rows,'symlinks':links,'queue':subprocess.check_output(['squeue','-u','aparame','-h','-o','%i|%j|%T|%E'],text=True)},indent=2))
