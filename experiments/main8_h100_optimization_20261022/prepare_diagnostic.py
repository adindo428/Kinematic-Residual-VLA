import argparse,json,shutil
from pathlib import Path
from common import atomic,sha
from placements import NAMES
from subset import require

BASE=Path('/scratch/aparame/Dynamic_VLA')
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
 common=BASE/'main8-h200-only-20261022'
 require(json.loads((common/'prepared-palmetto.json').read_text())['complete'],'H200 parent preservation must finish first')
 for filename in ['statistics.json','data-provenance.json','checkpoints.json','parquet_hashes.json','video_hashes.json','site.json','implementation-lock-palmetto.json']:
  shutil.copyfile(common/filename,r/filename)
 for folder in ['parents','recipes','runtimes','profiles','runs','trackers','logs','tuning']:(r/folder).mkdir(exist_ok=True)
 lock=json.loads((r/'implementation-lock-palmetto.json').read_text())
 vlash=BASE/'cross-vla-krfm-20261019/upstream-vlash'
 lock['files'].update({str(p):sha(p) for p in vlash.rglob('*.py') if '/.git/' not in str(p)})
 atomic(r/'implementation-lock-palmetto.json',lock)
 for name in NAMES:
  origin=common if name in ('smolvla_baseline','smolvla_kr','domv100') else BASE/'main8-replicas-h200-20261021'
  parent=r/'parents'/name;require(not parent.exists(),'Diagnostic already prepared; reconcile first');parent.symlink_to(origin/'parents'/name,target_is_directory=True)
  recipe=json.loads((origin/'recipes'/(name+'.json')).read_text());recipe.update(required_gpu='H100',minimum_gpu_memory_gib=75,world_size=2,diagnostic_only=True,replica_placement='h100_diagnostic')
  atomic(r/'recipes'/(name+'.json'),recipe)
  atomic(r/'trackers'/(name+'.json'),{'diagnostic_only':True,'wandb_disabled':True})
 atomic(r/'replica-config.json',{'mode':'bounded_h100_optimization_only','production_forbidden':True,'parent_references':'Read-only links to preserved H200 parents; no checkpoint duplication'})
 from freeze import main as freeze
 import sys
 sys.argv=['freeze.py','--root',str(r),'--site','palmetto'];freeze()
if __name__=='__main__':main()
