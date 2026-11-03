"""Remove only user-authorized obsolete result groups; preserve research checkpoints."""
import datetime
import json
import os
from pathlib import Path
import shutil

ROOT=Path('/home/aparame/Research/Dynamic_VLA/RESULTS')
AUDIT=Path('/home/aparame/Research/Dynamic_VLA/maintenance/results_cleanup_20261009/audit.json')
KEEP=['tracking_comparison_four_75epoch','tracking_table1_75epoch','official_dom_table1','official_dom_diagnostic','official_dom_friction']
REMOVE=['paper_backups','tracking_comparison_latest','tracking_coordinate_velocity_verification','tracking_current_eval','tracking_evaluator_verification','tracking_velocity_verification']
comparison=ROOT/'tracking_comparison_four_75epoch/Coordinate_Conditioned_Velocity/16787490/comparison_with_existing'
summary=json.loads((comparison/'summary.json').read_text())
assert summary['complete'] and summary['recorded_rollouts']==150 and len(summary['models'])==5
manifest=json.loads((comparison/'manifest.json').read_text())
checkpoints={name:Path(path) for name,path in manifest['models'].items()}
checkpoint_inventory={name:sorted(p.name for p in path.iterdir() if p.is_file()) for name,path in checkpoints.items()}
for path in checkpoints.values():
 assert not path.resolve().is_relative_to(ROOT)
for name in KEEP:assert (ROOT/name).is_dir(),name
for name in REMOVE:
 path=ROOT/name
 assert path.parent==ROOT and path.is_dir() and not path.is_symlink(),path
# Follow directory links only; do not resolve every regular telemetry file.
visited=set();bad=[]
def scan(path):
 if path.is_symlink():
  path=path.resolve()
  if any(path.is_relative_to(ROOT/name) for name in REMOVE):bad.append(str(path));return
 key=str(path)
 if key in visited:return
 visited.add(key)
 with os.scandir(path) as entries:
  for entry in entries:
   if entry.is_symlink():
    target=Path(entry.path).resolve()
    if any(target.is_relative_to(ROOT/name) for name in REMOVE):bad.append(entry.path)
    elif target.is_dir():scan(target)
   elif entry.is_dir(follow_symlinks=False):scan(Path(entry.path))
for name in KEEP:scan(ROOT/name)
assert not bad,bad
record={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'authorization':'User requested retaining five-VLA tracking, ongoing tracking Table 1, and DOM manipulation; other RESULTS can be removed.',
        'retained':KEEP,'preserved_models':summary['models'],'preserved_tracking_trials':150,
        'checkpoint_paths':{k:str(v) for k,v in checkpoints.items()},'retained_links_into_deleted_roots':bad,
        'removed':[],'logical_bytes_removed':0}
AUDIT.parent.mkdir(parents=True,exist_ok=True)
def save():
 tmp=AUDIT.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(AUDIT)
save()
for name in REMOVE:
 path=ROOT/name;count=size=0
 for base,dirs,files in os.walk(path,followlinks=False):
  for f in files:
   p=Path(base)/f
   if not p.is_symlink():count+=1;size+=p.stat().st_size
 shutil.rmtree(path)
 assert not path.exists()
 record['removed'].append({'name':name,'regular_files':count,'logical_bytes':size})
 record['logical_bytes_removed']+=size
 save()
 print('REMOVED',name,size,flush=True)
for name in KEEP:assert (ROOT/name).is_dir(),name
for name,path in checkpoints.items():
 assert sorted(p.name for p in path.iterdir() if p.is_file())==checkpoint_inventory[name]
assert json.loads((comparison/'summary.json').read_text())==summary
record['remaining_roots']=sorted(p.name for p in ROOT.iterdir())
record['completed_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
record['retained_checkpoint_inventory_unchanged']=True
record['completed']=True
save();print(json.dumps(record,indent=2),flush=True)
