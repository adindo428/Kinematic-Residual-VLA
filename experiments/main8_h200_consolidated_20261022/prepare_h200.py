"""Preserve the verified latest stage origins before A100 checkpoint removal."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import torch
from common import atomic,sha
from subset import require
from placements import NAMES

ORIGIN=Path('/scratch/aparame/Dynamic_VLA/main8-20k-10epochs-20261021-r2')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    require(not (r/'prepared-palmetto.json').exists(),'Already frozen')
    for name in NAMES:
        require(not (ORIGIN/'runs'/name/'latest.pt').exists(),'New production checkpoint appeared; reconcile newest state before staging')
    for filename in ['statistics.json','data-provenance.json','checkpoints.json','parquet_hashes.json','video_hashes.json','site.json','implementation-lock-palmetto.json']:
        shutil.copyfile(ORIGIN/filename,r/filename)
    require(sha(ORIGIN/'split.json')==sha(r/'split.json'),'Subset differs')
    for folder in ['parents','recipes','runtimes','trackers','runs','profiles','logs']:(r/folder).mkdir(exist_ok=True)
    summary={}
    for name in NAMES:
        source=ORIGIN/'parents'/name;parent=r/'parents'/name
        subprocess.check_call(['cp','-a','--reflink=auto',str(source),str(parent)])
        migration=json.loads((parent/'migration.json').read_text());statefile=parent/migration['checkpoint_file']
        require(sha(statefile)==migration['checkpoint_sha256'],'Copied parent digest differs')
        if name=='domv100':require(sha(parent/'model.safetensors')==migration['weights_sha256'],'DOM weights differ')
        state=torch.load(statefile,map_location='cpu',mmap=True,weights_only=False)
        require(state['progress']==migration['progress'],'Parent progress differs')
        require(state['scheduler']['last_epoch']==state['progress']['updates'],'Scheduler progress differs')
        require(len(state['optimizer']['state'])>0,'Missing saved optimizer')
        recipe=json.loads((ORIGIN/'recipes'/(name+'.json')).read_text())
        recipe.update(required_gpu='H200',world_size=2,minimum_gpu_memory_gib=120,replica=False,replica_placement='h200',
            budget_authorization='User requested H200-only continuation from latest own checkpoint on 2026-10-22; existing ten additional subset epochs and schedule retained')
        atomic(r/'recipes'/(name+'.json'),recipe)
        atomic(r/'runtimes'/(name+'.json'),{'microbatch':22 if name=='domv100' else 32,'workers':4 if name=='domv100' else 8,
            'main_threads':2,'prefetch':2,'autocast_cache_enabled':False,'allocator':'expandable_segments:True'})
        tracker=json.loads((parent/'wandb.json').read_text())
        atomic(r/'trackers'/(name+'.json'),{'replica_wandb_id':tracker['id'],'parent_wandb_id':tracker['id'],
            'parent_wandb_url':tracker.get('url'),'independent_replica':False,'canonical_continuation':True})
        summary[name]={'progress':migration['progress'],'checkpoint_sha256':migration['checkpoint_sha256'],
            'model_weights_sha256':migration.get('weights_sha256'),'source_parent':str(source),'retained_tracker':tracker,
            'steps':recipe['steps'],'global_batch':recipe['global_batch']}
        del state
    atomic(r/'parent-validation.json',{'passed':True,'models':summary,'latest_production_state_selected':True,'unstarted_20k_stage':True})
    from freeze import main as freeze
    import sys
    sys.argv=['freeze.py','--root',str(r),'--site','palmetto'];freeze()

if __name__=='__main__':main()
