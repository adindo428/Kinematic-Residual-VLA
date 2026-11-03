"""CPU-only immutable replica staging from the same verified study origins."""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import uuid
import torch
from common import atomic, sha
from subset import require
from placements import NAMES, layout

ORIGIN=Path('/scratch/aparame/Dynamic_VLA/main8-20k-10epochs-20261021')
PREP=Path('/scratch/aparame/Dynamic_VLA/main8-20k-10epochs-20261021-r2')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--placement',required=True)
    a=p.parse_args();r=a.root;s=layout(a.placement)
    require(not (r/'prepared-palmetto.json').exists(),'Already frozen; do not restage submitted replicas')
    config=json.loads((r/'replica-config.json').read_text());require(config['mode']=='separate_replicas','Explicit parallel-replica authorization required')
    expected=json.loads((r/'source-checkpoint-hashes.json').read_text())
    for name in ['statistics.json','data-provenance.json','checkpoints.json','parquet_hashes.json','video_hashes.json','site.json']:
        shutil.copyfile(PREP/name,r/name)
    require(sha(PREP/'split.json')==sha(r/'split.json'),'Different subset')
    files={}
    original=json.loads((PREP/'implementation-lock-palmetto.json').read_text())
    for module in ['lerobot.policies.pi05.modeling_pi05','lerobot.policies.smolvla.modeling_smolvla','transformers.models.gemma.modeling_gemma']:
        path=importlib.util.find_spec(module).origin;value=sha(path)
        suffix=path.split('site-packages/')[-1]
        require([v for k,v in original['files'].items() if k.endswith(suffix)]==[value],'Pinned implementation differs: '+module)
        files[path]=value
    vlash=Path('/scratch/aparame/Dynamic_VLA/cross-vla-krfm-20261019/upstream-vlash')
    require((vlash/'vlash/policies/pi05/modeling_pi05.py').exists(),'Missing pinned VLASH source')
    files.update({str(p):sha(p) for p in vlash.rglob('*.py') if '/.git/' not in str(p)})
    atomic(r/'implementation-lock-palmetto.json',{'files':files,'versions':{n:importlib.metadata.version(n) for n in ['torch','torchcodec','transformers','lerobot','numpy']}})
    for d in ['parents','recipes','runtimes','profiles','runs','logs','trackers']:(r/d).mkdir(exist_ok=True)
    summary={};tracker_ids=set()
    for name in NAMES:
        origin=ORIGIN/'parents'/name;parent=r/'parents'/name;parent.mkdir(exist_ok=True)
        require(not any((r/'runs'/name).glob('*')),'Replica already has output')
        source=origin/'latest.pt';target=parent/'latest.pt'
        require(source.stat().st_size==expected[name]['bytes'],'Parent size changed')
        if not target.exists():subprocess.check_call(['cp','--reflink=auto',str(source),str(target)])
        require(sha(target)==expected[name]['sha256'],'Parent digest changed: '+name)
        for filename in ['latest.json','provenance.json','wandb.json']:
            shutil.copyfile(origin/filename,parent/filename)
        receipt=json.loads((parent/'latest.json').read_text());provenance=json.loads((parent/'provenance.json').read_text())
        state=torch.load(target,map_location='cpu',mmap=True,weights_only=False)
        require(state['identity']==receipt['identity'] and state['progress']==receipt['progress'],'Parent identity/progress differs')
        require(state['scheduler']['last_epoch']==state['progress']['updates'],'Parent schedule counter differs')
        require(len(state['rng'])==2 and len(state['optimizer']['state'])>0,'Parent optimizer or rank RNG missing')
        require(all(k in v for v in state['optimizer']['state'].values() for k in ('step','exp_avg','exp_avg_sq')),'Incomplete Adam moments')
        progress=receipt['progress'];recipe=dict(provenance['recipe']);count=json.loads((r/'subset-report.json').read_text())['train_anchors']
        require(count==2406773,'Wrong train anchor count')
        recipe.update(training_anchors=count,training_episodes=20000,min_epochs=10,parent_updates=progress['updates'],parent_samples=progress['samples'],
            steps=progress['updates']+10*(count//recipe['global_batch']),initialization='own_verified_checkpoint',schedule_policy='preserve_parent_schedule',
            schedule_steps=30000,subset_sha256=sha(r/'split.json'),required_gpu='A100' if a.placement=='a100' else 'H200',
            world_size=s['world'],minimum_gpu_memory_gib=s['minimum_gib'],replica_placement=a.placement,
            replica=True,budget_authorization='User authorized separate Palmetto replicas alongside Delta/DeltaAI on 2026-10-21; ten additional subset epochs unchanged')
        atomic(r/'recipes'/(name+'.json'),recipe)
        runtime={'microbatch':4 if a.placement=='a100' else 16 if recipe['method']=='pi05' else 8,
            'workers':8 if a.placement=='a100' else 4,'main_threads':2,'prefetch':2,'autocast_cache_enabled':False,'allocator':'expandable_segments:True'}
        atomic(r/'runtimes'/(name+'.json'),runtime)
        atomic(parent/'migration.json',{'name':name,'checkpoint_file':'latest.pt','checkpoint_sha256':expected[name]['sha256'],
            'identity':receipt['identity'],'progress':progress,'origin':receipt.get('file',str(source)),
            'prior_full_data_exposure':True,'parent_normalization_preserved':True,'replica_branch_origin':str(source)})
        parent_tracker=json.loads((parent/'wandb.json').read_text())
        tracker_file=r/'trackers'/(name+'.json')
        tracker=json.loads(tracker_file.read_text()) if tracker_file.exists() else {'replica_wandb_id':uuid.uuid4().hex[:8],
            'parent_wandb_id':parent_tracker['id'],'parent_wandb_url':parent_tracker.get('url'),'placement':a.placement,'independent_replica':True}
        require(tracker['replica_wandb_id']!=parent_tracker['id'] and tracker['replica_wandb_id'] not in tracker_ids,'Tracker collision')
        tracker_ids.add(tracker['replica_wandb_id']);atomic(tracker_file,tracker)
        summary[name]={'parent_progress':progress,'checkpoint_sha256':expected[name]['sha256'],'tracker':tracker,'total_update_cap':recipe['steps']}
        del state
    atomic(r/'parent-validation.json',{'passed':True,'models':summary,'optimizer_scheduler_rng_preserved':True,'same_stage_branch_origin':'same frozen checkpoint as canonical main8 stage; separate independently tracked replica'})
    from freeze import main as freeze
    import sys
    sys.argv=['freeze.py','--root',str(r),'--site','palmetto'];freeze()

if __name__=='__main__':main()
