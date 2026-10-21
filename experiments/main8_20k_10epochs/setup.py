"""Freeze site-specific paths and checkpoint origins before GPU admission."""
import argparse
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import shutil
from common import atomic, sha, fingerprint
from subset import require

NAMES = ['pi05_baseline','pi05_kr','smolvla_baseline','smolvla_kr','vlash_baseline','vlash_kr','domv100']


def read(p):
    return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--site', choices=['palmetto','delta','deltaai'], required=True)
    p.add_argument('--lock-only', action='store_true')
    a=p.parse_args();r=a.root.resolve()
    if a.lock_only:
        # Runtime package locks are independent on x86 and ARM, with common
        # pinned model source bytes. No package installation occurs in a job.
        modules=['lerobot.policies.pi05.modeling_pi05','lerobot.policies.smolvla.modeling_smolvla',
                 'transformers.models.gemma.modeling_gemma']
        files={importlib.util.find_spec(m).origin:sha(importlib.util.find_spec(m).origin) for m in modules}
        candidate=r/'parent-preparation/implementation-lock.json'
        expected=read(candidate) if candidate.exists() else read(r/'parents/smolvla_baseline/implementation-lock.json')
        for path,value in files.items():
            suffix=path.split('site-packages/')[-1]
            original=[v for k,v in expected['files'].items() if k.endswith(suffix)]
            require(original==[value], 'Pinned policy implementation differs: '+suffix)
        if a.site=='palmetto':
            for file in (r/'delta-domv-f991015a').rglob('*.py'):
                files[str(file)]=sha(file)
        atomic(r/('implementation-lock-'+a.site+'.json'), {'files':files,'versions':{
            k:importlib.metadata.version(k) for k in ['torch','torchcodec','transformers','lerobot','numpy']}})
        return
    original = Path('/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020')
    subset = Path('/scratch/aparame/Dynamic_VLA/cross-vla-krfm-20k-20261020')
    if a.site != 'palmetto':
        original=r/'parent-preparation'
        subset=r
    for name in ['split.json','selected-episodes.json','subset-report.json']:
        if subset/name != r/name:
            shutil.copyfile(subset/name, r/name)
    for name in ['parquet_hashes.json','video_hashes.json','statistics.json']:
        if not (r/name).exists():shutil.copyfile(original/name, r/name)
    provenance=read(original/'data-provenance.json')
    if a.site != 'palmetto':
        data_family=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/domv-scale-ddp-20261014')
        provenance['dataset_root']=read(data_family/'manifest.json')['dataset_root']
    provenance.update(split_sha256=sha(r/'split.json'),
        training_protocol='continued_full_data_checkpoints_on_20k_subset',
        normalization='parent training statistics retained for checkpoint compatibility',
        minimum_additional_subset_epochs=10, prior_full_data_exposure=True)
    atomic(r/'data-provenance.json',provenance)
    refs=read(original/'checkpoints.json')
    if a.site != 'palmetto':
        refs={k:v for k,v in refs.items() if k in ('lerobot/pi05_base','google/paligemma-3b-pt-224')}
        for k,v in refs.items():
            v['path']=str(r/'assets'/('models--'+k.replace('/','--'))/'snapshots'/v['revision'])
    atomic(r/'checkpoints.json',refs)
    (r/'recipes').mkdir(exist_ok=True);(r/'runtimes').mkdir(exist_ok=True)
    names=['smolvla_baseline','smolvla_kr','domv100'] if a.site=='palmetto' else ['pi05_baseline','pi05_kr','vlash_baseline','vlash_kr']
    for name in names:
        parent=r/'parents'/name
        if not parent.exists():continue
        receipt=read(parent/'latest.json');progress=receipt['progress']
        if name == 'domv100':
            recipe={'name':name,'method':'dom','kr':False,'lr':3e-5,'weight_decay':1e-10,
                'clip':None,'global_batch':352,'max_delay':0,'decay_lr':0.,'horizon':20,'fps':25,
                'seed':42,'normalization':'native_checkpoint','velocity_conditioning':True,'phase_gate':False,
                'shared_observation':False,'schedule_steps':100*math.ceil(22370201/352)}
            checkpoint='training_state.pt';identity=receipt['fingerprint']
        else:
            recipe=read(parent/'provenance.json')['recipe']
            recipe['schedule_steps']=20000 if recipe['method']=='smolvla' else 30000
            checkpoint='latest.pt';identity=receipt['identity']
        count=read(r/'subset-report.json')['train_anchors']
        smol=recipe['method']=='smolvla';dom=recipe['method']=='dom'
        gpu='A100' if smol or dom else 'H200' if recipe['method']=='pi05' else 'GH200'
        recipe.update(training_anchors=count,training_episodes=20000,min_epochs=10,
            parent_updates=progress['updates'],parent_samples=progress['samples'],
            steps=progress['updates']+10*(count//recipe['global_batch']),
            initialization='own_verified_checkpoint',schedule_policy='preserve_parent_schedule',
            subset_sha256=sha(r/'split.json'),required_gpu=gpu,world_size=4 if smol or dom else 2,
            minimum_gpu_memory_gib=38 if gpu=='A100' else 90 if gpu=='GH200' else 120,
            budget_authorization='User requested own-checkpoint continuation and at least ten new 20k epochs on 2026-10-21')
        atomic(r/'recipes'/(name+'.json'),recipe)
        runtime={'microbatch':16 if smol else 4 if dom else 16 if gpu=='H200' else 8,
                 'workers':8 if smol else 4,'main_threads':2,'prefetch':2,
                 'autocast_cache_enabled':False,'allocator':'expandable_segments:True'}
        atomic(r/'runtimes'/(name+'.json'),runtime)
        migration={'name':name,'checkpoint_file':checkpoint,'checkpoint_sha256':sha(parent/checkpoint),
            'identity':identity,'progress':progress,'origin':receipt.get('file',str(parent)),
            'prior_full_data_exposure':True,'parent_normalization_preserved':True}
        if dom:migration['weights_sha256']=sha(parent/'model.safetensors')
        atomic(parent/'migration.json',migration)
    for folder in ['runs','profiles','logs']:(r/folder).mkdir(exist_ok=True)
    if a.site=='palmetto':
        site={'dom_family':'/scratch/aparame/Dynamic_VLA/dom100-audited-20261012'}
    else:site={'dom_family':'/work/nvme/bivv/aparameshwaran/dynamicvla/domv-scale-ddp-20261014'}
    atomic(r/'site.json',site)
    print(json.dumps({'configured_site':a.site,'recipes':[f.stem for f in (r/'recipes').glob('*.json')]}),flush=True)


if __name__=='__main__':main()
