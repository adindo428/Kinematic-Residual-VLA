"""Allocated-node admission for exact full-data resume and protected latest/best."""
import ast,json,math,os
from pathlib import Path
import torch
from core.dom_training import load_family
from utils.dom_experiment import digest,file_digest,atomic_json
root=Path(os.environ['DOM100_ROOT']);launch=Path(os.environ['DOMV_LAUNCH'])
lock=json.loads((launch/'launch-lock.json').read_text())
for name,value in lock['files'].items():assert file_digest(launch/name)==value,name
manifest=load_family(root)
assert manifest['fingerprint']==lock['manifest']
assert manifest['train_frames']==22370201 and manifest['counts']['train']==197198
recipe=json.loads((root/'recipe.json').read_text());selected=json.loads((root/'selected-runtime.json').read_text())
for value in [recipe,selected]:assert digest({k:v for k,v in value.items() if k!='fingerprint'})==value['fingerprint']
assert recipe['fingerprint']==lock['recipe'] and selected['fingerprint']==lock['runtime']
assert selected['world_size']==8 and selected['runtime']['microbatch']==44 and recipe['effective_batch']==352
assert int(os.environ['SLURM_JOB_NUM_NODES'])==2 and 'GH200' in torch.cuda.get_device_name(0)
control=json.loads((launch/'control.json').read_text())
assert control['epochs']==10 and control['scheduler_horizon_epochs']==100
reports={}
for pointer in ['latest.json','best.json']:
    run=root/'runs/domv100';record=json.loads((run/pointer).read_text())
    assert record['fingerprint']=='74e28089ac60a7fd84dae9786445bbcc2204796fdd34d88f60adf117e15b514d'
    generation=run/'generations'/record['generation']
    for name in ['model.safetensors','config.json','training_state.pt','provenance.json']:assert (generation/name).is_file()
    state=torch.load(generation/'training_state.pt',map_location='cpu',weights_only=False,mmap=True)
    assert state['progress']==record['progress'] and state['fingerprint']==record['fingerprint']
    assert state['optimizer']['state'] and state['scheduler']
    assert state['distributed']['world_size']==8 and len(state['distributed']['rng_by_rank'])==8
    for rng in state['distributed']['rng_by_rank']:assert all(k in rng for k in ['python','numpy','torch','cuda'])
    assert state['provenance']['recipe']['fingerprint']==recipe['fingerprint']
    if pointer=='best.json':
        best=state['best'];assert math.isfinite(best['value']) and best['updates']==state['progress']['updates']
    reports[pointer]={'generation':record['generation'],'progress':record['progress'],'best':state['best']}
    del state
assert reports['latest.json']['progress']['updates']>=218468
# The physically staged sources must retain the inspected save/resume implementation.
source=Path(os.environ['DVLA_PROJECT_DIR'])
for name,expected in lock['checkpoint_sources'].items():assert file_digest(source/name)==expected,name
profile=next(json.loads(p.read_text()) for p in (root/'distributed-profiles').glob('*.json') if json.loads(p.read_text()).get('fingerprint')==selected['distributed_profile_fingerprint'])
assert all(profile.get(k) for k in ['safe','strict_reload','rank_rng_reload','sustained_reload_passed','post_training_validation_passed'])
receipt=json.loads((root/'provenance/verification-reuse.json').read_text())
audit=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/setup-scale-domv-20261014/verified.json')
assert file_digest(audit)==receipt['derived_receipt_sha256']
atomic_json(launch/('admission-'+os.environ['SLURM_JOB_ID']+'.json'),{'passed':True,'checkpoints':reports,'manifest':manifest['fingerprint'],'recipe':recipe['fingerprint'],'operational_epochs':10,'scheduler_epochs':100,'world_size':8,'tracking':'offline'})
print('Full-data latest/best state admission passed; native eight-rank resume follows',flush=True)
