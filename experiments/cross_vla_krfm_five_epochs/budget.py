"""Explicit five-native-epoch continuation; original LR schedule never resets."""
import math
from common import fingerprint
EXTRA={'min_epochs','training_anchors','schedule_steps','schedule_policy','budget_authorization','required_gpu'}
CORE=('common.py','data.py','policy.py')

def require(ok,message):
    if not ok:raise ValueError(message)

def validate_budget(recipe,count):
    require(recipe['min_epochs']==5 and count==recipe['training_anchors']==22370201,'Wrong audited five-epoch split')
    require(recipe['steps']==5*(count//recipe['global_batch']),'Five complete native epochs required')
    require(recipe['schedule_policy']=='preserve_original_then_floor','Unexpected LR extension policy')
    require(recipe['schedule_steps']==(20000 if recipe['method']=='smolvla' else 30000),'Original schedule changed')

def validate_extension(parent,target):
    validate_budget(target,target['training_anchors'])
    require(set(target)==set(parent)|EXTRA,'Unexpected recipe fields')
    require(all(target[k]==v for k,v in parent.items() if k!='steps'),'Training math/recipe changed')
    require(target['schedule_steps']==parent['steps'] and target['steps']>parent['steps'],'Not a budget-only extension')

def lr_factor(step,recipe):
    if step<1000:return (step+1)/1000
    t=min(1,(step-1000)/max(1,recipe['schedule_steps']-1000))
    floor=recipe['decay_lr']/recipe['lr']
    return floor+(1-floor)*.5*(1+math.cos(math.pi*t))

def validate_progress(p,recipe):
    end=recipe['training_anchors']//recipe['global_batch']*recipe['global_batch']
    require(p['samples']==p['updates']*recipe['global_batch'],'Samples/update counters differ')
    require(p['samples']==p['epoch']*end+p['cursor'],'Epoch/cursor/consumed anchors differ')
    require(0<=p['epoch']<=5 and 0<=p['cursor']<=end and p['cursor']%recipe['global_batch']==0,'Invalid sampler cursor')
    require(0<p['updates']<=recipe['steps'],'Invalid saved updates')

def advance(p,recipe):
    p['updates']+=1;p['samples']+=recipe['global_batch'];p['cursor']+=recipe['global_batch']
    end=recipe['training_anchors']//recipe['global_batch']*recipe['global_batch']
    if p['cursor']==end:p['epoch']+=1;p['cursor']=0
    validate_progress(p,recipe)

def validate_migration(m,recipe,prepared,hardware,world,source,torch_version):
    require(m.get('latest_complete_production_checkpoint') and m.get('user_authorized_budget_extension'),'No exact-checkpoint budget extension approval')
    require(m['recipe_fingerprint']==fingerprint(recipe) and m['preparation']==prepared,'Target recipe/preparation differs')
    validate_extension(m['parent_recipe'],recipe)
    require(m['target_hardware']==hardware and world==m['world_size']==2 and m['torch_version']==torch_version,'Runtime/hardware differs')
    require(all(m['parent_source'][n]==source[n] for n in CORE),'Model/data/objective implementation changed')
    require(m['parent_source_sha256'] in ('34e920a5f55161ec50c98767a6115cb0aed62382457114cfd00ae33d030b076c','86b30f19e4680a2ec47b02a403dad71d62cc75af435849b2053e432bd5a41bec'),'Unexpected canonical parent source')
    require(fingerprint(m['parent_source'])==m['parent_source_sha256'],'Parent source hash differs')
    validate_progress(m['progress'],recipe)

def validate_release(release,approval,approval_sha):
    require(release.get('approved') and release.get('five_epoch_extension_authorized'),'Missing five-epoch production release')
    require(release['gpu_profile_sha256']==approval_sha and release['identity']==approval['identity'],'GPU readiness differs')
    require(approval.get('approved') and approval['decoded_optimizer_updates']==200,'Full decoded readiness incomplete')
    require(all(approval.get(k) for k in ('strict_checkpoint_rng_reload','strict_inference_reload','coordinated_checkpoint_test')),'Strict checkpoint/inference readiness missing')
    require(approval['peak_gpu_reserved_bytes']<=.85*approval['total_gpu_bytes'] and approval['node_process_peak_rss_bytes']<=.85*approval['memory_limit_bytes'],'Insufficient headroom')
