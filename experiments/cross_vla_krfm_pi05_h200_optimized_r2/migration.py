"""Fail-closed identity checks for an explicitly authorized A100-to-H200 continuation."""
from common import fingerprint

PARENT_SOURCE = '34e920a5f55161ec50c98767a6115cb0aed62382457114cfd00ae33d030b076c'

def validate_migration(m, recipe, prepared, target_hardware, world, source, torch_version):
    if not m.get('latest_complete_production_checkpoint') or not m.get('user_authorized_cross_hardware'):
        raise ValueError('Production checkpoint migration not approved')
    if m['recipe_fingerprint']!=fingerprint(recipe) or m['preparation']!=prepared:
        raise ValueError('Recipe or data identity differs')
    if m['parent_source_sha256']!=PARENT_SOURCE or m['parent_hardware']!='NVIDIA A100 80GB PCIe':
        raise ValueError('Unexpected parent run implementation or hardware')
    if m['target_hardware']!=target_hardware or 'H200' not in target_hardware or m['world_size']!=world or world!=2:
        raise ValueError('Target hardware or rank count differs')
    if m['torch_version']!=torch_version:
        raise ValueError('Torch version differs')
    for name in ['common.py','data.py','policy.py','instruction_generator.py']:
        if m['parent_source'][name]!=source[name]:raise ValueError('Model, data or objective implementation differs')
    p=m['progress']
    if not 0<p['updates']<recipe['steps'] or p['samples']!=p['updates']*recipe['global_batch']:
        raise ValueError('Invalid saved total-budget progress')

def validate_release(release, approval, approval_sha):
    if not release.get('approved') or not release.get('paired_migration_authorized'):
        raise ValueError('Migration release absent')
    if release['gpu_profile_sha256']!=approval_sha or release['identity']!=approval['identity']:
        raise ValueError('Readiness receipt differs')
    if approval['decoded_optimizer_updates']<200 or not approval.get('strict_checkpoint_rng_reload') or not approval.get('strict_inference_reload') or not approval.get('coordinated_checkpoint_test'):
        raise ValueError('Sustained reload readiness incomplete')
    if approval['peak_gpu_reserved_bytes']>.85*approval['total_gpu_bytes'] or approval['node_process_peak_rss_bytes']>.85*approval['memory_limit_bytes']:
        raise ValueError('Readiness headroom insufficient')
    if release['throughput_ratio_to_a100']<1.25:
        raise ValueError('Less than25% sustained throughput advantage')
