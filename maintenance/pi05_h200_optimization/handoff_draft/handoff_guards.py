"""Fail-closed contracts for the exact-run A100-to-H200 handoff.

Pure validation only: this module never signals, cancels, copies weights or submits.
The eventual allocation controller must call these contracts before each mutation.
"""
import hashlib
import json

NAMES = ('pi05_baseline', 'pi05_kr')
PARENTS = {'pi05_baseline': '17719993', 'pi05_kr': '17720948'}
WANDB = {'pi05_baseline': 'ifalqwwf', 'pi05_kr': 'l66vuxv6'}
CORE = ('common.py', 'data.py', 'policy.py', 'instruction_generator.py')
PARENT_SOURCE = '34e920a5f55161ec50c98767a6115cb0aed62382457114cfd00ae33d030b076c'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def validate_pair(approvals, comparisons, accounting, complete, source):
    require(accounting == {'state': 'COMPLETED', 'exit_code': '0:0'}, 'GPU profile accounting incomplete')
    require(complete.get('complete') and str(complete.get('job_id')) == '17740320', 'Wrong pair completion marker')
    require(set(approvals) == set(NAMES) and set(comparisons) == set(NAMES), 'Both variants required')
    for name in NAMES:
        a, c = approvals[name], comparisons[name]
        require(a == complete['profiles'][name], 'Completion marker receipt differs')
        require(a.get('approved') and a['source'] == source, 'Frozen GPU source differs')
        require(a['hardware'] == 'NVIDIA H200' and a['world_size'] == 2, 'Wrong GPU hardware/ranks')
        require(a['decoded_optimizer_updates'] == 200 and a['global_batch'] == 32, 'Wrong readiness budget')
        require(all(a.get(k) for k in ('strict_checkpoint_rng_reload', 'strict_inference_reload', 'coordinated_checkpoint_test', 'includes_intermediate_checkpoint_in_timed_training')), 'GPU readiness incomplete')
        require(a['peak_gpu_reserved_bytes'] <= .85 * a['total_gpu_bytes'], 'GPU headroom insufficient')
        require(a['node_process_peak_rss_bytes'] <= .85 * a['memory_limit_bytes'], 'Node headroom insufficient')
        require(c['identity'] == a['identity'] and c['h200_sustained_samples_per_second'] == a['samples_per_second'], 'Comparison identity/timing differs')
        observations = c['a100_observations']
        require(len(observations) == 20, 'Twenty A100 intervals required')
        require(all(observations[i]['Train/optimizer_updates'] - observations[i-1]['Train/optimizer_updates'] == 10 for i in range(1, 20)), 'A100 intervals are not consecutive ten-update intervals')
        import statistics
        median = statistics.median(v['Train/samples_per_second'] for v in observations)
        require(median > 0 and c['a100_production_median_samples_per_second'] == median, 'A100 reference differs')
        require(c['ratio_to_a100'] == a['samples_per_second'] / median and c['ratio_to_a100'] >= 1.25, 'Sustained benefit insufficient')
        require(len(c['gpu_profile_sha256']) == 64, 'Original GPU receipt SHA missing')
    return True

def validate_allocation(hardware, allocated_job, expected_job, source_verified, release_verified):
    require(str(allocated_job) == str(expected_job), 'Wrong replacement allocation')
    require(source_verified and release_verified, 'Frozen release/source not verified')
    require(len(hardware) == 2, 'Exactly two visible H200s required')
    for h in hardware:
        require(h['name'] == 'NVIDIA H200' and h['total_memory'] >= 120 * 1024**3 and h['cuda_usable'], 'Allocated H200 CUDA check failed')
    return True

def parent_decision(name, scheduler, provenance, latest, completed, allocation_verified):
    require(name in NAMES and allocation_verified, 'Replacement must be allocated before parent action')
    require(str(provenance['job_id']) == PARENTS[name] and provenance['wandb_id'] == WANDB[name], 'Wrong canonical parent')
    require(provenance['hardware'] == 'NVIDIA A100 80GB PCIe' and provenance['world_size'] == 2, 'Wrong parent hardware/ranks')
    require(latest['identity'] == provenance['identity'], 'Parent checkpoint identity differs')
    p = latest['progress']
    require(p['samples'] == p['updates'] * 32 and 0 < p['updates'] <= 30000, 'Invalid parent progress')
    if completed is not None:
        require(completed['stop_reason'] == 'official_step_budget' and completed['progress'] == p and p['updates'] == 30000, 'Unverified completion')
        return 'release_target_without_training'
    require(p['updates'] < 30000, 'Reconcile final checkpoint before migration')
    require(scheduler in ('RUNNING', 'PENDING', 'COMPLETING', 'CANCELLED', 'COMPLETED'), 'Unknown/failed parent scheduler; reconcile first')
    return 'checkpoint_running_parent' if scheduler == 'RUNNING' else 'inspect_saved_stop_before_mutation'

def validate_saved_stop(latest, stopped, checkpoint, signalled_after_ns, latest_mtime_ns, stopped_mtime_ns, original_allocation, current_allocation):
    require(stopped_mtime_ns > signalled_after_ns and latest_mtime_ns > signalled_after_ns, 'No new signal checkpoint/stop receipt')
    require(stopped['reason'] == 'allocation_signal' and stopped['progress'] == latest['progress'] == checkpoint['progress'], 'Saved stop progress differs')
    require(checkpoint['identity'] == latest['identity'], 'Saved checkpoint identity differs')
    require(original_allocation == current_allocation, 'Parent restarted; coordinate another checkpoint before cancel')
    return True

def validate_checkpoint(c, metadata, recipe, world=2):
    require(set(('model','optimizer','scheduler','rng','progress','identity')).issubset(c), 'Production state incomplete')
    require(c['identity'] == metadata['identity'] and c['progress'] == metadata['progress'], 'Checkpoint/sidecar differs')
    p = c['progress']
    require(0 < p['updates'] < recipe['steps'] == 30000 and p['samples'] == p['updates'] * recipe['global_batch'] and recipe['global_batch'] == 32, 'Original total budget/progress differs')
    require(p['epoch'] >= 0 and 0 <= p['cursor'] < 22370201 and p['cursor'] % 32 == 0, 'Saved data cursor invalid')
    require(c['scheduler']['last_epoch'] == p['updates'], 'Scheduler restart/offset detected')
    require(len(c['rng']) == world == 2 and all(set(('python','numpy','torch','cuda')).issubset(v) for v in c['rng']), 'Per-rank RNG state incomplete')
    opt = c['optimizer'];require(opt['state'] and opt['param_groups'], 'Optimizer moments absent')
    for group in opt['param_groups']:
        require(tuple(group['betas']) == (.9,.95) and group['eps'] == 1e-8 and group['weight_decay'] == .01 and group['initial_lr'] == 2.5e-5, 'Official optimizer recipe differs')
    return True

def migration_manifest(name, checkpoint_path, checkpoint_sha, checkpoint_metadata, recipe, preparation, parent_source, target_source, torch_version, actual_hardware, wandb, parent_terminal):
    require(parent_terminal and actual_hardware == 'NVIDIA H200', 'Parent must be terminal before freezing/migration')
    require(wandb['id'] == WANDB[name] and len(checkpoint_sha) == 64, 'Wrong run identity or absent checkpoint SHA')
    require(all(parent_source[n] == target_source[n] for n in CORE), 'Core objective/data implementation differs')
    p = checkpoint_metadata['progress'];require(0 < p['updates'] < recipe['steps'] == 30000, 'Expected latest production progress missing')
    return {'latest_complete_production_checkpoint':True,'user_authorized_cross_hardware':True,'recipe_fingerprint':fingerprint(recipe),'preparation':preparation,'parent_source_sha256':PARENT_SOURCE,'parent_hardware':'NVIDIA A100 80GB PCIe','target_hardware':actual_hardware,'world_size':2,'torch_version':torch_version,'parent_source':parent_source,'progress':p,'checkpoint':checkpoint_path,'checkpoint_sha256':checkpoint_sha,'identity':checkpoint_metadata['identity']}
