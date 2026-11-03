"""Finish requested profile adaptation, then submit a fresh dependency-free run."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
WORK=Path('/work/nvme/bivv/aparameshwaran/dynamicvla')
FAMILY=WORK/'domv-control-ddp-20261011'
PARENT=WORK/'dom100-palmetto-h20-r2'
SOURCE=Path('/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261011-control-gloo/dom100')
sys.path.insert(0,str(SOURCE))
from utils.dom_experiment import digest,atomic_json,source_digest,file_digest


def verified(path):
    result=json.loads(path.read_text());check=dict(result);expected=check.pop('fingerprint')
    if digest(check)!=expected:raise ValueError('Invalid fingerprint: '+str(path))
    return result


def state(stage,**kwargs):
    value=json.loads((HERE/'status.json').read_text()) if (HERE/'status.json').exists() else {}
    value.update(stage=stage,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kwargs)
    atomic_json(HERE/'status.json',value)
    print(json.dumps(value),flush=True)
    return value


def job_state(job):
    raw=subprocess.check_output(['sacct','-X','-j',str(job),'-P','-n','--format=JobIDRaw,State'],universal_newlines=True)
    rows=[line.split('|') for line in raw.splitlines()]
    return next((row[1].split()[0] for row in rows if row[0]==str(job)),None)


def wait_job(job):
    while True:
        status=job_state(job)
        if status in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'):
            return status
        time.sleep(60)


def submit(kind):
    # Preserve an uncertain submission for inspection; never blindly duplicate it.
    intent=HERE/(kind+'-submission.json')
    if intent.exists():
        saved=json.loads(intent.read_text())
        if saved.get('job_id'):return saved['job_id']
        raise RuntimeError('Submission outcome needs inspection: '+str(intent))
    atomic_json(intent,{'stage':'submitting','batch':str(HERE/(kind+'.sbatch'))})
    result=subprocess.run(['sbatch','--parsable',str(HERE/(kind+'.sbatch'))],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    saved={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    if result.returncode==0 and result.stdout.strip().split(';')[0].isdigit():
        saved['job_id']=result.stdout.strip().split(';')[0]
    atomic_json(intent,saved)
    if not saved.get('job_id'):raise RuntimeError('Slurm submission failed: '+result.stderr)
    return saved['job_id']


def candidates():
    original=verified(PARENT/'manifest.json')
    bounded=(HERE/'bounded-single-gpu.json').exists()
    if bounded:
        profile=verified(HERE/'bounded-single-gpu.json')
        if (profile['manifest']!=original['fingerprint'] or 'GH200' not in profile['hardware']['name']
            or not profile['bf16_fp32_agreement']):
            raise ValueError('Preserved single-GPU measurements have invalid provenance')
    else:
        profiles=list((PARENT/'profiles').glob('*.json'))
        if len(profiles)!=1:raise ValueError('Expected one completed GH200 profile')
        profile=verified(profiles[0])
        if (profile['manifest']!=original['fingerprint'] or not profile['all_variants_verified']
            or 'GH200' not in profile['hardware']['name'] or profile['runtimes']['frames']['precision']!='bf16'):
            raise ValueError('Single-GPU profiler provenance is not approved')
    # Revalidate the unchanged data/initialization and frozen training source.
    manifest=verified(FAMILY/'manifest.json')
    if source_digest(SOURCE)!=manifest['source_sha256']:raise ValueError('Training source changed')
    for key in ('split_sha256','weights_sha256','dataset_revision','train_frames'):
        if manifest[key]!=original[key]:raise ValueError('Paired data provenance differs: '+key)
    for name,expected in manifest['artifacts'].items():
        if file_digest(FAMILY/name)!=expected:raise ValueError('Frozen artifact changed: '+name)
    trials=profile['measurements'] if bounded else json.loads((PARENT/'profiling/gh200-3347735/trials.json').read_text())
    allowed=[r for r in trials if r.get('complete') and (bounded or r.get('safe')) and r.get('mode')=='end_to_end'
        and r['runtime']['sampling']=='frames' and r['runtime']['precision']=='bf16'
        and r['runtime']['microbatch']<=88]
    if not allowed:raise ValueError('No measured frame loader runtime')
    best=max((r for r in allowed if r['runtime']['workers']<=4),key=lambda r:r['samples_per_second'])
    light=max((r for r in allowed if r['runtime']['workers']<=2),key=lambda r:r['samples_per_second'])
    if bounded:
        safe_micro=next(r for r in trials if r.get('complete') and r.get('safe') and r.get('mode')=='gpu_only'
            and r['runtime']['precision']=='bf16' and r['runtime']['microbatch']==44)
        runtimes=[dict(light['runtime'],microbatch=44)]
    else:
        runtimes=[best['runtime'],light['runtime'],dict(best['runtime'],microbatch=44)]
        measured_frame=profile['runtimes']['frames']
        if measured_frame['microbatch']<=88:runtimes.insert(0,measured_frame)
    seen=set();folder=FAMILY/'runtime-candidates';folder.mkdir(exist_ok=True)
    paths=[]
    for runtime in runtimes:
        key=digest(runtime)
        if key in seen:continue
        seen.add(key)
        selected={'manifest':manifest['fingerprint'],'runtime':runtime,
            'single_gpu_profile_fingerprint':profile['fingerprint'],
            'reason':'Bounded four-GPU readiness check using measured loaders and a smaller safe microbatch; single-GPU loader combinations are candidate measurements, not production approval'}
        selected['fingerprint']=digest(selected)
        path=folder/(key[:16]+'.json');atomic_json(path,selected);paths.append(str(path))
    atomic_json(FAMILY/'provenance/single-gpu-profile.json',profile)
    atomic_json(HERE/'candidates.json',paths)
    return paths


def select():
    manifest=verified(FAMILY/'manifest.json')
    paths=json.loads((HERE/'candidates.json').read_text())
    allowed={digest(verified(Path(p))['runtime']) for p in paths}
    approved=[]
    for path in (FAMILY/'distributed-profiles').glob('*.json'):
        profile=verified(path)
        if (profile['manifest']==manifest['fingerprint'] and profile['world_size']==4
            and profile['variant']=='domv100' and digest(profile['runtime']) in allowed
            and profile.get('full_data_benchmark') and profile['complete'] and profile['safe']
            and profile['strict_reload'] and profile['rank_rng_reload']):
            approved.append(profile)
    if not approved:raise ValueError('No four-GPU full-data runtime passed; production remains unsubmitted')
    best=max(approved,key=lambda p:p['samples_per_second'])
    runtime=best['runtime']
    recipe=verified(FAMILY/'recipe.json')
    assert recipe['epochs']==100 and recipe['effective_batch']==352 and recipe['precision']=='bf16'
    assert recipe['sampling']=='frames' and recipe['lr']==3e-5 and recipe['early_stopping']['relative_min_delta']==.005
    assert not (FAMILY/'runs/domv100/provenance.json').exists()
    recipe.pop('fingerprint');recipe['runtime_fingerprint']=digest(runtime);recipe['fingerprint']=digest(recipe)
    atomic_json(FAMILY/'recipe.json',recipe)
    single=verified(FAMILY/'provenance/single-gpu-profile.json')
    selected={'manifest':manifest['fingerprint'],'runtime':runtime,
        'distributed_profile_fingerprint':best['fingerprint'],
        'single_gpu_profile_fingerprint':single['fingerprint'],
        'world_size':4,'global_effective_batch':352,
        'samples_per_second':best['samples_per_second'],
        'training_epoch_hours_excluding_validation_and_io':manifest['train_frames']/best['samples_per_second']/3600}
    selected['fingerprint']=digest(selected)
    atomic_json(FAMILY/'selected-runtime.json',selected)
    return selected,recipe


def main():
    lock=(HERE/'coordinator.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (HERE/'production-submission.json').exists() and json.loads((HERE/'production-submission.json').read_text()).get('job_id'):
        return
    state('waiting_for_completed_single_gpu_profile',profiler_job='3347735',old_training_job='3348823',old_training_cancelled=True,production_submitted=False)
    if (HERE/'bounded-single-gpu.json').exists():
        state('using_preserved_single_gpu_measurements',single_gpu_sweep_stopped=True)
    elif wait_job('3347735')!='COMPLETED':
        raise RuntimeError('Single-GPU profiling did not complete successfully')
    paths=candidates()
    profile_job=submit('profile')
    state('four_gpu_runtime_comparison',profile_job=profile_job,candidate_files=paths)
    if wait_job(profile_job)!='COMPLETED':raise RuntimeError('Four-GPU profiling failed; inspect its logs')
    selected,recipe=select()
    state('runtime_selected',selected_runtime=selected,recipe_fingerprint=recipe['fingerprint'])
    training_job=submit('production')
    state('production_submitted',production_submitted=True,training_job=training_job,dependency=None,
          family=str(FAMILY),source=str(SOURCE))
    if job_state('3348823') not in ('CANCELLED','COMPLETED','FAILED','TIMEOUT'):
        subprocess.check_call(['scancel','3348823'])
    state('production_submitted',old_training_cancelled=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        state('failed',error=type(exc).__name__+': '+str(exc))
        raise
