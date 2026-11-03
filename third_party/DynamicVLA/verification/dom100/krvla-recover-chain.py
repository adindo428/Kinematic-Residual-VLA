import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

root = Path('/scratch/aparame/Dynamic_VLA/dom100-h20-r2')
project = Path('/scratch/aparame/Dynamic_VLA/third_party/dom100-h20-r2-20261008')
recovery = root/'recovery/20261009'
sys.path.insert(0, str(project))
from core.dom_training import load_family
from utils.dom_experiment import atomic_json, digest, VARIANTS

manifest = load_family(root)
assert not (root/'recipe.json').exists() and not (root/'submissions/production.json').exists()
old = json.loads((root/'cluster-launch.json').read_text())
profiles = []
for job, key in [('16844385','d85389bf17ff282b'), ('16844386','9d30e6dd6cef64ca')]:
    profile = json.loads((root/'profiles'/f'{key}.json').read_text())
    check = dict(profile); fingerprint = check.pop('fingerprint')
    assert digest(check) == fingerprint and profile['manifest'] == manifest['fingerprint']
    assert profile['all_variants_verified']
    rows = json.loads((root/'profiling'/job/'trials.json').read_text())
    for sampling, runtime in profile['runtimes'].items():
        assert runtime['workers'] <= 4
        for variant in VARIANTS:
            assert any(r.get('complete') and r.get('safe') and r.get('mode')=='end_to_end'
                       and r.get('variant')==variant and r['runtime']==runtime for r in rows)
    profiles.append({'job_id':job,'gpu_key':key,'profile_fingerprint':fingerprint,
                     'hardware':profile['hardware'],'runtime':profile['runtime']})
atomic_json(recovery/'completed-profiles-verified.json',{'manifest':manifest['fingerprint'],'profiles':profiles})
record_path = recovery/'submissions.json'
record = json.loads(record_path.read_text()) if record_path.exists() else {}
env = os.environ.copy()
env.update(DVLA_PROJECT_DIR=str(project), DOM100_ROOT=str(root),
           SBATCH_ACCOUNT='cuuser_yue6_vae_verification', HF_HOME='/scratch/aparame/.cache/huggingface',
           HF_LEROBOT_HOME='/scratch/aparame/.cache/huggingface/lerobot', HF_HUB_DISABLE_XET='1',
           HF_HUB_DOWNLOAD_TIMEOUT='60', HF_HUB_ETAG_TIMEOUT='30')


def submit(key, args):
    if key not in record:
        record[key] = subprocess.check_output(['sbatch','--parsable','--account=cuuser_yue6_vae_verification',*args],cwd=project,env=env,text=True).strip().split(';')[0]
        assert record[key].isdigit()
        atomic_json(record_path, record)
    print(key,record[key],flush=True)
    return record[key]


def wrap(body):
    return '--wrap=bash -lc '+shlex.quote('set -e; source scripts/dom100_env.sh; '+body)


for key, gpu, previous, constraint in [('a10040','a100','16844384','gpu_a100_40gb'),('h200','h200','16844387',None)]:
    args=[] if constraint is None else ['--constraint='+constraint]
    submit('profile-'+key,[*args,'--job-name=domprof-'+gpu+'-resume','--gpus='+gpu+':1',
        '--cpus-per-task=8','--mem=64G','--time=04:00:00','--output='+str(recovery/f'profile-{gpu}-%j.out'),
        wrap('python '+str(recovery/'finish-profiles.py')+' --previous '+previous)])
pilot=submit('pilots',['--constraint=gpu_a100_80gb','slurm/pilots_dom100.sbatch'])
cpu=['--cpus-per-task=2','--mem=8G','--time=00:30:00']
recipe=submit('recipe',['--dependency=afterok:'+pilot,'--job-name=dom-recipe',*cpu,
    '--output='+str(recovery/'recipe-%j.out'),wrap('python scripts/dom100.py choose --family-root '+str(root))])
smoke=submit('smoke',['--dependency=afterok:'+recipe,'--constraint=gpu_a100_80gb','slurm/smoke_dom100.sbatch'])
release=submit('release',['--dependency=afterok:'+smoke,'--job-name=dom-launch',*cpu,
    '--output='+str(recovery/'launch-%j.out'),wrap('python scripts/dom100.py preflight --family-root '+str(root)
        +'; python scripts/dom100_submit.py --family-root '+str(root)+' --gpu a100 --constraint gpu_a100_80gb')])
approvals=[]
for key,gpu,constraint in [('a10040','a100','gpu_a100_40gb'),('h100','h100',None),('h200','h200',None)]:
    dependency='afterok:'+recipe
    if key!='h100':dependency+=':'+record['profile-'+key]
    args=[] if constraint is None else ['--constraint='+constraint]
    job=submit('approve-'+key,[*args,'--dependency='+dependency,'--job-name=dom-approve-'+gpu,
        '--gpus='+gpu+':1','--cpus-per-task=8','--mem=64G','--time=24:00:00',
        '--output='+str(recovery/f'approve-{gpu}-%j.out'),wrap('python '+str(root/'approve-gpu.py'))])
    approvals.append({'gpu':gpu,'constraint':constraint,'job_id':job})
archive = recovery/'cluster-launch-before-recovery.json'
if not archive.exists():atomic_json(archive,old)
old.update(profiling_jobs=[{'gpu':'a100','constraint':'gpu_a100_40gb','job_id':record['profile-a10040'],'resumes_trials_from':'16844384'},
    {'gpu':'a100','constraint':'gpu_a100_80gb','job_id':'16844385','profile_saved_and_verified':True,'scheduler_state':'OUT_OF_MEMORY'},
    {'gpu':'h100','constraint':None,'job_id':'16844386','profile_saved_and_verified':True,'scheduler_state':'OUT_OF_MEMORY'},
    {'gpu':'h200','constraint':None,'job_id':record['profile-h200'],'resumes_trials_from':'16844387'}],
    pilots_job=pilot,recipe_job=recipe,smoke_job=smoke,production_release_job=release,optional_gpu_approval_jobs=approvals,
    memory_recovery={'date':'2026-10-09','completed_profiles_retained':profiles,'stalled_jobs_cancelled':['16844384','16844387'],
        'maximum_workers':4,'trial_watchdog_seconds':600,'canonical_source_and_manifest_unchanged':True,'submission_record':str(record_path)})
atomic_json(root/'cluster-launch.json',old)
print('Recovery chain submitted; completed measurements retained.',flush=True)
