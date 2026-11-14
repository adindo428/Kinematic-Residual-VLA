"""Submit five approved allocation jobs behind two CPU staging gates."""
import argparse
import json
from pathlib import Path
import subprocess
from common import atomic, sha
from subset import require


def submit(root,key,script,dependency=None,hold=False):
    statefile=root/'submissions.json'
    state=json.loads(statefile.read_text()) if statefile.exists() else {}
    if key in state:
        require(state[key].get('job_id'),'Uncertain earlier submission; reconcile before retry')
        return state[key]['job_id']
    record={'script':str(script),'script_sha256':sha(script),'dependency':dependency,'held_for_shared_staging':hold,'state':'intent'}
    state[key]=record;atomic(statefile,state)
    args=['sbatch','--parsable']
    if hold:args.append('--hold')
    if dependency:args.append('--dependency=afterok:'+str(dependency))
    args.append(str(script));p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    record.update(stdout=p.stdout.strip(),stderr=p.stderr.strip(),returncode=p.returncode)
    job=p.stdout.strip().split(';')[0]
    if p.returncode==0 and job.isdigit():record.update(job_id=job,state='submitted')
    else:record['state']='unknown_or_failed_requires_reconciliation'
    atomic(statefile,state);require(record.get('job_id'),'Submission failed/unknown: '+key+' '+p.stderr)
    print(json.dumps({key:record}),flush=True);return job


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--site',choices=['palmetto','delta'],required=True)
    a=p.parse_args();r=a.root.resolve();pal=a.site=='palmetto';scripts=r/'scripts';scripts.mkdir(exist_ok=True)
    from render_jobs import main as render
    import sys
    specs=[('palmetto',name) for name in ['smolvla_baseline','smolvla_kr','domv100']] if pal else [('delta',None),('deltaai',None)]
    for site,name in specs:
        sys.argv=['render_jobs.py','--root',str(r),'--site',site]+(['--recipe',name] if name else [])
        render()
    if not pal:
        script=scripts/'delta-train.sbatch'
        subprocess.check_call(['bash','-n',str(script)])
        trial=subprocess.run(['sbatch','--test-only',str(script)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
        require(trial.returncode==0,'Scheduler rejected Delta request: '+trial.stderr)
        submit(r,'delta',script,hold=True)
        return
    gate=scripts/('staging-'+a.site+'.sbatch')
    account='cuuser_yue6_vae_verification' if pal else 'bivv-delta-gpu'
    partition='work1' if pal else 'cpu'
    python='/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/env/bin/python' if pal else str(r/'env-delta-x86/bin/python')
    activate='export LD_LIBRARY_PATH=/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/lib64:/home/aparame/.conda/envs/dynamicvla/lib:${LD_LIBRARY_PATH:-}' if pal else 'module load pytorch-conda/2.12'
    gate.write_text(f'''#!/usr/bin/env bash
#SBATCH --job-name=main8-stage-{a.site}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output={r}/logs/staging-{a.site}-%j.out
#SBATCH --error={r}/logs/staging-{a.site}-%j.err
set -euo pipefail
cd {r}
{activate}
export PYTHONPATH={r}:{'/scratch/aparame/Dynamic_VLA/cross-vla-krfm-20261019/upstream-vlash' if pal else str(r/'assets/upstream-vlash')}
{python} admit.py --site {a.site} --root {r}
'''+(f'{python} admit.py --site palmetto --root {r} --source-hashes\n' if pal and (r/'parents/vlash_baseline/latest.pt').exists() else ''))
    # Validate every request before submitting its gate or any GPU allocation.
    checks=[gate]+[scripts/((name or site)+'-train.sbatch') for site,name in specs if site==a.site or pal]
    for script in checks:
        subprocess.check_call(['bash','-n',str(script)])
        trial=subprocess.run(['sbatch','--test-only',str(script)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
        require(trial.returncode==0,'Scheduler rejected request '+str(script)+': '+trial.stderr)
    gatejob=submit(r,'staging-'+a.site,gate)
    for site,name in specs:
        if not pal and site=='deltaai':continue
        submit(r,name or site,scripts/((name or site)+'-train.sbatch'),gatejob)
    if not pal:
        atomic(r/'deltaai-dependency.json',{'afterok_delta_cpu_job':gatejob,'script':str(scripts/'deltaai-train.sbatch')})


if __name__=='__main__':main()
