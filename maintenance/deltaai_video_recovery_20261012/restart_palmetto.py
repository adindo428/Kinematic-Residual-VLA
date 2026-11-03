import json,subprocess
from pathlib import Path
work=Path('/scratch/aparame/Dynamic_VLA');setup=work/'setup-video-clean-20261012'
report=json.loads((setup/'deployment.json').read_text());source=Path(report['source']);family=Path(report['family'])
assert report['split_sha256']=='63ad508c2be409e7cfcdfd68cca2c6a51fe686045913d1df82f1c0f9d50c5f88'
intent=setup/'replacement-submissions.json'
assert not intent.exists(), 'Inspect existing submissions; do not duplicate them'
old_jobs=['17017544','17091380','17091381']
settings=[]
for job in old_jobs:
 raw=subprocess.check_output(['scontrol','show','job',job,'-o'],universal_newlines=True)
 bits=dict(w.split('=',1) for w in raw.split() if '=' in w)
 assert bits['JobState']=='PENDING', 'Inspect any started job before replacing'
 settings.append({k:bits[k] for k in ('Account','Partition','QOS')})
# Replace only the three authorized allocation alternatives for this paired run.
subprocess.check_call(['scancel']+old_jobs)
specs=[('krv100','slurm/dom100_distributed.sbatch',['--gpus=8']),('krv100-a8','slurm/krv100_a100_8gpu.sbatch',[]),('krv100-h2x8','slurm/krv100_h200_8gpu.sbatch',[])]
result={'old_jobs':old_jobs,'family':str(family),'source':str(source),'submissions':[],'stage':'submitting'}
intent.write_text(json.dumps(result,indent=2)+'\n')
for old,settings,(name,batch,extra) in zip(old_jobs,settings,specs):
 args=['sbatch','--parsable','--account='+settings['Account'],'--partition='+settings['Partition'],'--qos='+settings['QOS'],
 '--chdir='+str(source),'--job-name='+name,'--export=ALL,DVLA_PROJECT_DIR='+str(source)+',DOM100_ROOT='+str(family)+',DOM100_VARIANT=krv100']+extra+[str(source/batch)]
 entry={'name':name,'replaces':old,'args':args,'status':'submitting'};result['submissions'].append(entry);intent.write_text(json.dumps(result,indent=2)+'\n')
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
 entry.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 if r.returncode==0 and r.stdout.strip().split(';')[0].isdigit():entry['job_id']=r.stdout.strip().split(';')[0];entry['status']='submitted'
 else:entry['status']='failed';intent.write_text(json.dumps(result,indent=2)+'\n');raise RuntimeError('Submission failed: '+r.stderr)
 intent.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(entry),flush=True)
result['stage']='submitted';intent.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'stage':result['stage'],'job_ids':[s['job_id'] for s in result['submissions']]}),flush=True)
