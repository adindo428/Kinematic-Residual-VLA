import json,subprocess
from pathlib import Path
setup=Path('/scratch/aparame/Dynamic_VLA/setup-video-clean-20261012');intent=setup/'replacement-submissions.json'
r=json.loads(intent.read_text());assert not any(s.get('job_id') for s in r['submissions'])
assert len(r['submissions'])==1 and r['submissions'][0]['returncode']!=0
source=Path(r['source']);root=r['family'];failed=r['submissions'][0];r['failed_submission_attempts']=[failed];r['submissions']=[]
base=[a for a in failed['args'] if a.startswith(('--account=','--partition='))]
specs=[('krv100','17017544','slurm/dom100_distributed.sbatch',['--gpus=8']),('krv100-a8','17091380','slurm/krv100_a100_8gpu.sbatch',[]),('krv100-h2x8','17091381','slurm/krv100_h200_8gpu.sbatch',[])]
for name,old,batch,extra in specs:
 args=['sbatch','--parsable']+base+['--chdir='+str(source),'--job-name='+name,'--export=ALL,DVLA_PROJECT_DIR='+str(source)+',DOM100_ROOT='+root+',DOM100_VARIANT=krv100']+extra+[str(source/batch)]
 entry={'name':name,'replaces':old,'args':args,'status':'submitting'};r['submissions'].append(entry);intent.write_text(json.dumps(r,indent=2)+'\n')
 result=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
 entry.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
 if result.returncode==0 and result.stdout.strip().split(';')[0].isdigit():entry['job_id']=result.stdout.strip().split(';')[0];entry['status']='submitted'
 else:entry['status']='failed';intent.write_text(json.dumps(r,indent=2)+'\n');raise RuntimeError(result.stderr)
 intent.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(entry),flush=True)
r['stage']='submitted';intent.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'stage':'submitted','job_ids':[s['job_id'] for s in r['submissions']]}),flush=True)
