"""Submit an L40S validation run, dependent full benchmark, then comparison."""
import hashlib
import json
from pathlib import Path
import subprocess
from prepare import write_json

stage=Path('/scratch/aparame/Dynamic_VLA/runs/official_dom_table1_20261009')
root=Path('/home/aparame/Research/Dynamic_VLA/RESULTS/official_dom_table1/20261009')
assert json.loads((root/'preflight.json').read_text())['validated']
launch_path=root/'launch.json'
launch=json.loads(launch_path.read_text()) if launch_path.exists() else {
    'requested_gpu':'l40s:1','pilot_trials':9,'full_trials':1800,'full_array_dimensions':9,
    'max_concurrent_full_tasks':1,'results':str(root),'stage':str(stage)}
def submit(name,arguments):
    if name in launch:return launch[name]
    value=subprocess.check_output(['sbatch','--parsable',*arguments],text=True).strip().split(';')[0]
    assert value.isdigit(),value
    launch[name]=int(value);write_json(launch_path,launch)
    return int(value)
pilot=submit('pilot_job_id',['--export=ALL,MODE=pilot','--time=02:00:00',str(stage/'slurm/evaluate.sbatch')])
full=submit('full_array_job_id',['--export=ALL,MODE=full','--array=0-8%1',
    f'--dependency=afterok:{pilot}','--kill-on-invalid-dep=yes',str(stage/'slurm/evaluate.sbatch')])
submit('summary_job_id',[f'--dependency=afterok:{full}','--kill-on-invalid-dep=yes',str(stage/'slurm/summary.sbatch')])
write_json(root/'execution_source_sha256.json',{str(p.relative_to(stage)):hashlib.sha256(p.read_bytes()).hexdigest()
    for directory in ('scripts','slurm','tests') for p in (stage/directory).glob('*') if p.is_file()})
print(json.dumps(launch,indent=2))
