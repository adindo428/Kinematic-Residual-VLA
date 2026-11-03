"""Read-only evidence from NCSA Delta, separate from DeltaAI."""
import csv,io,json,subprocess
from pathlib import Path
ids=','.join(str(r['job_id']) for r in REQUEST if r.get('job_id'))
raw=subprocess.check_output(['sacct','-X','-j',ids,'-P','--format=JobIDRaw,JobName%50,State,Elapsed,ReqTRES%100'],universal_newlines=True)
rows={r['JobIDRaw']:r for r in csv.DictReader(io.StringIO(raw),delimiter='|')}
for run in REQUEST:
    row=rows.get(str(run.get('job_id')))
    if row:run.update(scheduler_state=row['State'],elapsed=row['Elapsed'],accounting=row)
    setup=Path(run['setup_directory'])
    for name in ['status.json','profile-results-'+str(run['job_id'])+'.json','approval.json']:
        p=setup/name
        if p.is_file():run[name]=json.loads(p.read_text())
    if run.get('run_directory'):
        for name in ['latest.json','best.json','training_complete.json']:
            p=Path(run['run_directory'])/name
            if p.is_file():run[name]=json.loads(p.read_text())
print(json.dumps(REQUEST))
