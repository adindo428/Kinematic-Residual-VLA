"""Read-only Palmetto run register refresh; no checkpoint/source mutations."""
from pathlib import Path
import json,subprocess


def refresh_audit(run):
    """Read audit evidence; scheduler completion alone never establishes clean data."""
    if not run.get('audit_results'): return
    root=Path(run['audit_results'])
    reports=[json.loads(p.read_text()) for p in sorted(root.glob('status-*.json'))]
    summary={'shards_seen':len(reports),'shards_complete':sum(r['state']=='complete' for r in reports),
             'completed_episodes':sum(r['completed_episodes'] for r in reports),
             'failed_episodes':sum(r['failed_episodes'] for r in reports),
             'frames_decoded':sum(r['frames_decoded'] for r in reports),
             'metadata_errors':sorted({x for r in reports for x in r['metadata_errors']}),
             'warnings':sorted({x for r in reports for x in r['warnings']})}
    summary['all_episodes_checked']=summary['shards_complete']==run['audit_expected_shards'] and summary['completed_episodes']==run['audit_expected_episodes']
    summary['clean_for_checked_criteria']=summary['all_episodes_checked'] and not summary['failed_episodes'] and not summary['metadata_errors']
    run['audit_summary']=summary
    raw=subprocess.check_output(['sacct','-X','-n','-j',str(run['job_id']),'-P','--format=JobID,State'],universal_newlines=True)
    states=[line.split('|')[1] for line in raw.splitlines() if '|' in line]
    run['audit_scheduler_states']=states
    if states:
        if any(s.startswith(('FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')) for s in states):run['scheduler_state']='AUDIT ALLOCATION ERROR'
        elif any(s=='RUNNING' for s in states):run['scheduler_state']='RUNNING'
        elif any(s=='PENDING' for s in states):run['scheduler_state']='PENDING'
        elif any(s.startswith('CANCELLED') for s in states):run['scheduler_state']='CANCELLED'
        elif all(s=='COMPLETED' for s in states):run['scheduler_state']='COMPLETED'
    run.setdefault('audit_note',run.get('note',''))
    run['note']=run['audit_note']+' Checked '+str(summary['completed_episodes'])+'/'+str(run['audit_expected_episodes'])+' episodes; '+str(summary['failed_episodes'])+' failed checks. '+('Full sweep complete.' if summary['all_episodes_checked'] else 'Full sweep not complete.')

def probe(runs):
    discovered=[]
    parent=Path('/scratch/aparame/Dynamic_VLA')
    for root in list(parent.glob('dom100*'))+list(parent.glob('cross-vla-krfm-*')):
        for filename in ('distributed.json','hardware_candidates.json'):
            path=root/'submissions'/filename
            if not path.is_file():continue
            for label,meta in json.loads(path.read_text()).items():
                if any(r.get('cluster')=='Palmetto' and str(r.get('job_id'))==str(meta['job_id']) for r in runs+discovered):continue
                variant=meta.get('variant',label)
                discovered.append({**meta,'cluster':'Palmetto','job_id':str(meta['job_id']),
                    'name':meta.get('name',variant),'variant':variant,
                    'purpose':'DOM manipulation training','group':'production','world_size':meta['world_size'],
                    'family':str(root),'run_directory':str(root/'runs'/variant)})
    runs=runs+discovered
    ids=sorted({str(r['job_id']) for r in runs if r.get('cluster')=='Palmetto' and r.get('job_id')})
    raw=subprocess.check_output(['sacct','-X','-j',','.join(ids),'-P',
        '--format=JobIDRaw,JobName%70,State,Submit,Start,Elapsed,ReqTRES%100,WorkDir%180'],text=True)
    import csv,io
    accounting={r['JobIDRaw']:r for r in csv.DictReader(io.StringIO(raw),delimiter='|')}
    live=subprocess.check_output(['squeue','-u','aparame','-h','-o','%i|%j|%T|%R'],text=True)
    queue={line.split('|')[0]:line.split('|')[1:] for line in live.splitlines()}
    for run in runs:
        if run.get('cluster')!='Palmetto' or not run.get('job_id'):continue
        job=str(run['job_id']);row=accounting.get(job)
        if row:
            run.update(scheduler_state=row['State'],elapsed=row['Elapsed'],submit=row['Submit'],accounting=row)
        if job in queue:
            run.update(scheduler_state=queue[job][1],reason_or_node=queue[job][2])
            detail=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
            run['slurm_details']=detail.strip()
            for key,target in [('StdOut','stdout'),('StdErr','stderr'),('WorkDir','working_directory')]:
                token=next((x.split('=',1)[1] for x in detail.split() if x.startswith(key+'=')),None)
                if token:run[target]=token
        cfgpath=run.get('checkpoint_directory')
        if cfgpath and (Path(cfgpath)/'config.json').is_file():
            config=json.loads((Path(cfgpath)/'config.json').read_text())
            run['checkpoint_epoch']=config.get('completed_epochs',config.get('epoch'))
            run['checkpoint_lr']=config.get('optimizer_lr')
        directory=run.get('run_directory')
        if directory:
            directory=Path(directory)
            allocation=directory/f'allocation-{job}.json'
            if allocation.is_file():run['allocation_outcome']=json.loads(allocation.read_text())
            for name in ('latest.json','best.json','final.json','training_complete.json'):
                path=directory/name
                if path.is_file():run[name]=json.loads(path.read_text())
            provenance=directory/'provenance.json'
            if provenance.is_file():run['wandb_id']=json.loads(provenance.read_text()).get('wandb_id')
        refresh_audit(run)
    return runs
