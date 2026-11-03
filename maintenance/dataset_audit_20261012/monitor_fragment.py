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
