"""Run existing bounded checks interactively, without starting or relabeling training."""
import json,os
from pathlib import Path
import prepare_combined as check
from coordinate import state as write_state,atomic_json,HERE

job=os.environ['SLURM_JOB_ID']
training=os.environ['DOM100_QUEUED_TRAINING_JOB']

def state(stage,**kwargs):
    if stage=='training_starting':stage='interactive_readiness_passed'
    kwargs.update(training_job=training,interactive_check_job=job,training_started=False,production_held=True)
    return write_state(stage,**kwargs)

check.state=state
try:
    check.main()
    result={'interactive_job':job,'training_job':training,'passed':True,'training_started':False}
    atomic_json(HERE/('interactive-check-'+job+'.json'),result)
    print(json.dumps(result),flush=True)
except Exception as exc:
    state('interactive_readiness_failed',error=type(exc).__name__+': '+str(exc),readiness_complete=False)
    atomic_json(HERE/('interactive-check-'+job+'.json'),{'interactive_job':job,'training_job':training,
        'passed':False,'error':str(exc),'training_started':False})
    raise
