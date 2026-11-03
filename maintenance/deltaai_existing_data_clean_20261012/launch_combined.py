"""Replace the waiting three-allocation coordinator with one authorized allocation."""
import fcntl,json,os,signal,subprocess,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE=Path('/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261012-video-clean/dom100')
import sys
sys.path.insert(0,str(SOURCE))
from utils.dom_experiment import atomic_json,file_digest


def main():
    lock=(HERE/'combined-launch.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    intent=HERE/'combined-submission.json'
    if intent.exists():
        saved=json.loads(intent.read_text())
        if saved.get('job_id'):print(json.dumps(saved),flush=True);return
        raise RuntimeError('Inspect uncertain combined submission before any retry')
    status=json.loads((HERE/'status.json').read_text())
    assert not status.get('training_job') and not status.get('production_submitted')
    assert status['verification_job']=='3370057' and not status.get('profile_job')
    previous=HERE/'superseded-separate-workflow.json'
    atomic_json(previous,{'status':status,'verification_submission':json.loads((HERE/'verify-submission.json').read_text()),
        'reason':'User requested verification, readiness and training in one allocation on 2026-10-13'})
    process=json.loads((HERE/'coordinator-process.json').read_text())
    pid=process['pid']
    result=subprocess.run(['ps','-p',str(pid),'-o','args='],stdout=subprocess.PIPE,universal_newlines=True)
    if result.returncode==0:
        assert str(HERE/'coordinate.py') in result.stdout,'Unexpected coordinator process'
        os.kill(pid,signal.SIGTERM)
        for _ in range(25):
            probe=subprocess.run(['ps','-p',str(pid),'-o','stat='],stdout=subprocess.PIPE,universal_newlines=True)
            if probe.returncode or probe.stdout.strip().startswith('Z'):break
            time.sleep(.2)
        else:raise RuntimeError('Old coordinator did not stop; no new submission attempted')
    subprocess.run(['scancel','3370057'],check=True)
    atomic_json(HERE/'coordinator-process.json',dict(process,terminated=True,superseded_by='combined.sbatch'))
    batch=HERE/'combined.sbatch'
    saved={'stage':'submitting','batch':str(batch),'batch_sha256':file_digest(batch),
        'old_verification_job':'3370057','combined_allocation':True,'dependency':None}
    atomic_json(intent,saved)
    result=subprocess.run(['sbatch','--parsable',str(batch)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    saved.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    if result.returncode==0 and result.stdout.strip().split(';')[0].isdigit():
        saved['job_id']=result.stdout.strip().split(';')[0]
        saved['stage']='submitted'
    atomic_json(intent,saved)
    if not saved.get('job_id'):raise RuntimeError('Combined submission failed: '+result.stderr)
    # Alias for readers of the old coordinator; it now exits without new jobs.
    atomic_json(HERE/'production-submission.json',saved)
    status.update(stage='combined_allocation_queued',training_job=saved['job_id'],
        production_submitted=True,combined_allocation=True,dependency=None,verification_job=saved['job_id'],
        old_verification_job='3370057',old_verification_cancelled=True,
        old_training_job='3367506',readiness_total_limit_minutes=15,
        updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    atomic_json(HERE/'status.json',status)
    old=subprocess.run(['scancel','3367506'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    status.update(old_training_cancelled=old.returncode==0)
    if old.returncode:status['old_training_cancellation_error']=old.stderr
    atomic_json(HERE/'status.json',status)
    print(json.dumps(saved),flush=True)


if __name__=='__main__':main()
