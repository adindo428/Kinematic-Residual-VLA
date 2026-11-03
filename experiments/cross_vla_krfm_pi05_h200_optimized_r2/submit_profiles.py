"""One bounded H200 profiling allocation for the authorized pi05 pair; no production submission."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,json,subprocess,os
from common import atomic,sha
ROOT=Path(__file__).resolve().parent

def main():
    with (ROOT/'.submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((ROOT/'submissions.json').read_text())
        if state.get('pending_submission'):raise RuntimeError('Reconcile uncertain submission before retry')
        if state.get('paired_profile'):
            print(json.dumps(state['paired_profile']));return
        if not json.loads((ROOT/'preflight.json').read_text())['passed']:raise RuntimeError('CPU/preflight checks incomplete')
        manifest=json.loads((ROOT/'source-manifest.json').read_text())
        for name,expected in manifest['files'].items():
            if sha(ROOT/name)!=expected:raise RuntimeError('Source changed')
        script=ROOT/'profile_pair.sbatch';jobname='pi05_h200_opt_r2_pair'
        for job in subprocess.check_output(['squeue','-u','aparame','-h','-n',jobname,'-o','%i'],text=True).split():
            details=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
            if 'Command='+str(script) in details:raise RuntimeError('Unrecorded matching job exists: '+job)
        state['pending_submission']={'script':str(script),'attempted_at':datetime.now(timezone.utc).isoformat()};atomic(ROOT/'submissions.json',state)
        result=subprocess.run(['sbatch','--parsable',str(script)],text=True,capture_output=True,timeout=45)
        if result.returncode:
            state.setdefault('submission_errors',[]).append({**state.pop('pending_submission'),'stderr':result.stderr});atomic(ROOT/'submissions.json',state);raise RuntimeError(result.stderr)
        jid=result.stdout.strip().split(';')[0]
        if not jid.isdigit():raise RuntimeError('Uncertain sbatch outcome; reconcile pending receipt')
        state['paired_profile']={'job_id':jid,'script':str(script),'command':'bounded_pair_profile','recipes':['pi05_baseline','pi05_kr'],'gpus':2,'gpu':'H200','cpus':32,'memory_gib':256,'time_limit':'02:00:00','submitted_at':datetime.now(timezone.utc).isoformat()};state.pop('pending_submission');atomic(ROOT/'submissions.json',state);print(json.dumps(state['paired_profile']),flush=True)
if __name__=='__main__':main()
