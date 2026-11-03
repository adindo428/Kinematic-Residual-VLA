"""Persist each scheduler ID immediately; never retry an uncertain submission."""
from pathlib import Path
import fcntl,json,subprocess,os
from handoff import atomic,read,sha,now,require,HERE,NAMES

def main():
    with (HERE/'.submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m=read(HERE/'control-manifest.json')
        for n,h in m['files'].items():require(sha(HERE/n)==h,'Frozen control changed: '+n)
        pf=read(HERE/'remote-preflight.json');require(pf['passed'] and pf['control_manifest_sha256']==sha(HERE/'control-manifest.json'),'No matching remote controller preflight')
        state=read(HERE/'submissions.json');require(not state.get('pending_submission'),'Reconcile uncertain previous scheduler submission')
        def submit(key,script,dependency=None):
            if key=='verification' and state.get(key):return state[key]['job_id']
            if key in NAMES and state['production'].get(key):return state['production'][key]['job_id']
            jobs=subprocess.check_output(['squeue','-u','aparame','-h','-o','%i'],text=True).split()
            for job in jobs:
                details=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
                require('Command='+str(script) not in details,'Unrecorded matching job '+job+'; reconcile')
            args=['sbatch','--parsable']+(['--dependency=afterok:'+dependency] if dependency else [])+[str(script)]
            state['pending_submission']={'key':key,'script':str(script),'dependency':dependency,'attempted_at':now()};atomic(HERE/'submissions.json',state)
            try:r=subprocess.run(args,text=True,capture_output=True,timeout=45)
            except Exception:raise RuntimeError('Unknown submission outcome; retained pending marker, reconcile before retry')
            if r.returncode:
                state.setdefault('definitive_rejections',[]).append({**state.pop('pending_submission'),'stderr':r.stderr});atomic(HERE/'submissions.json',state);raise RuntimeError(r.stderr)
            jid=r.stdout.strip().split(';')[0];require(jid.isdigit(),'Unknown sbatch response; reconcile pending marker')
            row={'job_id':jid,'script':str(script),'dependency':dependency,'submitted_at':now(),'gpu_profile_job_id':'17740320'}
            if key=='verification':state[key]=row
            else:
                row.update(recipe=key,command='train',world_size=2,gpus=2,gpu='H200',cpus=32,memory_gib=256,readiness_release_job_id=dependency,family=read(HERE/'config.json')['root'],source_sha256=read(HERE/'config.json')['source_sha256'],target_updates=30000,lr=2.5e-5,method='pi05 + KRFM' if key.endswith('_kr') else 'pi05 baseline',variant=key,name=key+'-h200-continuation',parent_job_id=read(HERE/'config.json')['runs'][key]['parent_job_id'])
                state['production'][key]=row
            state.pop('pending_submission');atomic(HERE/'submissions.json',state)
            atomic(Path(read(HERE/'config.json')['root'])/'submissions'/'hardware_candidates.json',state['production'])
            print(json.dumps({'key':key,**row}),flush=True);return jid
        dependency=submit('verification',HERE/'verify.sbatch')
        for name in NAMES:submit(name,HERE/f'{name}.sbatch',dependency)
if __name__=='__main__':main()
