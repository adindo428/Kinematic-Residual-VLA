"""Immediately persist IDs; uncertain scheduler responses require reconciliation."""
import fcntl,json,subprocess,os
from pathlib import Path
from handoff import read,sha,atomic,require,HERE,now

def main():
 with (HERE/'.submit.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  c=read(HERE/'config.json');root=Path(c['root']);manifest=read(HERE/'control-manifest.json')
  for n,h in manifest['files'].items():require(sha(HERE/n)==h,'Control changed: '+n)
  pf=read(HERE/'preflight.json');require(pf['passed'] and pf['control_manifest_sha256']==sha(HERE/'control-manifest.json'),'Missing matching preflight')
  require(pf['scripts_manifest_sha256']==sha(HERE/'scripts-manifest.json'),'Scheduler scripts inventory changed')
  for n,h in read(HERE/'scripts-manifest.json').items():require(sha(HERE/n)==h,'Scheduler script changed: '+n)
  source=read(root/'source-manifest.json')
  for n,h in source['files'].items():require(sha(root/n)==h,'Training source changed')
  state=read(HERE/'submissions.json');require(not state.get('pending_submission'),'Reconcile uncertain submission')
  # Old frozen 30k H200 controllers cannot race with this budget extension.
  for jid in c['supersedes_pending_h200']:
   q=subprocess.run(['squeue','-h','-j',jid,'-o','%i'],text=True,capture_output=True)
   require(not q.stdout.strip(),'Obsolete old-budget H200 controller still queued/active')
   a=subprocess.check_output(['sacct','-X','-j',jid,'-n','-P','--format=JobIDRaw,State,ExitCode'],text=True)
   rows=[r.split('|') for r in a.splitlines() if r.startswith(jid+'|')]
   require(len(rows)==1 and rows[0][1].startswith('CANCELLED'),'Old-budget controller retirement not verified')
  def submit(kind,name,dependency=None):
   if name in state[kind]:return state[kind][name]['job_id']
   script=HERE/(name+('-profile.sbatch' if kind=='profiles' else '-train.sbatch'))
   # Compare exact script identity across the current user's live submissions.
   jobs=subprocess.check_output(['squeue','-u','aparame','-h','-o','%i'],text=True).split()
   for jid in jobs:
    d=subprocess.run(['scontrol','show','job',jid,'-o'],text=True,capture_output=True)
    require('Command='+str(script) not in d.stdout,'Unrecorded identical script active: '+jid)
   marker={'kind':kind,'recipe':name,'script':str(script),'script_sha256':sha(script),'dependency':dependency,'attempted_at':now()}
   state['pending_submission']=marker;atomic(HERE/'submissions.json',state)
   try:r=subprocess.run(['sbatch','--parsable']+(['--dependency=afterok:'+dependency] if dependency else [])+[str(script)],text=True,capture_output=True,timeout=45)
   except Exception:raise RuntimeError('Uncertain sbatch result; pending marker retained; reconcile before retry')
   if r.returncode:
    state.setdefault('definitive_rejections',[]).append({**state.pop('pending_submission'),'stderr':r.stderr});atomic(HERE/'submissions.json',state);raise RuntimeError(r.stderr)
   jid=r.stdout.strip().split(';')[0];require(jid.isdigit(),'Unknown response; reconcile pending marker')
   row={**marker,'job_id':jid,'submitted_at':now(),'source_sha256':source['source_sha256'],'family':str(root),'gpus':2,'gpu':c['runs'][name]['hardware'],'cpus':32,'memory_gib':c['runs'][name]['memory_gib'],'target_epochs':5,'target_updates':read(root/'recipes'/(name+'.json'))['steps'],'parent_job_id':c['runs'][name]['parent_job_id'],'original_wandb_id':c['runs'][name]['parent_wandb_id'],'command':'profile' if kind=='profiles' else 'train','variant':name}
   state[kind][name]=row;state.pop('pending_submission');atomic(HERE/'submissions.json',state)
   atomic(root/'submissions'/'hardware_candidates.json',state['production'])
   print(json.dumps({'kind':kind,**row}),flush=True);return jid
  for name in c['runs']:
   profile=submit('profiles',name)
   submit('production',name,profile)
if __name__=='__main__':main()
