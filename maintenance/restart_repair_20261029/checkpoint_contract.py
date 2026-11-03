"""Admission checks for exact own-checkpoint continuation; no training changes."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def validate(c,m,r,p):
 assert c['identity']==m['identity']==p['identity'],'Identity mismatch'
 assert c['progress']==m['progress'],'Checkpoint/metadata mismatch'
 x=c['progress'];assert 0<x['updates']<=r['steps'],'Invalid progress'
 assert x['samples']==r['parent_samples']+(x['updates']-r['parent_updates'])*r['global_batch'],'Sample counter mismatch'
 assert c['scheduler']['last_epoch']==x['updates'],'Scheduler mismatch'
 assert len(c['rng'])==2 and all(all(k in a for k in ('python','numpy','torch','cuda')) for a in c['rng']),'Missing rank RNG'
 assert c['model'] and c['optimizer']['state'] and c['optimizer']['param_groups'],'Missing model/optimizer'
 usable=r['training_anchors']//r['global_batch']*r['global_batch']
 assert 0<=x['cursor']<usable and x['cursor']%r['global_batch']==0,'Cursor mismatch'
 assert (x['updates']-r['parent_updates'])*r['global_batch']==x['epoch']*usable+x['cursor'],'Epoch mismatch'
 assert (x['updates']==r['steps'])==(x['epoch']>=10),'Incomplete ten-epoch budget'
 return x['epoch']>=10

def main():
 a=argparse.ArgumentParser();a.add_argument('--spec',required=True);args=a.parse_args();s=json.load(open(args.spec));root=Path(s['root']);out=root/'runs'/s['name']
 for path,want in s['hashes'].items():assert sha(path)==want,'Frozen file changed: '+path
 queue=subprocess.check_output(['squeue','-u','aparame','-h','-o','%i|%j|%T'],text=True)
 for line in queue.splitlines():
  j,n,state=line.split('|');assert not (n in (s['parent_job_name'],s['job_name']) and j!=os.environ['SLURM_JOB_ID']),'Another placement remains active'
 account=subprocess.check_output(['sacct','-n','-X','-j',s['parent'],'--format=JobIDRaw,State','-P'],text=True)
 rows=[x.split('|') for x in account.splitlines() if x.startswith(s['parent']+'|')]
 assert rows and rows[-1][1].split()[0].rstrip('+') in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','PREEMPTED','NODE_FAIL','OUT_OF_MEMORY','BOOT_FAIL','DEADLINE'),'Parent accounting not terminal'
 import torch
 m=json.load(open(out/'latest.json'));r=json.load(open(root/'recipes'/(s['name']+'.json')));p=json.load(open(out/'provenance.json'))
 assert Path(m['file']).resolve()==(out/'latest.pt').resolve(),'Not own checkpoint'
 c=torch.load(out/'latest.pt',map_location='cpu',mmap=True,weights_only=False)
 done=validate(c,m,r,p)
 receipt={'job_id':os.environ['SLURM_JOB_ID'],'parent':s['parent'],'progress':c['progress'],'identity':c['identity'],'action':'already_completed' if done else 'resume_own_latest','checkpoint':str(out/'latest.pt')}
 target=Path(s['control'])/'receipts'/(s['name']+'-'+os.environ['SLURM_JOB_ID']+'.json');target.parent.mkdir(exist_ok=True);target.write_text(json.dumps(receipt,indent=2)+'\n')
 return 10 if done else 0
if __name__=='__main__':raise SystemExit(main())
