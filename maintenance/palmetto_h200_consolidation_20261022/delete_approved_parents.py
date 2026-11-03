"""Delete exactly eight user-approved redundant parents after full SHA checks."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,json,os,subprocess

ROOT=Path(__file__).resolve().parent
APPROVED_SOURCES={
 '/scratch/aparame/Dynamic_VLA/main8-replicas-a100-20261021/parents/pi05_baseline/latest.pt',
 '/scratch/aparame/Dynamic_VLA/main8-replicas-a100-20261021/parents/pi05_kr/latest.pt',
 '/scratch/aparame/Dynamic_VLA/main8-replicas-a100-20261021/parents/vlash_baseline/latest.pt',
 '/scratch/aparame/Dynamic_VLA/main8-replicas-a100-20261021/parents/vlash_kr/latest.pt',
 '/scratch/aparame/Dynamic_VLA/main8-20k-10epochs-20261021-r2/parents/smolvla_baseline/latest.pt',
 '/scratch/aparame/Dynamic_VLA/main8-20k-10epochs-20261021-r2/parents/smolvla_kr/latest.pt',
 '/scratch/aparame/Dynamic_VLA/main8-20k-10epochs-20261021-r2/parents/domv100/training_state.pt',
 '/scratch/aparame/Dynamic_VLA/main8-20k-10epochs-20261021-r2/parents/domv100/model.safetensors',
}

def atomic(name,value):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()

def signature(p):
 s=p.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)

def verify(row):
 src=Path(row['delete_candidate']);dst=Path(row['preserved_h200_copy']);expected=row['verified_migration_sha256']
 assert str(src) in APPROVED_SOURCES and not src.is_symlink() and not dst.is_symlink()
 assert src.resolve(strict=True)==src and dst.resolve(strict=True)==dst
 assert '/main8-replicas-h200-20261021/parents/' in str(dst) or '/main8-h200-only-20261022/parents/' in str(dst)
 assert src.stat().st_uid==os.getuid() and src.stat().st_size==dst.stat().st_size==row['bytes']
 a=signature(src);b=signature(dst);assert a[:2]!=b[:2],'H200 copy must be independent'
 key='weights_sha256' if src.name=='model.safetensors' else 'checkpoint_sha256'
 assert json.loads((src.parent/'migration.json').read_text())[key]==expected
 assert json.loads((dst.parent/'migration.json').read_text())[key]==expected
 assert digest(src)==expected and digest(dst)==expected,'Actual checkpoint SHA differs'
 assert signature(src)==a and signature(dst)==b,'Checkpoint changed during verification'
 return {**row,'source_signature':a,'h200_signature':b,'both_actual_sha256_verified':True}

def main():
 authorization=json.loads((ROOT/'authorization.json').read_text())
 assert authorization['user_answer']=='Approve those eight redundant parent files'
 manifest=json.loads((ROOT/'approved-files.json').read_text());rows=manifest['parent_files']
 assert {r['delete_candidate'] for r in rows}==APPROVED_SOURCES and len(rows)==8
 assert sum(r['bytes'] for r in rows)==129348096607
 live=subprocess.check_output(['squeue','-u','aparame','-h','-o','%i|%j'],text=True)
 assert not any('main8-rep-a100-' in line or 'main8-20k-smolvla_' in line or 'main8-20k-domv100' in line for line in live.splitlines()),'A100 source-family job is active again'
 retired=['17859069','17859070','17859071','17859072','17839146','17839147','17839148']
 raw=subprocess.check_output(['sacct','-X','-n','-j',','.join(retired),'-P','--format=JobIDRaw,State'],text=True)
 state={x.split('|')[0]:x.split('|')[1] for x in raw.splitlines() if '|' in x}
 assert all(state.get(job,'').startswith('CANCELLED') for job in retired),'Missing terminal cancellation proof'
 atomic('progress.json',{'state':'verifying_exact_eight_pairs','file_count':8,'at':datetime.now(timezone.utc).isoformat()})
 with ThreadPoolExecutor(max_workers=4) as pool:verified=list(pool.map(verify,rows))
 atomic('verified-pairs.json',{'files':verified,'all_actual_hashes_match':True,'checked_at':datetime.now(timezone.utc).isoformat()})
 removed=[]
 for row in verified:
  src=Path(row['delete_candidate']);dst=Path(row['preserved_h200_copy'])
  assert signature(src)==tuple(row['source_signature']) and signature(dst)==tuple(row['h200_signature'])
  src.unlink();assert not src.exists() and dst.is_file()
  removed.append(row)
  atomic('deletion-receipt.json',{'complete':False,'removed_files':removed,'removed_bytes':sum(r['bytes'] for r in removed)})
 assert all(not Path(r['delete_candidate']).exists() and Path(r['preserved_h200_copy']).is_file() for r in rows)
 atomic('deletion-receipt.json',{'complete':True,'removed_files':removed,'removed_bytes':sum(r['bytes'] for r in removed),
  'source_absence_verified':True,'all_h200_copies_retained':True,'completed_at':datetime.now(timezone.utc).isoformat(),
  'scope':'Exactly eight explicit user-approved parent files; no other checkpoint, historical family, source, dataset or metadata deleted'})
 print(json.dumps({'complete':True,'removed_files':len(removed),'removed_bytes':sum(r['bytes'] for r in removed),'all_h200_copies_retained':True}),flush=True)

if __name__=='__main__':main()
