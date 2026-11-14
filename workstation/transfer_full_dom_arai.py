"""Copy only the audited public DOM files to arai through allocated CPU steps."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import subprocess
import shlex
import tarfile
import time
import palmetto_arai_allocated_transport as transport

ROOT=Path(__file__).resolve().parents[1]
FAMILY=ROOT/'scratch/dom45/training/krv100-full-20261010/family'
STAGE=ROOT/'scratch/dom45/training/full-distance-arai-h2004-20261113'
PREVIOUS=ROOT/'scratch/dom45/training/original20k-distance-arai-20261113'
REMOTE='/scratch/arai/Dynamic_VLA/full-distance-h2004-20261113'

def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def prepare():
    STAGE.mkdir(exist_ok=True);(STAGE/'data-transfer').mkdir(exist_ok=True)
    manifest=json.loads((FAMILY/'manifest.json').read_text())
    split=json.loads((FAMILY/'split.json').read_text())['splits']
    existing=json.loads((PREVIOUS/'base-dataset-index.json').read_text())
    for name,entry in json.loads((PREVIOUS/'dataset-tar-index.json').read_text()).items():
        if entry['archive']=='overlay':existing[name]=entry
    pq=json.loads((FAMILY/'parquet_hashes.json').read_text());video=json.loads((FAMILY/'video_hashes.json').read_text())
    parts=[{} for _ in range(4)];total=0;missing=[];data=Path(manifest['dataset_root'])
    for ep in sorted(sum(split.values(),[])):
        names=[f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet']
        names += [f'videos/chunk-{ep//1000:03d}/{cam}/episode_{ep:06d}.mp4' for cam in ['observation.images.opst_cam','observation.images.wrist_cam']]
        for name in names:
            if name in existing:continue
            path=data/name
            if not path.is_file():missing.append(name);continue
            size=path.stat().st_size
            expected=pq[str(ep)] if name.endswith('.parquet') else video[name]['sha256']
            parts[ep%4][name]={'size':size,'sha256':expected};total+=size
    assert not missing, f'{len(missing)} required local files missing; first={missing[:3]}'
    for i,part in enumerate(parts):write(STAGE/'data-transfer'/f'part-{i}.json',part)
    plan=dict(dataset_root=str(data),remote_root=REMOTE,split_sha256=manifest['split_sha256'],
              missing_files=sum(map(len,parts)),missing_bytes=total,reused_files=len(existing),
              training_episodes=len(split['train']),training_frames=manifest['train_frames'],
              method='Dataset-only copy; no ACL modifications; four CPU-routed streams')
    write(STAGE/'data-transfer/plan.json',plan);print(json.dumps(plan),flush=True)

RECEIVER='''import hashlib,json,os,sys,tarfile
from pathlib import Path
root=Path(DEST);part=PART
assert os.environ['SLURM_JOB_ID']=='20264351' and 'login' not in os.environ['SLURMD_NODENAME']
count=total=0
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|',bufsize=1024*1024) as archive:
 iterator=iter(archive);first=next(iterator);assert first.name=='transfer-plan.json'
 expected=json.load(archive.extractfile(first));seen=set()
 for member in iterator:
  name=member.name;rel=Path(name)
  assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts and name in expected and name not in seen
  assert member.size==expected[name]['size']
  dest=root/'dataset'/rel;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.is_symlink()
  temp=dest.with_name(dest.name+'.copy-part');h=hashlib.sha256()
  with archive.extractfile(member) as source,temp.open('wb') as target:
   while True:
    data=source.read(1024*1024)
    if not data:break
    h.update(data);target.write(data)
  assert h.hexdigest()==expected[name]['sha256'],name
  temp.replace(dest);seen.add(name);count+=1;total+=member.size
  if count%10000==0:print('part',part,'verified_files',count,'bytes',total,flush=True)
 assert len(seen)==len(expected)
(root/'data-transfer'/('part-'+str(part)+'-complete.json')).write_text(json.dumps({'complete':True,'files':count,'bytes':total})+'\\n')
print('COMPLETE',part,count,total,flush=True)
'''

def transfer(part):
    plan=json.loads((STAGE/'data-transfer/plan.json').read_text());data=Path(plan['dataset_root'])
    path=STAGE/'data-transfer'/f'part-{part}.json';raw=path.read_bytes();expected=json.loads(raw)
    code=RECEIVER.replace('DEST',repr(REMOTE)).replace('PART',str(part))
    command=transport.allocated_command(['/usr/bin/python3','-u','-c',code])
    allocation=json.loads((ROOT/'scratch/dom45/training/palmetto-arai-control-20261112/cpu_allocation.json').read_text())
    remote=shlex.split(command[-1]);remote.insert(1,'--nodelist='+allocation['allocated_nodes'][part])
    command[-1]=shlex.join(remote)
    with (STAGE/'data-transfer'/f'part-{part}.log').open('a') as log:
        child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
        try:
            with tarfile.open(fileobj=child.stdin,mode='w|',format=tarfile.USTAR_FORMAT,bufsize=1024*1024) as archive:
                info=tarfile.TarInfo('transfer-plan.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
                for name,entry in expected.items():
                    info=tarfile.TarInfo(name);info.size=entry['size'];info.mode=0o644
                    with (data/name).open('rb') as f:archive.addfile(info,f)
            child.stdin.close();result=child.wait(timeout=120);assert result==0,(part,result)
        finally:
            if child.poll() is None:child.terminate()
    return dict(part=part,files=len(expected),complete=True)

def copy_existing():
    code='''import json,hashlib,os,tarfile
from pathlib import Path
root=Path(DEST)
expected=json.loads((root/'data-transfer/reused-file-hashes.json').read_text());seen=set()
archives=['/scratch/arai/Dynamic_VLA/low-clutter-velocity-gated-20261112/dataset.tar','/scratch/arai/Dynamic_VLA/original20k-distance-20261113/dataset-overlay.tar']
for archive in archives:
 with tarfile.open(archive,mode='r|') as stream:
  for member in stream:
   rel=Path(member.name)
   if rel.parts and rel.parts[0]=='dataset':rel=Path(*rel.parts[1:])
   name=str(rel)
   if rel.parts and rel.parts[0]=='meta':continue
   assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts and name in expected and name not in seen,name
   dest=root/'dataset'/rel;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.is_symlink()
   temp=dest.with_name(dest.name+'.reuse-part');h=hashlib.sha256()
   with stream.extractfile(member) as source,temp.open('wb') as target:
    while True:
     data=source.read(1024*1024)
     if not data:break
     h.update(data);target.write(data)
   assert h.hexdigest()==expected[name],name
   temp.replace(dest);seen.add(name)
   if len(seen)%10000==0:print('Verified reused files',len(seen),flush=True)
assert len(seen)==len(expected)
(root/'data-transfer/reused-complete.json').write_text(json.dumps({'complete':True,'files':len(seen)})+'\\n')
'''.replace('DEST',repr(REMOTE))
    with (STAGE/'data-transfer/reused.log').open('a') as log:
        subprocess.run(transport.allocated_command(['/usr/bin/python3','-u','-c',code]),stdout=log,stderr=subprocess.STDOUT,check=True)
    return dict(part='reused',complete=True)

def run():
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures=[pool.submit(transfer,i) for i in range(4)]+[pool.submit(copy_existing)]
        results=[f.result() for f in futures]
    write(STAGE/'data-transfer/complete.json',dict(complete=True,parts=results,seconds=time.monotonic()-started))

if __name__=='__main__':
    import sys
    if sys.argv[1:] == ['prepare']:prepare()
    else:run()
