"""Stream only approved staging files to the existing arai CPU allocation."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time
import palmetto_arai_allocated_transport as transport

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'scratch/dom45/training/original20k-distance-arai-20261113'
REMOTE='/scratch/arai/Dynamic_VLA/original20k-distance-20261113'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def main(files=None):
    files=sorted(p for p in STAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts) if files is None else sorted(files)
    expected={str(p.relative_to(STAGE)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files}
    receiver='''import hashlib,json,os,sys,tarfile
from pathlib import Path
root=Path(REMOTE);expected=EXPECTED
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|',bufsize=1024*1024) as archive:
 for member in archive:
  rel=Path(member.name)
  assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
  assert member.name in expected and member.size==expected[member.name]['bytes']
  dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  assert not dest.is_symlink()
  tmp=dest.with_name(dest.name+'.transfer-part');h=hashlib.sha256()
  with archive.extractfile(member) as source,tmp.open('wb') as target:
   while True:
    data=source.read(8<<20)
    if not data:break
    target.write(data);h.update(data)
   target.flush();os.fsync(target.fileno())
  assert h.hexdigest()==expected[member.name]['sha256'],member.name
  tmp.replace(dest)
  if member.size>1024*1024*1024:print('Verified archive transfer',member.size,flush=True)
(root/'upload-receipt.json').write_text(json.dumps({'complete':True,'files':expected},indent=2)+'\\n')
print('Verified staged files',len(expected),flush=True)
'''.replace('REMOTE',repr(REMOTE)).replace('EXPECTED',repr(expected))
    start=time.monotonic()
    with (STAGE/'upload.log').open('a') as log:
        child=subprocess.Popen(transport.allocated_command(['/usr/bin/python3','-u','-c',receiver]),stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
        try:
            with tarfile.open(fileobj=child.stdin,mode='w|',bufsize=1024*1024) as archive:
                for p in files:archive.add(p,arcname=str(p.relative_to(STAGE)),recursive=False)
            child.stdin.close();code=child.wait(timeout=120)
            assert code==0,code
        finally:
            if child.poll() is None:child.terminate()
    print(json.dumps(dict(complete=True,files=len(files),seconds=time.monotonic()-start)))

if __name__=='__main__':main()
