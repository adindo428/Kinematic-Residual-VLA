"""Recover a closed tar after the final checkpoint symlink was missing."""
import hashlib,json,sys,tarfile,time
from pathlib import Path
root=Path(__file__).resolve().parent
family=Path('/scratch/aparame/Dynamic_VLA/dom100-audited-20261012')
out=root/'dom-clean.tar'
weights=Path('/scratch/aparame/Dynamic_VLA/dom100/initialization/model.safetensors')
m=json.loads((family/'manifest.json').read_text())
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert digest(weights)==m['weights_sha256'],'Pinned checkpoint hash mismatch'
link=family/'initialization/model.safetensors'
assert link.is_symlink()
if not link.exists():
 link.unlink();link.symlink_to(weights)
assert digest(link)==m['weights_sha256']
sys.argv=['pack.py',str(family),str(out)]
exec((root/'pack.py').read_text().split("progress=out.with_suffix")[0])
partial=out.with_suffix('.partial')
assert not out.exists()
expected=list(entries)
with tarfile.open(partial,'r:') as archive:
 names=[];end=0
 for member in archive:
  names.append(member.name);end=member.offset_data+((member.size+511)//512)*512
assert names==expected[:-1],(len(names),len(expected))
assert expected[-1]=='family/initialization/model.safetensors'
import shutil
with partial.open('r+b') as f:
 f.seek(end);tail=f.read();assert not tail.strip(b'\0'),'Unexpected data after last member'
 f.seek(end)
 item=tarfile.TarInfo(expected[-1]);item.size=weights.stat().st_size;item.mode=0o644
 f.write(item.tobuf(format=tarfile.USTAR_FORMAT))
 with weights.open('rb') as src:shutil.copyfileobj(src,f,8*1024*1024)
 f.write(b'\0'*((-item.size)%512+1024))
 f.write(b'\0'*((-f.tell())%10240));f.truncate()

partial.rename(out)
out.with_suffix('.sha256').write_text(digest(out)+'\n')
p=out.with_suffix('.status.tmp');p.write_text(json.dumps({'state':'complete','files':len(entries),'total':len(entries),'archive':str(out),'updated_unix':time.time(),'recovered_from_job':'17123683'}));p.replace(out.with_suffix('.status.json'))
