"""Copy an explicit manifest between allocated CPU nodes, with no shared ACL.

One-use bearer authorization is passed only through private SSH pipes. The
source serves only the frozen manifest and terminates after the transfer.
Credentials and private environments must never be included in the manifest.
"""
import argparse
import json
from pathlib import Path
import secrets
import subprocess
import sys

SENDER = r'''
import hashlib,hmac,json,os,pathlib,sys,tarfile
from http.server import HTTPServer,BaseHTTPRequestHandler
assert os.environ['SLURM_JOB_ID']=='20066627' and 'login' not in os.environ['SLURMD_NODENAME']
config=json.load(sys.stdin)
entries=json.loads(pathlib.Path(config['manifest']).read_text())
if config.get('dataset_archive'):
 entries=[(src,name) for src,name in entries if name.startswith(('dataset/data/','dataset/videos/'))]
for src,name in entries:
 assert src.startswith(('/scratch/aparame/Dynamic_VLA/','/scratch/aparame/.cache/huggingface/'))
 assert not pathlib.PurePosixPath(name).is_absolute() and '..' not in pathlib.PurePosixPath(name).parts
 assert not any(part in {'token','.netrc','secrets.env','conda-meta','.git'} for part in pathlib.Path(src).parts)
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  if self.path!='/artifacts' or not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+config['token']):
   self.send_error(403);return
  self.send_response(200);self.send_header('Content-Type','application/x-tar');self.end_headers()
  index={}
  options={'format':tarfile.USTAR_FORMAT} if config.get('dataset_archive') else {}
  with tarfile.open(fileobj=self.wfile,mode='w|',dereference=True,bufsize=4*1024*1024,copybufsize=4*1024*1024,**options) as archive:
   if config.get('dataset_archive'):
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque
    from io import BytesIO
    remaining=iter(entries);pending=deque()
    with ThreadPoolExecutor(max_workers=16) as pool:
     def enqueue():
      try:src,name=next(remaining)
      except StopIteration:return
      pending.append((name,pool.submit(pathlib.Path(src).read_bytes)))
     for _ in range(16):enqueue()
     while pending:
      name,future=pending.popleft();raw=future.result()
      info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o600;info.mtime=0
      index[name[len('dataset/'):]]={'offset':archive.offset+512,'size':len(raw)}
      archive.addfile(info,BytesIO(raw));enqueue()
   else:
    for src,name in entries:archive.add(src,arcname=name,recursive=False)
  if index:(pathlib.Path(config['manifest']).parent/'dataset-tar-index.json').write_text(json.dumps(index))
  self.server.complete=True
server=HTTPServer(('0.0.0.0',0),Handler);server.complete=False;server.timeout=120
print(json.dumps({'node':os.environ['SLURMD_NODENAME'],'port':server.server_port,'files':len(entries)}),flush=True)
while not server.complete:server.handle_request()
server.server_close()
print(json.dumps({'complete':True,'files':len(entries)}),flush=True)
'''

RECEIVER = r'''
import hashlib,json,os,pathlib,sys,tarfile,urllib.request,time
assert os.environ['SLURM_JOB_ID']=='20264351' and 'login' not in os.environ['SLURMD_NODENAME']
config=json.load(sys.stdin);root=pathlib.Path(config['destination']);root.mkdir(parents=True,exist_ok=True)
assert str(root)=='/scratch/arai/Dynamic_VLA/low-clutter-velocity-gated-20261112'
request=urllib.request.Request(config['url'],headers={'Authorization':'Bearer '+config['token']})
files=0;size=0;started=time.time()
with urllib.request.urlopen(request,timeout=3600) as response:
 if config.get('dataset_archive'):
  temporary=root/'dataset.tar.transfer-tmp';checksum=hashlib.sha256()
  with temporary.open('wb') as output:
   chunks=0;tail=b''
   while True:
    block=response.read(4*1024*1024)
    if not block:break
    output.write(block);checksum.update(block);size+=len(block);chunks+=1;tail=(tail+block)[-10240:]
    if chunks%64==0:(root/'transfer-progress.json').write_text(json.dumps({'bytes':size,'elapsed_seconds':time.time()-started,'storage':'indexed_tar'}))
  assert size%512==0 and tail[-1024:]==b'\0'*1024, 'Archive stream ended before final tar markers'
  os.chmod(str(temporary),0o600);os.replace(str(temporary),str(root/'dataset.tar'));files=config['files']
 else:
  with tarfile.open(fileobj=response,mode='r|',bufsize=4*1024*1024) as archive:
   for member in archive:
    path=pathlib.PurePosixPath(member.name)
    assert member.isfile() and not path.is_absolute() and '..' not in path.parts
    target=root/str(path);target.parent.mkdir(parents=True,exist_ok=True)
    stream=archive.extractfile(member);temp=target.with_name(target.name+'.transfer-tmp')
    with temp.open('wb') as output:
     while True:
      block=stream.read(4*1024*1024)
      if not block:break
      output.write(block);size+=len(block)
    os.chmod(str(temp),0o600);os.replace(str(temp),str(target));files+=1
    if files%1000==0:(root/'transfer-progress.json').write_text(json.dumps({'files':files,'bytes':size,'elapsed_seconds':time.time()-started}))
receipt={'complete':True,'files':files,'bytes':size,'seconds':time.time()-started,'credentials_transferred':False,'private_environments_copied':False}
if config.get('dataset_archive'):receipt.update(storage='indexed_tar',archive_sha256=checksum.hexdigest())
(root/'transfer-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--dataset-archive',action='store_true')
    args = parser.parse_args()
    import palmetto_allocated_transport as source
    source_command = source.allocated_command(['python3', '-c', SENDER])
    index_command = source.allocated_command(['cat',str(Path(args.manifest).parent/'dataset-tar-index.json')])
    import palmetto_arai_allocated_transport as destination
    receiver_command = destination.allocated_command(['python3', '-c', RECEIVER])
    token = secrets.token_hex(32)
    sender = subprocess.Popen(source_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
    sender.stdin.write(json.dumps({'token': token, 'manifest': args.manifest,'dataset_archive':args.dataset_archive}))
    sender.stdin.close()
    try:
        line = sender.stdout.readline()
        if not line:
            raise RuntimeError('Sender startup failed: ' + sender.stderr.read()[-1500:])
        ready = json.loads(line)
        config = {'token': token, 'url': 'http://{}:{}/artifacts'.format(ready['node'], ready['port']),
                  'destination': '/scratch/arai/Dynamic_VLA/low-clutter-velocity-gated-20261112',
                  'dataset_archive':args.dataset_archive,'files':ready['files']}
        print(json.dumps({'state': 'copying_scoped_artifacts', 'files': ready['files']}), flush=True)
        receiver = subprocess.run(receiver_command, input=json.dumps(config), stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, timeout=10800)
        if receiver.returncode:
            raise RuntimeError('Receiver failed: ' + receiver.stderr[-2000:])
        receipt = json.loads(receiver.stdout)
        assert receipt['files'] == ready['files']
        sender.wait(timeout=60)
        assert sender.returncode == 0, sender.stderr.read()[-1500:]
        if args.dataset_archive:
            process=subprocess.Popen(index_command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            code="import pathlib,json,sys; p=json.load(sys.stdin); assert len(p)==67920; pathlib.Path('/scratch/arai/Dynamic_VLA/low-clutter-velocity-gated-20261112/dataset-tar-index.json').write_text(json.dumps(p)); print('archive_index_saved')"
            indexed=subprocess.run(destination.allocated_command(['python3','-c',code]),stdin=process.stdout,
                                   capture_output=True,text=True,timeout=120)
            process.stdout.close();process.wait(timeout=30)
            assert indexed.returncode==process.returncode==0, 'Archive index transfer failed'
        args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt), flush=True)
    finally:
        if sender.poll() is None:
            sender.terminate()


if __name__ == '__main__':
    main()
