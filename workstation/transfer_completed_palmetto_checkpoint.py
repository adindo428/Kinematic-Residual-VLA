"""Stream one verified checkpoint from a Slurm CPU node through an SSH tunnel."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import select
import socket
import subprocess
import sys
import time

import palmetto_allocated_transport as transport

ROOT=transport.REPO/'scratch/dom45/training/palmetto-completed-models-20261110'


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    source=json.loads((ROOT/'remote_checkpoint_hash_and_training.private.json').read_text())
    metadata=json.loads((ROOT/'remote_metadata_audit.private.json').read_text())
    dest=ROOT/'vlash_baseline';dest.mkdir(exist_ok=True)
    for name,value in metadata['vlash'].items():save(dest/name,value)
    token=secrets.token_hex(32)
    server="""
import hmac,json,os,socket,sys
from pathlib import Path
path=Path(SOURCE)
assert os.environ['SLURM_JOB_ID']=='20066627'
listener=socket.socket();listener.bind(('0.0.0.0',0));listener.listen(4);listener.settimeout(120)
print(json.dumps({'host':socket.gethostname(),'port':listener.getsockname()[1],'slurm_job_id':os.environ['SLURM_JOB_ID']}),flush=True)
try:
 for attempt in range(4):
  connection,address=listener.accept();connection.settimeout(120)
  with connection:
   request=json.loads(connection.makefile('rb').readline(4096))
   if not hmac.compare_digest(request.get('token',''),TOKEN):continue
   offset=request['offset'];stat=path.stat();assert 0<=offset<=stat.st_size
   connection.sendall((json.dumps({'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns})+'\\n').encode())
   with path.open('rb') as f:connection.sendfile(f,offset=offset)
   after=path.stat();assert (after.st_size,after.st_mtime_ns)==(stat.st_size,stat.st_mtime_ns)
   break
finally:listener.close()
""".replace('SOURCE',repr(source['checkpoint'])).replace('TOKEN',repr(token))
    process=None;forward=None;ssh=transport.ssh_base();errorlog=(dest/'cpu-file-server.stderr.log').open('w')
    try:
        process=subprocess.Popen(transport.allocated_command(['python3','-u','-c',server]),stdout=subprocess.PIPE,stderr=errorlog,text=True)
        assert select.select([process.stdout],[],[],90)[0],'Allocated CPU server did not become ready'
        ready=json.loads(process.stdout.readline());assert ready['slurm_job_id']=='20066627'
        assert ready['host'].startswith('node') and 'login' not in ready['host']
        with socket.socket() as reserve:reserve.bind(('127.0.0.1',0));port=reserve.getsockname()[1]
        forward=f'127.0.0.1:{port}:{ready["host"]}:{ready["port"]}'
        control=subprocess.run([*ssh[:-1],'-O','forward','-L',forward,ssh[-1]],capture_output=True,text=True,timeout=30)
        assert control.returncode==0,'SSH tunnel request failed: '+control.stderr[:300]
        partial=dest/'latest.pt.part';offset=partial.stat().st_size if partial.exists() else 0
        total=source['bytes'];assert offset<=total
        with socket.create_connection(('127.0.0.1',port),timeout=30) as client:
            client.settimeout(120);client.sendall((json.dumps({'token':token,'offset':offset})+'\n').encode())
            reader=client.makefile('rb');header=json.loads(reader.readline(4096))
            assert header['bytes']==total and header['mtime_ns']==source['mtime_ns']
            written=offset;tick=time.monotonic();last=tick
            with partial.open('ab') as target:
                while block:=reader.read(4*1024*1024):
                    target.write(block);written+=len(block)
                    if time.monotonic()-last>5:
                        target.flush();last=time.monotonic()
                        save(dest/'transfer_progress.json',{'stage':'streaming','bytes':written,'total':total,'host':ready['host'],
                            'slurm_job_id':ready['slurm_job_id'],'rate_MB_s':(written-offset)/(last-tick)/1e6,'updated_utc':datetime.now(timezone.utc).isoformat()})
            assert written==total and partial.stat().st_size==total,'Incomplete checkpoint stream'
        assert process.wait(timeout=90)==0,'CPU file server failed'
        save(dest/'transfer_progress.json',{'stage':'verifying_local_hash','bytes':total,'total':total})
        h=hashlib.sha256()
        with partial.open('rb') as f:
            while block:=f.read(8*1024*1024):h.update(block)
        assert h.hexdigest()==source['sha256'],'Transferred checkpoint SHA-256 differs'
        final=dest/'latest.pt';partial.replace(final);os.utime(final,ns=(source['mtime_ns'],source['mtime_ns']))
        import torch
        state=torch.load(final,map_location='cpu',mmap=True,weights_only=False)
        assert state['identity']=='e02ca1ddf295d045cab81269cbc5c69420d2ccce2df0151c7650262bb29b9874'
        assert state['progress']==metadata['vlash']['latest.json']['progress']
        assert state['progress']['epoch']==10 and state['progress']['cursor']==0
        receipt={'verified':True,'checkpoint':str(final),'bytes':total,'mtime_ns':final.stat().st_mtime_ns,'sha256':h.hexdigest(),
            'identity':state['identity'],'progress':state['progress'],'training_complete':True,'evaluation_eligible':True,
            'source':source['checkpoint'],'remote_hash_host':source['host'],'transfer_host':ready['host'],
            'slurm_job_id':ready['slurm_job_id'],'transport':'Token-protected one-file stream inside CPU allocation; encrypted SSH tunnel; login host is transport only',
            'verified_utc':datetime.now(timezone.utc).isoformat()}
        save(dest/'transfer_receipt.json',receipt);save(dest/'transfer_progress.json',{'stage':'complete','bytes':total,'total':total})
        print(json.dumps(receipt,indent=2),flush=True)
    finally:
        if forward:subprocess.run([*ssh[:-1],'-O','cancel','-L',forward,ssh[-1]],capture_output=True,timeout=30)
        if process and process.poll() is None:process.terminate()
        errorlog.close()


if __name__=='__main__':main()
