import argparse, concurrent.futures, hashlib, json, os, shutil, time, threading, re
from urllib.parse import quote
import requests
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download, HfApi
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()
root=Path('/work/nvme/bivv/aparameshwaran/dynamicvla')
setup=root/'setup-20261008'
parser=argparse.ArgumentParser()
parser.add_argument('--repo',required=True)
args=parser.parse_args()
record=next(r for r in json.loads((setup/'download-inventory.json').read_text()) if r['repo']==args.repo)
dest=root/'dom100/initialization' if record['kind']=='model' else root/'huggingface/lerobot'/record['repo']
dest.mkdir(parents=True,exist_ok=True)
free=shutil.disk_usage(root).free
assert record['bytes'] < free-100*1024**3, f'Dataset needs {record["bytes"]} bytes; only {free} free with 100GiB reserved for training'
status={'repo':args.repo,'revision':record['revision'],'destination':str(dest),'files_total':len(record['files']),'bytes_total':record['bytes'],'files_verified':0,'bytes_verified':0,'state':'downloading'}
status_file=setup/(args.repo.replace('/','--')+'-status.json')
resolver_lock=threading.Lock()
next_resolver_request=0.0
def throttle_resolver():
    global next_resolver_request
    with resolver_lock:
        scheduled=max(time.monotonic(),next_resolver_request)
        next_resolver_request=scheduled+1/14
    time.sleep(max(0,scheduled-time.monotonic()))
def save():
    status['updated_at']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    temp=status_file.with_suffix('.tmp'); temp.write_text(json.dumps(status,indent=2)); temp.replace(status_file)
def matches(path,e):
    if not path.is_file() or path.stat().st_size!=e['size']: return False
    checksum=hashlib.sha256() if e['sha256'] else hashlib.sha1()
    if not e['sha256']: checksum.update(f'blob {e["size"]}\0'.encode())
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): checksum.update(block)
    return checksum.hexdigest()==(e['sha256'] or e['blob_id'])
def fetch(e):
    target=dest/e['path']
    if matches(target,e): return e
    for attempt in range(8):
        try:
            if record['repo']=='hzxie/DOM':
                # A direct pinned download avoids permanent per-file lock/cache
                # files: the full corpus must fit the project's inode quota.
                target.parent.mkdir(parents=True,exist_ok=True)
                url='https://huggingface.co/datasets/'+record['repo']+'/resolve/'+record['revision']+'/'+quote(e['path'],safe='/')
                if not hasattr(thread_local,'session'): thread_local.session=requests.Session()
                headers={'Authorization':'Bearer '+os.environ['HF_TOKEN']} if os.environ.get('HF_TOKEN') else {}
                throttle_resolver()
                with thread_local.session.get(url,headers=headers,stream=True,timeout=(30,120)) as response:
                    if response.status_code==429:
                        reset=re.search(r';t=(\d+)',response.headers.get('RateLimit',''))
                        wait=int(response.headers.get('Retry-After',reset.group(1) if reset else '60'))
                        print(json.dumps({'rate_limit_wait_seconds':wait+1,'path':e['path']}),flush=True)
                        time.sleep(min(301,max(1,wait+1)))
                        raise RuntimeError('Download rate limit')
                    if response.status_code!=200: raise RuntimeError('HTTP '+str(response.status_code)+' for '+e['path'])
                    tmp=target.with_suffix(target.suffix+'.download.tmp')
                    with tmp.open('wb') as f:
                        for block in response.iter_content(1024**2): f.write(block)
                if not matches(tmp,e): raise ValueError('Checksum mismatch: '+e['path'])
                tmp.replace(target)
            else:
                hf_hub_download(record['repo'],e['path'],repo_type=record['kind'],revision=record['revision'],local_dir=str(dest),force_download=target.exists())
            if not matches(target,e): raise ValueError('Checksum mismatch: '+e['path'])
            return e
        except Exception as error:
            if attempt==7: raise
            print(json.dumps({'retry_path':e['path'],'attempt':attempt+1,'error_type':type(error).__name__}),flush=True)
            time.sleep(min(60,2**attempt))
thread_local=threading.local()
save()
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=8 if record['repo']=='hzxie/DOM' else 4) as pool:
        # Keep the queue bounded for DOM's hundreds of thousands of objects.
        files=record['files']
        for start in range(0,len(files),32):
            for e in pool.map(fetch,files[start:start+32]):
                status['files_verified']+=1; status['bytes_verified']+=e['size']
            save()
            print(json.dumps(status),flush=True)
    if record['kind']=='model':
        config=json.loads((dest/'config.json').read_text())
        backbone=config.get('vlm_model_name','HuggingFaceTB/SmolLM2-360M')
        revision=HfApi().model_info(backbone).sha
        snapshot_download(backbone,revision=revision,allow_patterns=['*.json','*.model','tokenizer*','vocab*','merges*'])
        status['backbone']={'repo':backbone,'revision':revision}
    status['state']='complete'; save()
except Exception as e:
    status['state']='failed'; status['error']=str(e); save(); raise
