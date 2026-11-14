#!/usr/bin/env python3
"""Verified, resumable public upload of the prepared checkpoint collection."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

from prepare_hf_checkpoint_release import ROOT, STAGE, REPO_ID, SENSITIVE, write


def stamp(): return datetime.now(timezone.utc).isoformat()


def status(stage, **kw):
    write(ROOT/'upload_progress.json', {'stage':stage,'updated_utc':stamp(),**kw})


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(8*1024**2):h.update(block)
    return h.hexdigest()


def prepare_hashes():
    paths=[p for p in sorted(STAGE.rglob('*')) if p.is_file() and not {'.cache','__pycache__'}&set(p.relative_to(STAGE).parts) and p.suffix!='.pyc'
           and p.name not in {'SHA256SUMS','files.json','upload-status.json'}]
    snapshots={str(p.relative_to(STAGE)):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths}
    credential=re.compile(rb'(?i)(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*[\"\x27]([A-Za-z0-9_-]{20,})[\"\x27]')
    for p in paths:
        if p.suffix in {'.pt','.safetensors','.model'}:continue
        text=p.read_bytes()
        if SENSITIVE.search(text) or credential.search(text):raise ValueError('Credential-like literal in staged public file: '+str(p.relative_to(STAGE)))
    previous=json.loads((ROOT/'hash_cache.private.json').read_text()) if (ROOT/'hash_cache.private.json').exists() else {}
    results={};done=0;total=sum(n[0] for n in snapshots.values())
    def one(p):
        key=str(p.relative_to(STAGE));s=snapshots[key];old=previous.get(key,{})
        sha=old['sha256'] if old.get('bytes')==s[0] and old.get('mtime_ns')==s[1] else digest(p)
        assert (p.stat().st_size,p.stat().st_mtime_ns)==s,'Checkpoint changed during hashing'
        return key,{'bytes':s[0],'mtime_ns':s[1],'sha256':sha}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for key,entry in pool.map(one,paths):
            results[key]=entry;done+=entry['bytes']
            status('hashing',files_hashed=len(results),files_total=len(paths),bytes_hashed=done,bytes_total=total)
            if len(results)%20==0 or entry['bytes']>100*1024**2:write(ROOT/'hash_cache.private.json',results)
    write(ROOT/'hash_cache.private.json',results)
    expected={
      'checkpoints/full/krv100/epoch-003-u000196707/training_state.pt':'ab7c4253b1fe68f3e19d7ccc30092bafa12b4674f162c4cb4d1936990e9d3a04',
      'checkpoints/full/krv100/epoch-004-partial-next-u000304483/training_state.pt':'b43c01ad6783e7a3e44b28aac170c79903cbb2dddff2a2ea7909d4dd69a16590',
      'checkpoints/large-audited-197198/domv/epoch-003-u000190656/training_state.pt':'cde3eae0b968b3928dac4c3810806d4e5b7198f15acc60d6878869031637b2f9'}
    for path,sha in expected.items():assert results[path]['sha256']==sha,'Transferred optimizer state hash differs from DeltaAI'
    public={k:{q:v[q] for q in ['bytes','sha256']} for k,v in results.items()}
    write(STAGE/'files.json',{'files':public,'sha256_scope':'All immutable release files; excludes checksum manifests and the mutable upload-status marker.'})
    (STAGE/'SHA256SUMS').write_text(''.join(f'{v["sha256"]}  {k}\n' for k,v in sorted(results.items())))
    return results


def main():
    assert json.loads((ROOT/'preparation.json').read_text())['prepared']
    from huggingface_hub import HfApi
    token=(Path(__file__).resolve().parents[1]/'scratch/dom45/cache/huggingface/token').read_text().strip()
    api=HfApi(token=token);assert api.whoami()['name']=='adi2440'
    checkpoint_count=len(json.loads((STAGE/'model-index.json').read_text())['checkpoints'])
    api.create_repo(REPO_ID,repo_type='model',private=False,exist_ok=True)
    assert not api.model_info(REPO_ID).private,'The requested release must be public'
    write(STAGE/'upload-status.json',{'state':'uploading','complete':False,'checkpoint_count':checkpoint_count,'started_utc':stamp()})
    status('publishing_index_and_usage',repo_id=REPO_ID)
    api.upload_folder(repo_id=REPO_ID,repo_type='model',folder_path=STAGE,
                      allow_patterns=['README.md','model-index.json','environment-*.json','requirements-*.txt','upload-status.json','licenses/**','tools/**'],
                      ignore_patterns=['**/__pycache__/**','**/*.pyc'],commit_message='Publish checkpoint inventory and portable download instructions')
    hashes=prepare_hashes()
    metadata=[str(p.relative_to(STAGE)) for p in STAGE.rglob('*') if p.is_file() and p.suffix not in {'.pt','.safetensors','.pyc'} and not {'.cache','__pycache__'}&set(p.relative_to(STAGE).parts)]
    status('uploading_documentation',repo_id=REPO_ID)
    api.upload_folder(repo_id=REPO_ID,repo_type='model',folder_path=STAGE,allow_patterns=metadata,
                      commit_message='Publish checkpoint index, reproducibility code, recipes, splits, and notices')
    status('uploading_large_scale_checkpoints',repo_id=REPO_ID)
    api.upload_folder(repo_id=REPO_ID,repo_type='model',folder_path=STAGE,allow_patterns=['checkpoints/full/**','checkpoints/large-audited-197198/**'],
                      commit_message='Upload large-scale KRV100 and audited DOMV checkpoints with optimizer states')
    status('uploading_all_retained_checkpoints',repo_id=REPO_ID)
    api.upload_large_folder(repo_id=REPO_ID,repo_type='model',folder_path=STAGE,num_workers=2,
                           print_report=True,print_report_every=60,ignore_patterns=['.cache/**','**/__pycache__/**','**/*.pyc'])
    status('verifying_remote_files',repo_id=REPO_ID)
    remote={f.path:f for f in api.list_repo_tree(REPO_ID,repo_type='model',recursive=True) if hasattr(f,'size')}
    verified=0
    for path,entry in hashes.items():
        assert path in remote and remote[path].size==entry['bytes'],'Missing or truncated remote file: '+path
        lfs=getattr(remote[path],'lfs',None)
        if lfs:
            actual=lfs['sha256'] if isinstance(lfs,dict) else lfs.sha256
            assert actual==entry['sha256'],'Remote content hash differs: '+path
        else:
            data=(STAGE/path).read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            assert remote[path].blob_id==blob,'Remote Git blob differs: '+path
        verified+=1
    completed={'state':'complete','complete':True,'checkpoint_count':checkpoint_count,'verified_files':verified,
               'verified_bytes':sum(x['bytes'] for x in hashes.values()),'completed_utc':stamp(),'repo_id':REPO_ID}
    write(STAGE/'upload-status.json',completed)
    api.upload_file(repo_id=REPO_ID,repo_type='model',path_or_fileobj=STAGE/'upload-status.json',
                    path_in_repo='upload-status.json',commit_message='Verify all checkpoint files and mark release complete')
    status('complete',**completed)
    write(ROOT/'upload_complete.json',completed)
    print(json.dumps(completed),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        status('failed',error_type=type(exc).__name__,error=str(exc)[:500])
        raise
