"""Reconcile Hub-generated LFS attributes and verify the complete public release."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from prepare_hf_checkpoint_release import ROOT, STAGE, REPO_ID, write
from upload_hf_checkpoint_release import stamp, status


def main():
    hashes=json.loads((ROOT/'hash_cache.private.json').read_text())
    api=HfApi(token=(ROOT.parents[1]/'cache/huggingface/token').read_text().strip())
    assert api.whoami()['name']=='adi2440'
    assert not api.model_info(REPO_ID).private
    attrs=Path(hf_hub_download(REPO_ID,'.gitattributes',token=False,force_download=True)).read_bytes()
    suffix=' filter=lfs diff=lfs merge=lfs -text'
    for line in attrs.decode().splitlines():
        assert line.endswith(suffix),'Unexpected generated Git attribute'
        assert line[:-len(suffix)] in hashes or line[:-len(suffix)] in {'*.pt','*.safetensors'}
    (ROOT/'hf_generated_attributes.txt').write_bytes(attrs)
    p=STAGE/'.gitattributes';p.write_bytes(attrs);s=p.stat()
    hashes['.gitattributes']={'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':hashlib.sha256(attrs).hexdigest()}
    write(ROOT/'hash_cache.private.json',hashes)
    write(STAGE/'files.json',{'files':{k:{q:v[q] for q in ['bytes','sha256']} for k,v in hashes.items()},'sha256_scope':'All immutable release files; excludes checksum manifests and the mutable upload-status marker.'})
    (STAGE/'SHA256SUMS').write_text(''.join(f'{v["sha256"]}  {k}\n' for k,v in sorted(hashes.items())))
    api.upload_folder(repo_id=REPO_ID,folder_path=STAGE,allow_patterns=['.gitattributes','files.json','SHA256SUMS'],commit_message='Reconcile Hub-generated large-file rules with release checksums')
    status('verifying_remote_files',repo_id=REPO_ID)
    remote={f.path:f for f in api.list_repo_tree(REPO_ID,recursive=True) if hasattr(f,'size')}
    for name,entry in hashes.items():
        assert name in remote and remote[name].size==entry['bytes'],'Missing or truncated: '+name
        f=remote[name]
        if f.lfs:
            actual=f.lfs['sha256'] if isinstance(f.lfs,dict) else f.lfs.sha256
            assert actual==entry['sha256'],'Content hash differs: '+name
        else:
            data=(STAGE/name).read_bytes()
            assert f.blob_id==hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),'Git hash differs: '+name
    for name in ['files.json','SHA256SUMS']:
        data=(STAGE/name).read_bytes();f=remote[name]
        assert f.size==len(data)
        assert f.blob_id==hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    completed={'state':'complete','complete':True,'checkpoint_count':len(json.loads((STAGE/'model-index.json').read_text())['checkpoints']),
               'verified_files':len(hashes),'verified_bytes':sum(x['bytes'] for x in hashes.values()),'completed_utc':stamp(),'repo_id':REPO_ID,
               'reconciled_generated_attributes':True}
    write(STAGE/'upload-status.json',completed)
    api.upload_file(repo_id=REPO_ID,path_or_fileobj=STAGE/'upload-status.json',path_in_repo='upload-status.json',commit_message='Verify complete checkpoint release')
    write(ROOT/'upload_complete.json',completed);status('complete',**completed)
    print(json.dumps(completed),flush=True)


if __name__=='__main__':main()
