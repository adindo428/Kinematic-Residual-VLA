"""Refresh the final documentation and direct runtime dependencies after upload."""
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime,timezone

from prepare_hf_checkpoint_release import ROOT,STAGE,REPO_ID,write


def main():
    assert json.loads((ROOT/'upload_complete.json').read_text())['complete']
    marker=(STAGE/'upload-status.json').read_bytes()
    attributes=(STAGE/'.gitattributes').read_bytes()
    from write_hf_release_card import main as card
    card();(STAGE/'upload-status.json').write_bytes(marker)
    (STAGE/'.gitattributes').write_bytes(attributes)
    changed={'README.md','environment-native.json','environment-adapters.json','requirements-native.txt','requirements-adapters.txt','tools/checkpoint_tools.py'}
    import shutil
    shutil.copyfile(Path(__file__).with_name('hf_release_checkpoint_tools.py'),STAGE/'tools/checkpoint_tools.py')
    extra=['easydict','scipy','datasets','diffusers','psutil','pyzmq','omegaconf','rich','termcolor','deepdiff','gymnasium']
    work=ROOT.parents[1]
    for flavor,environment in [('native','train'),('adapters','eval20k')]:
        script='import importlib.metadata as m,json;out={};\nfor n in '+repr(extra)+':\n try:out[n]=m.version(n)\n except m.PackageNotFoundError:pass\nprint(json.dumps(out))'
        installed=json.loads(subprocess.check_output([str(work/'envs'/environment/'bin/python'),'-c',script],text=True))
        path=STAGE/f'environment-{flavor}.json';versions=json.loads(path.read_text());versions.update(installed);write(path,versions)
        (STAGE/f'requirements-{flavor}.txt').write_text('# Install CUDA-compatible PyTorch first. Matching LeRobot source is bundled.\n'+'\n'.join(f'{k}=={v.split("+")[0]}' for k,v in versions.items() if k not in {'torch','torchvision','torchcodec','lerobot'})+'\n')
    hashes=json.loads((ROOT/'hash_cache.private.json').read_text())
    for name in changed:
        path=STAGE/name;stat=path.stat();hashes[name]={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    write(ROOT/'hash_cache.private.json',hashes)
    write(STAGE/'files.json',{'files':{k:{q:v[q] for q in ['bytes','sha256']} for k,v in hashes.items()},'sha256_scope':'All immutable release files; excludes checksum manifests and the mutable upload-status marker.'})
    (STAGE/'SHA256SUMS').write_text(''.join(f'{v["sha256"]}  {k}\n' for k,v in sorted(hashes.items())))
    from huggingface_hub import HfApi
    token=(work/'cache/huggingface/token').read_text().strip();api=HfApi(token=token)
    assert api.whoami()['name']=='adi2440'
    api.upload_folder(repo_id=REPO_ID,repo_type='model',folder_path=STAGE,
                      allow_patterns=[*changed,'files.json','SHA256SUMS'],commit_message='Finalize portable runtime dependencies and checkpoint documentation')
    files={f.path:f for f in api.list_repo_tree(REPO_ID,repo_type='model',recursive=True) if hasattr(f,'size')}
    for name in changed|{'files.json','SHA256SUMS'}:
        data=(STAGE/name).read_bytes();assert files[name].size==len(data)
        if files[name].lfs:assert files[name].lfs.sha256==hashlib.sha256(data).hexdigest()
        else:assert files[name].blob_id==hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    complete=json.loads((ROOT/'upload_complete.json').read_text())
    complete['verified_bytes']=sum(v['bytes'] for v in hashes.values())
    complete['metadata_finalized_utc']=datetime.now(timezone.utc).isoformat()
    write(STAGE/'upload-status.json',complete)
    api.upload_file(repo_id=REPO_ID,repo_type='model',path_or_fileobj=STAGE/'upload-status.json',path_in_repo='upload-status.json',commit_message='Confirm finalized release verification')
    write(ROOT/'upload_complete.json',complete)
    write(ROOT/'metadata_finalized.json',{'complete':True,'updated_utc':datetime.now(timezone.utc).isoformat(),'files':sorted(changed)})
    print('Final documentation and runtime dependency files verified',flush=True)


if __name__=='__main__':main()
