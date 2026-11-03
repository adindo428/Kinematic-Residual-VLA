import json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from huggingface_hub import HfApi

root = Path('/work/nvme/bivv/aparameshwaran/dynamicvla')
out = root / 'setup-20261008'
out.mkdir(exist_ok=True)
api = HfApi(token=os.environ.get('HF_TOKEN'))
repos = [('hzxie/dynamic-vla-DOM','model'), ('adi2440/dom-franka-tracking','dataset'), ('hzxie/DOM','dataset')]
inventory = []
for repo, kind in repos:
    info = api.repo_info(repo, repo_type=kind)
    entries=[]
    def tree(prefix):
        return list(api.list_repo_tree(repo, repo_type=kind, revision=info.sha, path_in_repo=prefix or None, recursive=True))
    if repo=='hzxie/DOM':
        prefixes=['meta']
        for parent in ('data','videos'):
            prefixes += [e.path for e in api.list_repo_tree(repo,repo_type=kind,revision=info.sha,path_in_repo=parent) if not hasattr(e,'size')]
        pool=ThreadPoolExecutor(max_workers=8)
        futures=[pool.submit(tree,p) for p in prefixes]
        def all_entries():
            for n,future in enumerate(as_completed(futures),1):
                yield from future.result()
                if n%20==0:
                    progress={'repo':repo,'revision':info.sha,'chunks_scanned':n,'chunks_total':len(prefixes),'selected_files':len(entries),'selected_bytes':sum(e['size'] for e in entries)}
                    (out/'dom-inventory-progress.json').write_text(json.dumps(progress))
                    print(json.dumps(progress),flush=True)
        iterator=all_entries()
    else: iterator=tree('')
    for e in iterator:
        if not hasattr(e,'size'): continue
        p=e.path
        selected = (p in ('config.json','model.safetensors') if kind=='model' else
                    p.startswith('meta/') or (p.startswith('data/') and p.endswith('.parquet')) or
                    (p.startswith('videos/') and p.endswith('.mp4') and any('/'+c+'/' in p for c in ('observation.images.opst_cam','observation.images.wrist_cam'))))
        if selected:
            lfs=getattr(e,'lfs',None)
            entries.append({'path':p,'size':e.size,'blob_id':e.blob_id,'sha256':getattr(lfs,'sha256',None) if lfs else None})
    record={'repo':repo,'kind':kind,'revision':info.sha,'files':entries,'bytes':sum(e['size'] for e in entries)}
    if repo=='hzxie/DOM': pool.shutdown()
    inventory.append(record)
    (out/'download-inventory.json').write_text(json.dumps(inventory))
    print(json.dumps({k:v for k,v in record.items() if k!='files'}), 'files',len(entries),flush=True)
