import json, sys, tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

root=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/dom100-palmetto-h20')
assert not (root/'REUSE_REPORT.json').exists(), 'Refusing to replace a prepared family'
root.mkdir(exist_ok=True); (root/'phase_cache').mkdir(exist_ok=True)
def write(item):
    name,raw=item
    (root/name).write_bytes(raw)
count=0; pending=[]
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as tar, ThreadPoolExecutor(max_workers=8) as pool:
    for member in tar:
        path=Path(member.name)
        assert not path.is_absolute() and '..' not in path.parts
        if member.isdir(): continue
        assert member.isfile(), 'Archive links are not allowed'
        assert len(path.parts)==1 or path.parts[0]=='phase_cache'
        pending.append((member.name,tar.extractfile(member).read()))
        if len(pending)==512:
            list(pool.map(write,pending)); count+=len(pending); pending=[]
            if count%10000<512: print(json.dumps({'files_extracted':count}),flush=True)
    list(pool.map(write,pending)); count+=len(pending)
print(json.dumps({'files_extracted':count,'family':str(root)}),flush=True)
