"""Bounded decoded benchmarks, isolated OOMs, then full readiness approval."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from common import atomic

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--recipe',required=True)
args=parser.parse_args()
root=args.root
recipe=json.loads((root/'recipes'/f'{args.recipe}.json').read_text())
folder=root/'tuning'/args.recipe
folder.mkdir(parents=True,exist_ok=True)
records=[]


def run(runtime,index,command='benchmark'):
    path=folder/f'runtime-{index}.json'
    atomic(path,runtime)
    result=folder/f'result-{index}.json'
    log=folder/f'{command}-{index}.log'
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
         str(root/'train.py'),command,'--root',str(root),'--recipe',args.recipe,
         '--runtime',str(path),'--result',str(result)]
    with log.open('w') as output:
        process=subprocess.Popen(cmd,cwd=root,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=1200 if command=='profile' else 900)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=124
    record={'runtime':runtime,'returncode':code,'log':str(log),'command':command}
    if code:
        tail=log.read_text(errors='replace')[-20000:]
        record['error_tail']=tail
        records.append(record);atomic(folder/'attempts.json',records)
        if command=='benchmark' and ('out of memory' in tail.lower() or 'cuda error: out of memory' in tail.lower()):
            return None
        raise RuntimeError(f'{command} failed; preserved {log}')
    if command=='benchmark':
        record.update(json.loads(result.read_text()))
        record['eligible']=record['peak_gpu_reserved_bytes']<=.85*record['total_gpu_bytes'] and record['conservative_node_rss_bytes']<=.85*int(os.environ.get('SLURM_MEM_PER_NODE','229376'))*1024**2
    records.append(record);atomic(folder/'attempts.json',records)
    print(json.dumps({k:record[k] for k in ['runtime','command','returncode','samples_per_second','eligible'] if k in record}),flush=True)
    return record


if recipe['method']!='pi05':raise ValueError('Only pi05 pair is authorized')
micros=[1,2,4,8,16]
for micro in micros:
    run(dict(microbatch=micro,workers=4,main_threads=4,prefetch=2),len(records))
eligible=[r for r in records if r.get('eligible')]
if not eligible:raise RuntimeError('No candidate retained 15% GPU and node memory headroom')
best=max(eligible,key=lambda r:r['samples_per_second'])
for workers in (8,12):
    run(dict(microbatch=best['runtime']['microbatch'],workers=workers,main_threads=2,prefetch=2),len(records))
eligible=[r for r in records if r.get('eligible')]
best=max(eligible,key=lambda r:r['samples_per_second'])
atomic(folder/'selection.json',{'selected':best,'control':records[0],
    'throughput_ratio_to_control':best['samples_per_second']/records[0]['samples_per_second'],
    'criterion':'Highest measured decoded throughput retaining at least15% GPU/node memory headroom',
    'recipe_unchanged':True,'sampling_order_unchanged':True})
run(best['runtime'],'final',command='profile')
approval=json.loads((root/'profiles'/f'{args.recipe}.json').read_text())
if not approval.get('approved'):raise RuntimeError('Selected runtime failed full readiness')
print(json.dumps({'recipe':args.recipe,'runtime':best['runtime'],
    'samples_per_second':best['samples_per_second'],'readiness':approval}),flush=True)
