from pathlib import Path
import json,subprocess,os,sys
from common import atomic
root=Path(__file__).resolve().parent
records=[]
for name,trim,allocator in [('control',False,''),('trim',True,''),('expandable_trim',True,'expandable_segments:True')]:
    runtime=root/(name+'-runtime.json');atomic(runtime,dict(microbatch=4,workers=4,main_threads=4,prefetch=2))
    result=root/(name+'-result.json');log=root/'logs'/(name+'.log')
    env=dict(os.environ,KRFM_TRIM_INITIAL_CACHE='1' if trim else '0',PYTORCH_CUDA_ALLOC_CONF=allocator)
    with log.open('w') as output:
        r=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(root/'train.py'),'benchmark','--root',str(root),'--recipe','pi05_baseline','--runtime',str(runtime),'--result',str(result)],env=env,cwd=root,stdout=output,stderr=subprocess.STDOUT,timeout=660)
    record={'name':name,'returncode':r.returncode,'trim':trim,'allocator':allocator,'log':str(log)}
    if result.exists():record['benchmark']=json.loads(result.read_text())
    record['memory_stages']=[json.loads(line[len('MEMORY_STAGE '):]) for line in log.read_text(errors='replace').splitlines() if line.startswith('MEMORY_STAGE ')]
    records.append(record);atomic(root/'diagnostic-results.json',records)
    print(json.dumps({k:v for k,v in record.items() if k!='memory_stages'}),flush=True)
    if r.returncode:raise RuntimeError('Diagnostic variant failed; inspect preserved logs')
atomic(root/'diagnostic-complete.json',{'complete':True,'production_submitted':False,'variants':len(records)})
