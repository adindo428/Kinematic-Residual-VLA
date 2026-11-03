"""Revalidate an already completed actual-GPU readiness receipt for Slurm retention."""
import sys,json,hashlib,subprocess,os
from pathlib import Path
import torch
config=json.loads(Path(sys.argv[1]).read_text());r=Path(config['root']);name=config['recipe']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fingerprint(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
p=r/'profiles'/f'{name}.json';assert sha(p)==config['profile_sha256'],'GPU receipt changed'
receipt=json.loads(p.read_text());selection=json.loads((r/'tuning'/name/'selection.json').read_text());manifest=json.loads((r/'source-manifest.json').read_text());prepared=json.loads((r/'prepared.json').read_text());recipe=json.loads((r/'recipes'/f'{name}.json').read_text())
source={p.name:sha(p) for p in r.glob('*.py')};assert source==manifest['files']==receipt['source'];assert fingerprint(source)==manifest['source_sha256']==config['source_sha256']
assert prepared['complete']
for key,namefile in [('statistics_sha256','statistics.json'),('split_sha256','split.json'),('checkpoints_sha256','checkpoints.json'),('implementation_sha256','implementation-lock.json')]:assert sha(r/namefile)==prepared[key]
lock=json.loads((r/'implementation-lock.json').read_text())
for filename,expected in lock['files'].items():assert sha(filename)==expected
assert fingerprint(recipe)==config['recipe_fingerprint'];assert receipt['runtime']==selection['selected']['runtime'];assert receipt['identity']==selection['selected']['identity']
assert receipt['hardware']=='NVIDIA A100 80GB PCIe' and receipt['runtime']['autocast_cache_enabled'] is False and receipt['runtime']['allocator']=='expandable_segments:True'
expected=fingerprint({'recipe':recipe,'prepared':prepared,'source':source,'hardware':receipt['hardware'],'world':2,'torch':torch.__version__,'runtime':receipt['runtime']});assert expected==receipt['identity']
assert receipt['approved'] and receipt['decoded_optimizer_updates']==20
assert receipt['strict_checkpoint_rng_reload'] and receipt['strict_inference_reload'] and receipt['coordinated_checkpoint_test']
assert receipt['peak_gpu_reserved_bytes']<=.85*selection['selected']['total_gpu_bytes'] and receipt['node_process_peak_rss_bytes']<=.85*224*1024**3
account=subprocess.check_output(['sacct','-X','-j',config['gpu_profile_job_id'],'-n','-P','--format=JobIDRaw,State,ExitCode'],text=True)
rows=[line.split('|') for line in account.splitlines() if line.strip()];assert any(v[0]==config['gpu_profile_job_id'] and v[1]=='COMPLETED' and v[2]=='0:0' for v in rows)
result={'approved':True,'gpu_profile_job_id':config['gpu_profile_job_id'],'gpu_profile_state':'COMPLETED','gpu_profile_sha256':config['profile_sha256'],'identity':expected,'source_sha256':manifest['source_sha256'],'recipe':name,'actual_gpu_readiness_revalidated':True,'new_gpu_test_performed':False,'job_id':os.environ.get('SLURM_JOB_ID'),'retention_recovery':True}
if '--check-only' not in sys.argv:
 p=Path(config['receipt']);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));os.replace(tmp,p)
print(json.dumps(result),flush=True)
