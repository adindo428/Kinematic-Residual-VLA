"""One bounded sustained benchmark at the actual allocation's node count."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,os.environ['DVLA_PROJECT_DIR'])
from utils.dom_experiment import atomic_json,digest,file_digest

HERE=Path(__file__).resolve().parent
ROOT=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/krvp-scale-ddp-20261014')
def main():
    lock=(HERE/'scale-check.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    nodes=int(os.environ['SLURM_JOB_NUM_NODES']);world=nodes*4;assert nodes==4 and world==16
    manifest=json.loads((ROOT/'manifest.json').read_text());assert digest({k:v for k,v in manifest.items() if k!='fingerprint'})==manifest['fingerprint']
    for name,expected in manifest['artifacts'].items():assert file_digest(ROOT/name)==expected,name
    runtime={'microbatch':352//world,'workers':4,'prefetch':2,'pin_memory':False,'fused':False,'sampling':'frames','precision':'bf16'}
    candidate={'manifest':manifest['fingerprint'],'runtime':runtime,'reason':'Sustained checkpoint/validation stress and multi-node scaling at fixed global352'}
    candidate['fingerprint']=digest(candidate);path=ROOT/'runtime-candidates'/('scale-'+str(world)+'.json');atomic_json(path,candidate)
    job=os.environ['SLURM_JOB_ID'];receipt=HERE/('scale-check-'+job+'-'+str(world)+'.json')
    result={'job_id':job,'nodes':nodes,'world_size':world,'runtime':runtime,'manifest':manifest['fingerprint'],
        'complete':False,'passed':False,'training_started':False,'variant':'krvp100','diagnostic_stack_dump_mode':'on-crash only'}
    atomic_json(receipt,result)
    env=dict(os.environ,DOM100_RUNTIME_FILE=str(path),DOM100_ROOT=str(ROOT),WANDB_MODE='disabled')
    deadline=time.monotonic()+15*60
    for phase in ['smoke','benchmark']:
        output=ROOT/'logs'/('scale-'+phase+'-'+job+'-'+str(world)+'.log');result[phase+'_log']=str(output)
        with output.open('ab',buffering=0) as stream:
            run=subprocess.run(['timeout','--signal=TERM','--kill-after=30s',str(max(1,int(deadline-time.monotonic())))+'s',
                'bash',str(ROOT/'run_nodes.sh'),phase],env=env,stdout=stream,stderr=subprocess.STDOUT)
        result[phase+'_exit']=run.returncode;atomic_json(receipt,result)
        if run.returncode:
            result.update(complete=True,error=phase+' failed: '+str(run.returncode));atomic_json(receipt,result)
            print(json.dumps(result),flush=True);return 1
    profiles=[json.loads(p.read_text()) for p in (ROOT/'distributed-profiles').glob('*.json')]
    profile=next(p for p in profiles if p['world_size']==world and p['runtime']==runtime and p.get('benchmark_job_id')==job)
    assert digest({k:v for k,v in profile.items() if k!='fingerprint'})==profile['fingerprint']
    assert all(profile.get(k) for k in ['complete','safe','full_data_benchmark','strict_reload','rank_rng_reload','post_training_validation_passed','sustained_reload_passed'])
    assert profile['variant']=='krvp100' and profile['world_size']==16
    assert profile['benchmark_samples']==70400 and profile['measured_updates']==180
    result.update(complete=True,passed=True,profile=profile,elapsed_seconds=15*60-(deadline-time.monotonic()))
    atomic_json(receipt,result);print(json.dumps(result),flush=True);return 0

if __name__=='__main__':sys.exit(main())
