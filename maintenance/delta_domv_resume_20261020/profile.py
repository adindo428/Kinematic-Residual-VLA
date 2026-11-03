"""Bounded interactive comparisons; production requires a later explicit submission."""
import json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,os.environ['DVLA_PROJECT_DIR'])
from utils.dom_experiment import atomic_json,digest
HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ['DOM100_ROOT'])

def main():
    job=os.environ['SLURM_JOB_ID'];deadline=time.monotonic()+25*60
    result={'job_id':job,'production_started':False,'complete':False,'cases':[]}
    path=HERE/('profile-results-'+job+'.json')
    def save():atomic_json(path,result)
    def run(phase,env,label,limit):
        log=HERE/'logs'/(label+'-'+phase+'-'+job+'.log')
        with log.open('ab',buffering=0) as stream:
            command=[sys.executable,str(HERE/'check.py')] if phase=='baseline' else ['bash',str(HERE/'launch_ranks.sh'),phase]
            p=subprocess.run(['timeout','--signal=TERM','--kill-after=20s',str(max(1,min(limit,int(deadline-time.monotonic()))))+'s']+command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        return {'exit':p.returncode,'log':str(log)}
    save();env=dict(os.environ,WANDB_MODE='disabled')
    base=run('baseline',env,'8gpu-mb44-w4',15*60+30)
    result['baseline']=base;save()
    if base['exit']:
        result.update(complete=True,error='Baseline readiness failed; production blocked');save();return 1
    result['baseline_approval']=json.loads((HERE/'approval.json').read_text());save()
    selected=json.loads((HERE/'runtime.json').read_text())
    for label,world,mb,workers in [('4gpu-mb88-w4',4,88,4),('8gpu-mb44-w8',8,44,8)]:
        if deadline-time.monotonic()<180:
            result['cases'].append({'label':label,'skipped':'Interactive time limit'});save();continue
        runtime={**selected['runtime'],'microbatch':mb,'workers':workers}
        candidate={'manifest':selected['manifest'],'runtime':runtime,'reason':'Interactive comparison only; canonical checkpoint unchanged'}
        candidate['fingerprint']=digest(candidate)
        runtime_path=HERE/(label+'.json');atomic_json(runtime_path,candidate)
        case={'label':label,'world_size':world,'runtime':runtime};result['cases'].append(case);save()
        case_env=dict(env,DOM100_DIAGNOSTIC_WORLD=str(world),DOM100_DIAGNOSTIC_RUNTIME=str(runtime_path),DOM100_BENCHMARK_DRIVER=str(HERE/'speed_probe.py'))
        case_deadline=min(deadline,time.monotonic()+6*60)
        for phase in ['smoke','benchmark']:
            case[phase]=run(phase,case_env,label,max(1,int(case_deadline-time.monotonic())));save()
            if case[phase]['exit']:break
        if case.get('benchmark',{}).get('exit')==0:
            matches=[json.loads(p.read_text()) for p in (ROOT/'distributed-profiles').glob('*.json')]
            case['profile']=next(p for p in matches if p.get('benchmark_job_id')==job and p.get('runtime')==runtime and p.get('world_size')==world and p.get('comparison_only'))
        save()
    result.update(complete=True,elapsed_limit_minutes=25);save()
    status=json.loads((HERE/'status.json').read_text());status.update(stage='profiling_completed',production_started=False,profile_results=str(path));atomic_json(HERE/'status.json',status)
    print(json.dumps(result),flush=True)
    return 0

if __name__=='__main__':sys.exit(main())
