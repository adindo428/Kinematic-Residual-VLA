"""Read-only result collection; never submit/cancel jobs or transfer weights."""
from pathlib import Path
import subprocess,json,statistics,datetime
from stage import D,DEST
HERE=Path(__file__).resolve().parent
def main():
    subprocess.run(D[:-1]+['-O','check',D[-1]],check=True)
    job=str(json.loads((HERE/'status.json').read_text())['job_id'])
    program='''import pathlib,json,subprocess,datetime
p=pathlib.Path(%r);job=%r
result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':job,'results':{},'exits':{},'errors':{},'production_changed':False}
for kind,pattern in [('results','*-result.json'),('exits','*-exit.json'),('errors','*-error-*.json')]:
 for f in p.glob(pattern):result[kind][f.name]=json.loads(f.read_text())
for key,args in [('squeue',['squeue','-h','-j',job,'-o','%%i|%%T|%%M|%%l|%%R']),('sacct',['sacct','-j',job,'-X','-nP','-o','JobID,State,ExitCode,Elapsed,Start,End'])]:
 r=subprocess.run(args,text=True,capture_output=True);result[key]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
if (p/'sweep-complete.json').exists():result['complete']=json.loads((p/'sweep-complete.json').read_text())
print(json.dumps(result))
'''%(DEST,job)
    import shlex
    raw=json.loads(subprocess.check_output(D+[DEST+'/env/bin/python -c '+shlex.quote(program)],text=True))
    rows=list(raw['results'].values())
    assert all(str(r['slurm_job_id'])==job for r in rows),'Mixed job results; reconcile before comparing'
    def valid(r):
        return r['headroom_pass'] and r['global_batch']==32 and r['world_size']==2 and r['timed_updates']==200 and all(p['prepared_tensors_exact'] and p['loss_allclose'] and p['gradient_tensors']==805 for p in r['equivalence'])
    controls=[r for r in rows if r['variant'] in ('control','control_repeat') and valid(r)]
    candidates=[r for r in rows if not r['variant'].startswith('control') and valid(r)]
    assessment={'job_id':job,'production_changed':False,'measurement_only':True,'complete_repeated_controls':len(controls)==2,'eligible_candidates':[r['variant'] for r in candidates]}
    if len(controls)==2:
        rates=[r['samples_per_second'] for r in controls];median=statistics.median(rates)
        assessment.update(control_median_samples_per_second=median,control_drift_percent=(max(rates)/min(rates)-1)*100)
        if candidates:
            best=max(candidates,key=lambda r:r['samples_per_second']);assessment.update(best_variant=best['variant'],best_samples_per_second=best['samples_per_second'],gain_vs_control_median_percent=(best['samples_per_second']/median-1)*100,gain_vs_fastest_control_percent=(best['samples_per_second']/max(rates)-1)*100)
    raw['assessment']=assessment
    (HERE/'latest-results.json').write_text(json.dumps(raw,indent=2)+'\n')
    print(json.dumps({'squeue':raw['squeue'],'sacct':raw['sacct'],'assessment':assessment},indent=2))
if __name__=='__main__':main()
