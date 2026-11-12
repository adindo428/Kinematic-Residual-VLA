"""Save all per-checkpoint benchmarks before starting the remaining training jobs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

import psutil


def atomic_json(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def benchmark_matches(report,entry):
    return (report.get('state')=='completed' and report.get('passed') is True
        and report.get('family')==entry['family'] and report.get('checkpoint_sha256')==entry['sha256']
        and report.get('effective_batch')==16 and bool(report.get('selected'))
        and report.get('architecture_changes') is False and report.get('weights_saved') is False
        and bool(report.get('architecture_signature'))
        and report.get('velocity_conditioning')==entry.get('velocity_conditioning'))


def commit_profiles(campaign,reports):
    for entry in campaign['entries']:
        if not benchmark_matches(reports.get(entry['family'],{}),entry):
            raise ValueError('Missing successful benchmark for '+entry['family'])
    result=copy.deepcopy(campaign);result['benchmark_before_training']=True
    result['family_performance']={entry['family']:copy.deepcopy(reports[entry['family']]['selected']) for entry in result['entries']}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--adopt-family');parser.add_argument('--adopt-pid',type=int)
    parser.add_argument('--adopt-create-time',type=float)
    args=parser.parse_args();campaign=json.loads(args.campaign.read_text())
    root=args.campaign.parent.resolve();project=Path(__file__).resolve().parents[1]
    queue={'state':'waiting_for_current_campaign','models':{e['family']:{'state':'pending_benchmark'} for e in campaign['entries']}}
    def save():atomic_json(root/'queue.json',queue)
    def read(path):return json.loads(path.read_text()) if path.exists() else {}
    def gpu_ready(family):
        required=12000 if family.startswith('smolvla') else 18000
        while True:
            try:free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            except (FileNotFoundError,subprocess.CalledProcessError,ValueError) as exc:
                queue.update(state='gpu_unavailable',error=str(exc));save();return False
            if free>=required:return True
            queue.update(state='waiting_for_gpu_memory',active=family,free_mib=free,required_mib=required);save();time.sleep(30)
    save()
    for file,expected in campaign['source_hashes'].items():
        if hashlib.sha256((project/file).read_bytes()).hexdigest()!=expected:
            queue.update(state='source_changed',file=file);save();return
    if campaign.get('current_queue_dependency'):
        dependency=Path(campaign['current_queue_dependency'])
        while True:
            status=read(dependency/'final-evaluation-status.json')
            if status.get('state')=='completed':break
            if status.get('state') in ['training_failed','evaluation_failed','wait_timeout']:
                queue.update(state='dependency_failed',dependency_status=status);save();return
            time.sleep(30)
    elif campaign.get('prior_campaign_completion',{}).get('state')!='completed':
        raise RuntimeError('Missing prior campaign completion evidence')
    if args.adopt_family:
        entry=next(e for e in campaign['entries'] if e['family']==args.adopt_family)
        report=read(root/'benchmarks'/f'{args.adopt_family}.json')
        if not benchmark_matches(report,entry):raise RuntimeError('Active adopted run must already have a saved benchmark')
        queue.update(state='running',active=args.adopt_family,mode='train',adopted_pid=args.adopt_pid)
        queue['models'][args.adopt_family]={'state':'train'};save()
        while True:
            try:
                process=psutil.Process(args.adopt_pid)
                if args.adopt_create_time is not None and abs(process.create_time()-args.adopt_create_time)>.01:break
                if process.status()==psutil.STATUS_ZOMBIE:break
                command=process.cmdline()
                if not any(x.endswith('train_family_finetune.py') for x in command) or args.adopt_family not in command:
                    raise RuntimeError('Adopted process identity differs')
            except psutil.NoSuchProcess:break
            time.sleep(15)
        status=read(root/'runs'/args.adopt_family/'status.json')
        if status.get('state')!='completed':
            queue.update(state='adopted_run_failed',details=status);save();return
        queue['models'][args.adopt_family]={'state':'completed'};save()

    # Global gate: persist every remaining checkpoint's benchmark before
    # launching any of the remaining training processes.
    reports={};failures=[]
    for entry in campaign['entries']:
        family=entry['family'];path=root/'benchmarks'/f'{family}.json';report=read(path)
        if benchmark_matches(report,entry):
            reports[family]=report
            queue['models'][family]={'state':'completed' if read(root/'runs'/family/'status.json').get('state')=='completed' else 'benchmarked'}
            save();continue
        queue.update(state='waiting_for_download',active=family,mode='benchmark');save()
        while True:
            downloads=read(root/'downloads.json');status=downloads.get('models',{}).get(family,{})
            if status.get('state')=='verified':break
            if status.get('state')=='failed' or (not status and any(x.get('state')=='failed' for x in downloads.get('models',{}).values())):break
            time.sleep(30)
        if status.get('state')!='verified':
            failures.append(family);queue['models'][family]={'state':'download_failed'};save();continue
        if not gpu_ready(family):return
        path.parent.mkdir(parents=True,exist_ok=True)
        queue.update(state='benchmarking',active=family);queue['models'][family]={'state':'benchmark'};save()
        with (root/f'{family}-benchmark.log').open('w') as log:
            result=subprocess.run([str(project/'workstation/family-python.sh'),str(project/'workstation/profile_family_performance.py'),
                '--campaign',str(args.campaign.resolve()),'--family',family,'--output',str(path)],cwd=project,stdout=log,stderr=subprocess.STDOUT)
        report=read(path)
        if result.returncode or not benchmark_matches(report,entry):
            report.update(state='failed',passed=False,returncode=result.returncode);atomic_json(path,report)
            failures.append(family);queue['models'][family]={'state':'benchmark_failed','report':str(path)};save();continue
        reports[family]=report;queue['models'][family]={'state':'benchmarked','report':str(path)};save()
    if failures:
        queue.update(state='benchmark_failed',failed_models=failures,training_started=False);save();return
    campaign=commit_profiles(campaign,reports)
    campaign['benchmark_manifest']={family:dict(path=str(root/'benchmarks'/f'{family}.json'),
        sha256=hashlib.sha256((root/'benchmarks'/f'{family}.json').read_bytes()).hexdigest()) for family in reports}
    atomic_json(args.campaign,campaign)
    atomic_json(root/'benchmark-summary.json',{family:dict(selected=r['selected'],compute=r.get('selected_compute'),loader=r.get('selected_loader')) for family,r in reports.items()})
    queue.update(state='benchmarks_completed',mode='training_ready');save()
    for entry in campaign['entries']:
        family=entry['family'];run_output=root/'runs'/family
        if read(run_output/'status.json').get('state')=='completed':queue['models'][family]={'state':'completed'};save();continue
        if not gpu_ready(family):return
        for mode in ['check','train']:
            output=root/f'{family}-gpu-readiness.json' if mode=='check' else run_output
            output.parent.mkdir(parents=True,exist_ok=True)
            queue.update(state='running',active=family,mode=mode);queue['models'][family]={'state':mode};save()
            command=[str(project/'workstation/family-python.sh'),str(project/'workstation/train_family_finetune.py'),
                '--campaign',str(args.campaign.resolve()),'--family',family,'--mode',mode,'--output',str(output)]
            if mode=='train' and output.exists():command+=['--resume']
            with (root/f'{family}-{mode}.log').open('w') as log:
                result=subprocess.run(command,cwd=project,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                queue['models'][family]=dict(state=mode+'_failed',returncode=result.returncode,log=str(root/f'{family}-{mode}.log'));save();break
            if mode=='train':queue['models'][family]={'state':'completed'};save()
    queue.update(state='completed' if all(x['state']=='completed' for x in queue['models'].values()) else 'completed_with_failures');save()


if __name__=='__main__':main()
