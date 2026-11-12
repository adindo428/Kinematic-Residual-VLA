"""Run verified native-family checks/training after the existing queue/evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True);args=parser.parse_args()
    campaign=json.loads(args.campaign.read_text());root=args.campaign.parent.resolve();project=Path(__file__).resolve().parents[1]
    queue={'state':'waiting_for_current_campaign','models':{e['family']:{'state':'pending'} for e in campaign['entries']}}
    def save():
        temp=root/'queue.tmp';temp.write_text(json.dumps(queue,indent=2)+'\n');temp.replace(root/'queue.json')
    save();dependency=Path(campaign['current_queue_dependency'])
    while True:
        p=dependency/'final-evaluation-status.json'
        status=json.loads(p.read_text()) if p.exists() else {}
        if status.get('state')=='completed':break
        if status.get('state') in ['training_failed','evaluation_failed','wait_timeout']:
            queue.update(state='dependency_failed',dependency_status=status);save();return
        time.sleep(30)
    for file,expected in campaign['source_hashes'].items():
        if hashlib.sha256((project/file).read_bytes()).hexdigest()!=expected:
            queue.update(state='source_changed',file=file);save();return
    for entry in campaign['entries']:
        family=entry['family'];queue.update(state='waiting_for_download',active=family);save()
        run_output=root/'runs'/family
        run_status=run_output/'status.json'
        if run_status.exists() and json.loads(run_status.read_text()).get('state')=='completed':
            queue['models'][family]={'state':'completed'};save();continue
        while True:
            p=root/'downloads.json';downloads=json.loads(p.read_text()) if p.exists() else {}
            status=downloads.get('models',{}).get(family,{})
            if status.get('state')=='verified':break
            if status.get('state')=='failed':
                queue['models'][family]=dict(state='download_failed',details=status);save();break
            if not status and any(x.get('state')=='failed' for x in downloads.get('models',{}).values()):
                queue['models'][family]=dict(state='download_dependency_failed',details='Downloader stopped before reaching this checkpoint');save();break
            time.sleep(30)
        if status.get('state')!='verified':continue
        required_mib=12000 if family.startswith('smolvla') else 18000
        while True:
            try:
                free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            except (FileNotFoundError,subprocess.CalledProcessError,ValueError) as exc:
                queue.update(state='gpu_unavailable',error=str(exc));save();return
            if free>=required_mib:break
            queue.update(state='waiting_for_gpu_memory',active=family,free_mib=free,required_mib=required_mib);save()
            time.sleep(30)
        # Child checks fail closed on strict loading, architecture, gradients,
        # inference parity or device memory; an unverified model never trains.
        for mode in ['check','train']:
            output=root/f'{family}-gpu-readiness.json' if mode=='check' else root/'runs'/family
            output.parent.mkdir(parents=True,exist_ok=True)
            queue.update(state='running',active=family,mode=mode);queue['models'][family]={'state':mode};save()
            with (root/f'{family}-{mode}.log').open('w') as log:
                command=[str(project/'workstation/family-python.sh'),str(project/'workstation/train_family_finetune.py'),
                    '--campaign',str(args.campaign.resolve()),'--family',family,'--mode',mode,'--output',str(output)]
                if mode=='train' and output.exists():command+=['--resume']
                result=subprocess.run(command,
                    cwd=project,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                queue['models'][family]=dict(state=mode+'_failed',returncode=result.returncode,log=str(root/f'{family}-{mode}.log'));save();break
            if mode=='train':queue['models'][family]={'state':'completed'};save()
    queue.update(state='completed' if all(x['state']=='completed' for x in queue['models'].values()) else 'completed_with_failures')
    save()


if __name__=='__main__':main()
