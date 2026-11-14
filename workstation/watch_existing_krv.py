"""Observe an already running KRV child without launching a local DOMv duplicate."""
import argparse
import json
import os
from pathlib import Path
import time


def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().split()
        return None if fields[2]=='Z' else fields[21]
    except FileNotFoundError:return None


def watch(root,pid,expected_identity,remote):
    write(root/'supervisor.json',dict(pid=os.getpid(),state='monitoring_existing_krv',training_pid=pid,remote_domv=remote))
    while identity(pid)==expected_identity:
        write(root/'queue.json',dict(state='running',active='krv100',pid=pid,order=['krv100'],remote_domv=remote))
        time.sleep(40)
    record=json.loads((root/'krv100/status.json').read_text())
    state=record.get('state','unknown')
    write(root/'queue.json',dict(state=state,active=None,order=['krv100'],remote_domv=remote,krv100_final_status=record))
    write(root/'supervisor.json',dict(pid=os.getpid(),state=state,remote_domv=remote))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--pid',type=int,required=True)
    parser.add_argument('--identity',required=True);parser.add_argument('--remote-job',required=True);args=parser.parse_args()
    watch(args.root,args.pid,args.identity,dict(cluster='Palmetto',slurm_job_id=args.remote_job,run='/scratch/aparame/Dynamic_VLA/domv-real-v100x2-20261114'))
