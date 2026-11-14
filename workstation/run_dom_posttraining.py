"""Sequential quality-first post-training queue; child processes release GPU state."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'workstation'))
from prepare_dom_posttraining import sha


def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def run(root):
    contract_path=root/'launch_contract.json';contract=json.loads(contract_path.read_text())
    reports={family:json.loads((root/f'preflight/{family}.json').read_text()) for family in contract['families']}
    for family,report in reports.items():
        if not report['passed'] or report['original_base']!=contract['checkpoint_preferences'][0] or report['native_sampler_max_error']>2e-5:raise ValueError('Required independent base preflight failed')
    for path,expected in contract['source_hashes'].items():
        if sha(PROJECT/path)!=expected:raise ValueError('Training source changed since launch: '+path)
    (root/'logs').mkdir(exist_ok=True);results=[]
    write(root/'supervisor.json',dict(pid=os.getpid(),state='running',started_unix_s=time.time()))
    try:
        for family in contract['families']:
            if (root/'STOP').exists():break
            command=[str(PROJECT/'workstation/finetune-python.sh'),'-u',str(PROJECT/'workstation/train_dom_posttraining.py'),
                '--contract',str(contract_path),'--family',family,'--output',str(root)]
            with (root/f'logs/{family}.log').open('a',buffering=1) as log:
                process=subprocess.Popen(command,cwd=PROJECT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',WANDB_DIR=str(root)),stdout=log,stderr=subprocess.STDOUT)
                write(root/'queue.json',dict(state='running',active=family,pid=process.pid,order=contract['families'],results=results))
                returncode=process.wait()
            results.append(dict(family=family,returncode=returncode))
            if returncode:raise RuntimeError(f'{family} exited {returncode}; see logs/{family}.log')
        state='stopped' if (root/'STOP').exists() else 'completed'
        write(root/'queue.json',dict(state=state,order=contract['families'],results=results))
        write(root/'supervisor.json',dict(pid=os.getpid(),state=state,finished_unix_s=time.time()))
    except BaseException as error:
        write(root/'queue.json',dict(state='failed',results=results,error=str(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);run(p.parse_args().root.resolve())
