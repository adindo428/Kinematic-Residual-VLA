"""Benchmark/readiness gates followed by the explicitly ordered training queue."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

PROJECT=Path(__file__).resolve().parents[1]
ORDER=['krv100','domv','dom','smolvla_baseline','smolvla_kr']


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    root=args.root.resolve();(root/'logs').mkdir(exist_ok=True);(root/'benchmarks').mkdir(exist_ok=True)
    q=dict(state='preflight',order=ORDER,models={f:dict(state='pending') for f in ORDER},pid=os.getpid())
    def save():q['updated_utc']=datetime.now(timezone.utc).isoformat();write(root/'queue.json',q)
    def run(family,stage,command,env=None):
        log=root/'logs'/f'{family}-{stage}.log';q.update(state=stage,active=family,log=str(log));q['models'][family]['state']=stage;save()
        with log.open('w') as stream:
            child=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            q['child_pid']=child.pid;save();code=child.wait()
        if code:
            q.update(state='failed',failed_stage=stage,returncode=code);save()
            raise RuntimeError(f'{family} {stage} failed ({code}); see {log}')
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
             PYTHONPATH=str(PROJECT)+':'+str(PROJECT/'third_party/DynamicVLA')+':'+os.environ.get('PYTHONPATH',''))
    family_wrapper=str(PROJECT/'workstation/family-python.sh');dom_wrapper=str(PROJECT/'workstation/finetune-python.sh')
    for family in ORDER[:3]:
        run(family,'benchmark',[dom_wrapper,str(PROJECT/'workstation/profile_static_dom.py'),
            '--contract',str(root/'dom-contract.json'),'--family',family,'--output',str(root/'benchmarks'/f'{family}.json')],env)
    campaign=json.loads((root/'family-campaign.json').read_text())
    for family in ORDER[3:]:
        run(family,'benchmark',[family_wrapper,str(PROJECT/'workstation/profile_family_performance.py'),
            '--campaign',str(root/'family-campaign.json'),'--family',family,'--output',str(root/'benchmarks'/f'{family}.json')],env)
        report=json.loads((root/'benchmarks'/f'{family}.json').read_text())
        if not report.get('passed'):raise RuntimeError('Benchmark did not pass')
        campaign['family_performance'][family]=report['selected']
    campaign['benchmark_before_training']=True;campaign['run_prefix']='static-continuation-constant'
    campaign['benchmark_manifest']={f:dict(sha256=hashlib.sha256((root/'benchmarks'/f'{f}.json').read_bytes()).hexdigest()) for f in ORDER[3:]}
    write(root/'family-campaign.json',campaign)
    for family in ORDER[3:]:
        run(family,'readiness',[family_wrapper,str(PROJECT/'workstation/train_family_finetune.py'),
            '--campaign',str(root/'family-campaign.json'),'--family',family,'--mode','check','--output',str(root/f'{family}-gpu-readiness.json')],env)
    for family in ORDER:
        if family in ORDER[:3]:
            command=[dom_wrapper,str(PROJECT/'workstation/train_real_finetune.py'),
                '--contract',str(root/'dom-contract.json'),'--output',str(root/'dom-runs')]
            training_env=dict(env,REAL_FINETUNE_FAMILY=family)
        else:
            command=[family_wrapper,str(PROJECT/'workstation/train_family_finetune.py'),
                '--campaign',str(root/'family-campaign.json'),'--family',family,'--mode','train','--output',str(root/'runs'/family)]
            training_env=env
        run(family,'training',command,training_env)
        q['models'][family]['state']='completed';save()
    q.update(state='completed',active=None);save()


if __name__=='__main__':main()
