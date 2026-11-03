#!/usr/bin/env python3
"""Submit independent four/eight-GPU runs, prioritizing KRVLA+velocity."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.dom_training import load_family
from core.dom_checkpoint import run_lock
from utils.dom_experiment import VARIANTS, atomic_json, digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family-root',type=Path,required=True)
    parser.add_argument('--variants',choices=VARIANTS,nargs='+',default=['krv100'])
    parser.add_argument('--gpus',type=int,choices=(4,8),default=8)
    parser.add_argument('--gpu',choices=('auto','h100','h200','a100'),default='auto')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();root=args.family_root.resolve();project=Path(__file__).resolve().parents[1]
    manifest=load_family(root)
    recipe=json.loads((root/'recipe.json').read_text());check=dict(recipe);fingerprint=check.pop('fingerprint')
    if digest(check)!=fingerprint or recipe['manifest']!=manifest['fingerprint']:
        raise ValueError('Recipe differs from the frozen distributed family')
    if len(set(args.variants))!=len(args.variants):raise ValueError('Duplicate variants requested')
    commands=[]
    for variant in args.variants:
        command=['sbatch','--parsable','--chdir',str(project),'--job-name='+variant,
            '--export=ALL,DOM100_VARIANT='+variant,'--gpus='+((str(args.gpus)) if args.gpu=='auto' else args.gpu+':'+str(args.gpus))]
        if args.gpu!='auto':command.append('--constraint='+('gpu_a100_80gb' if args.gpu=='a100' else 'gpu_'+args.gpu))
        command.append(str(project/'slurm/dom100_distributed.sbatch'))
        commands.append((variant,command))
    if args.dry_run:
        print(json.dumps({'commands':[command for _,command in commands],'dependencies':[],'smoke_before_training':True},indent=2));return
    env=os.environ.copy();env.update(DVLA_PROJECT_DIR=str(project),DOM100_ROOT=str(root))
    with run_lock(root/'submissions'):
        path=root/'submissions/distributed.json'
        jobs=json.loads(path.read_text()) if path.exists() else {}
        for variant,command in commands:
            if variant in jobs:
                if jobs[variant]['world_size']!=args.gpus:raise ValueError('An existing submission uses a different world size')
                continue
            job=subprocess.check_output(command,env=env,text=True).strip().split(';')[0]
            if not job.isdigit():raise RuntimeError('Invalid Slurm job ID')
            jobs[variant]={'job_id':job,'world_size':args.gpus,'gpu_selection':args.gpu,'wandb_name':variant}
            atomic_json(path,jobs)
        print(json.dumps(jobs,indent=2))


if __name__=='__main__':main()
