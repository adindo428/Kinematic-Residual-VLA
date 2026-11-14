"""Wait for completed constant-gate DOMV/KRV training and prepare launch commands.

This preparation opens no cameras or robot devices and starts no inference.
"""
import argparse
import json
from pathlib import Path
import shlex
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
DEFAULT_ROOT=PROJECT/'scratch/real_robot/finetuning-runs/static-continuation-constant-20261112'


def pair_status(root, task='striped_ball'):
    root=Path(root).resolve();run=root/'dom-runs';models={}
    for family in ('domv','krv100'):
        training=run/family
        status_path=training/'status.json'
        status=json.loads(status_path.read_text()) if status_path.exists() else {'state':'pending'}
        record=dict(state=status['state'],epoch=status.get('epoch'))
        if status['state']=='completed':
            pointer=json.loads((training/'best_checkpoint.json').read_text())
            source=Path(pointer['path']).resolve()
            if source.parent!=training.resolve():raise ValueError('Checkpoint outside model run')
            provenance=json.loads((source/'provenance.json').read_text())
            cfg=json.loads((source/'config.json').read_text())
            if not provenance.get('completed_epoch') or not (source/'model.safetensors').is_file():
                raise ValueError('Incomplete selected checkpoint')
            if cfg.get('residual_phase_gate_mode')!='constant':raise ValueError('Expected trained constant gate')
            if not cfg.get('condition_on_object_velocity') or bool(cfg.get('enable_residual_flow'))!=(family=='krv100'):
                raise ValueError('Unexpected native policy conditioning or residual configuration')
            if pointer['epoch']!=provenance['epoch']:raise ValueError('Checkpoint epoch mismatch')
            record.update(selected_epoch=pointer['epoch'],source=str(source),gating='constant',
                residual_flow=bool(cfg['enable_residual_flow']))
        models[family]=record
    ready=all(row['state']=='completed' for row in models.values())
    result=dict(ready=ready,models=models,hardware_accessed=False)
    if ready:
        result['commands']={family:shlex.join([str(PROJECT/'.venv/bin/python'),'-m','real_robot.evaluate',
            '--run',str(run),'--model',family,'--checkpoint','best','--task',task,
            '--velocity-innovation','on','--record-inputs','--new-mask','--max-trials','1','--execute'])
            for family in models}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--task',choices=['striped_ball','marker','orange'],default='striped_ball')
    parser.add_argument('--wait',action='store_true',help='Poll training records every 30 seconds; only print commands when ready')
    args=parser.parse_args()
    while True:
        result=pair_status(args.root,args.task)
        if result['ready'] or not args.wait:
            print(json.dumps(result,indent=2));return 0 if result['ready'] else 2
        if any(row['state']=='failed' for row in result['models'].values()):
            print(json.dumps(result,indent=2));return 2
        print('Waiting: '+', '.join(f'{key}={value["state"]}' for key,value in result['models'].items()),file=sys.stderr,flush=True)
        time.sleep(30)


if __name__=='__main__':sys.exit(main())
