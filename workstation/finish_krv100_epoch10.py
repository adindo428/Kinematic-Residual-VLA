#!/usr/bin/env python3
"""User-authorized epoch-six continuation with validation stopping disabled."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=Path,required=True)
    args=parser.parse_args()
    stage=args.stage.resolve()
    request=json.loads((stage/'finish_epoch10_request.json').read_text())
    if not request.get('user_authorized') or request['target_epoch']!=10 or not request['disable_early_stopping']:
        raise ValueError('Missing explicit continuation override')
    source=Path(request['source'])
    sys.path.insert(0,str(stage/'controller'))
    sys.path.insert(0,str(source))
    import torch.distributed as dist
    import core.dom_distributed_training as training
    from utils.dom_experiment import atomic_json,file_digest
    if file_digest(__file__)!=request['launcher_sha256']:
        raise ValueError('Authorized continuation launcher changed')
    original_validation=training.record_validation
    def record_validation(progress,metrics,rule):
        original_validation(progress,metrics,rule)
        progress['user_finish_epoch10_override']={
            'request_sha256':file_digest(stage/'finish_epoch10_request.json'),
            'early_stopping_disabled':True,'target_epoch':10}
        return False
    training.record_validation=record_validation
    original_train=training.train_distributed
    def train(*positional,**keywords):
        keywords['resume_stopped']=True
        return original_train(*positional,**keywords)
    training.train_distributed=train
    original_save=training.save_distributed
    def save(root,*positional,**keywords):
        if dist.get_rank()==0:
            path=Path(root)/'provenance.json'
            provenance=json.loads(path.read_text())
            provenance['user_finish_epoch10_override']=request
            atomic_json(path,provenance)
        return original_save(root,*positional,**keywords)
    training.save_distributed=save
    import wandb
    original_init=wandb.init
    def init(*positional,**keywords):
        run=original_init(*positional,**keywords)
        run.config.update({'finish_epoch10_override':request},allow_val_change=True)
        return run
    wandb.init=init
    spec=importlib.util.spec_from_file_location('frozen_continuation',stage/'controller/continue_krv100_20k.py')
    controller=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    return controller.train(argparse.Namespace(stage=stage,source=source))


if __name__=='__main__':
    try:
        code=main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    sys.exit(code)
