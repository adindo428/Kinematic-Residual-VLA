#!/usr/bin/env python3
"""Evaluate the retained final DOM generation using the existing local suite."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / 'scratch/dom45'
ORIGINAL = WORK / 'evaluations/multivla20k-20261029'
ROOT = WORK / 'evaluations/dom-latest-table1-20261031'
TRAIN = WORK / 'training/fresh20k-20261025/family/runs/dom100'
spec = importlib.util.spec_from_file_location('local_suite', ORIGINAL / 'run_local.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
suite.ROOT = ROOT


def upload():
    import wandb
    report = ROOT / 'reports/dom-latest-table1'
    report.mkdir(parents=True, exist_ok=True)
    run = wandb.Api(timeout=30).run('aparame-clemson-university/Kinematic-Residual-VLA/2qh0ijb1')
    for path in (ROOT / 'status.json', ROOT / 'catalog.json', ROOT / 'results/dom100/summary.json'):
        if path.exists():
            dest = report / path.name
            shutil.copy2(path, dest)
            run.upload_file(str(dest), root=str(ROOT))
    suite.write(ROOT / 'upload_status.json', {'state': 'uploaded'})


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    pointer = suite.read(TRAIN / 'latest.json')
    generation = TRAIN / 'generations' / pointer['generation']
    result = ROOT / 'results/dom100'
    checkpoint = result / 'checkpoint'
    checkpoint.mkdir(parents=True, exist_ok=True)
    for name in ('model.safetensors', 'config.json'):
        target = checkpoint / name
        if not target.exists():
            os.link(generation / name, target)
        if suite.sha256(target) != suite.sha256(generation / name):
            raise ValueError('Checkpoint mismatch')
    original_result = ORIGINAL / 'results/dom100'
    shutil.copytree(original_result / 'test_lists', result / 'test_lists', dirs_exist_ok=True)
    catalog = suite.read(ORIGINAL / 'catalog.json')
    entry = dict(catalog['models']['dom100'], label='DOM · fresh 20k (latest epoch 10)',
                 checkpoint=str(checkpoint), checkpoint_sha256=suite.sha256(checkpoint / 'model.safetensors'),
                 progress=pointer['progress'], selection='final/latest retained generation', generation=str(generation))
    catalog['models']['dom100'] = entry
    protocol = suite.read(original_result / 'protocol.json')
    protocol['checkpoint'] = entry
    suite.write(result / 'protocol.json', protocol)
    suite.write(ROOT / 'catalog.json', catalog)
    suite.status('waiting_for_free_gpus', checkpoint=entry['label'])
    while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        time.sleep(30)
    try:
        suite.status('pilot', checkpoint=entry['label'])
        suite.phase(catalog, 'dom100', 'pilot', 0)
        for i, dim in enumerate(suite.DIMS):
            suite.status('evaluating', dimension=dim, checkpoint=entry['label'])
            suite.phase(catalog, 'dom100', 'full', i)
            try:
                upload()
            except Exception as error:
                suite.write(ROOT / 'upload_status.json', {'state': 'failed', 'error': str(error)})
        suite.summarize(catalog, 'dom100')
        suite.status('complete', verified_trials=360, checkpoint=entry['label'])
    except Exception as error:
        suite.status('failed', error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            upload()
        except Exception as error:
            suite.write(ROOT / 'upload_status.json', {'state': 'failed', 'error': str(error)})


if __name__ == '__main__':
    main()
