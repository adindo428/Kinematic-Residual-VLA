#!/usr/bin/env python3
"""Evaluate the saved partial-epoch checkpoint after the best-checkpoint suite."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import task_aligned_table1 as suite


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    request=suite.read(root/'request.json')
    old_validate=suite.validate
    def validate(request):
        if (not request.get('user_authorized') or request.get('checkpoint_selection')!='latest_saved_after_user_stop'
                or request['target_epoch']!=8 or request['expected_trials']!=360
                or request['partition']!='new_task_aligned_scenarios'):
            raise ValueError('Missing authorization for the saved 8.09-epoch checkpoint')
        for path,expected in request['files_sha256'].items():
            if suite.sha256(path)!=expected:raise ValueError(f'Frozen evaluation file changed: {path}')
        for test in request['cases']:
            if (suite.sha256(test['path'])!=test['config_sha256']
                    or suite.sha256(test['parent_config'])!=test['parent_config_sha256']):
                raise ValueError('Comparison scenarios changed')
        for path,expected in request['asset_sha256'].items():
            if suite.sha256(path)!=expected:raise ValueError('Scenario asset changed')
        prior=suite.read(Path(request['prior_results'])/'request.json')
        if prior['cases']!=request['cases'] or prior['dimension_order']!=request['dimension_order']:
            raise ValueError('Checkpoint comparison must use identical scenarios')
        sys.path.insert(0,request['policy_source'])
        from core.dom_training import load_family
        manifest=load_family(request['family_root'])
        if manifest['fingerprint']!=request['manifest']:raise ValueError('Training family changed')
        return manifest
    suite.validate=validate
    def dependency(request):
        prior=Path(request['prior_results'])
        status=suite.read(prior/'suite_status.json')
        if status['state']=='failed':
            raise RuntimeError('Prior checkpoint evaluation failed; fix it before running the paired comparison')
        summary=prior/'summary.json'
        if not summary.is_file() or not suite.read(summary).get('complete'):return None
        if suite.read(summary)['verified_trials']!=360:raise ValueError('Prior suite is incomplete')
        for service in (request['prior_service'],request['training_service']):
            active=subprocess.run(['systemctl','--user','is-active',service],capture_output=True,
                                  text=True,check=False).stdout.strip()
            if active in ('active','activating','deactivating'):return None
        receipt=suite.read(request['stop_receipt'])
        if suite.sha256(request['stop_receipt'])!=request['stop_receipt_sha256'] or not receipt['checkpoint_saved']:
            raise ValueError('Saved training-stop receipt changed')
        pointer=suite.read(Path(request['family_root'])/'runs/krv100/latest.json')
        if pointer!=request['selected_pointer'] or pointer!=receipt['latest']:
            raise ValueError('Training advanced or the selected latest checkpoint changed')
        checkpoint=Path(request['selected_checkpoint'])
        for name,expected in request['selected_files_sha256'].items():
            if suite.sha256(checkpoint/name)!=expected:raise ValueError('Selected checkpoint changed')
        return checkpoint
    suite.dependency=dependency
    old_state=suite.state
    def state(root,phase,**details):
        if phase=='waiting_training_stop':phase='waiting_epoch3_evaluation'
        return old_state(root,phase,checkpoint_label=request['checkpoint_label'],**details)
    suite.state=state
    old_bind=suite.bind
    def bind(request,root,checkpoint):
        protocol=old_bind(request,root,checkpoint)
        protocol.update(checkpoint_label=request['checkpoint_label'],
            effective_training_epochs=request['effective_training_epochs'],
            training_progress_at_checkpoint=request['selected_pointer']['progress'],
            comparison_checkpoint_results=request['prior_results'])
        suite.write_json(root/'protocol.json',protocol)
        return protocol
    suite.bind=bind
    old_report=suite.report
    def report(root,request):
        old_report(root,request)
        label='KRV100 '+request['checkpoint_label']
        path=root/'table1_task_subset.csv'
        with path.open() as stream:
            reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
        rows[0]['Method']=label
        with path.open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        path=root/'table1_task_subset.md'
        path.write_text(path.read_text().replace('KRV100 epoch 8',label))
        first=suite.read(Path(request['prior_results'])/'summary.json')['observed']
        last=suite.read(root/'summary.json')['observed']
        comparison=[{'Method':'KRV100 epoch 3 (validation best)',**first},
                    {'Method':label,**last},
                    {'Method':'Later minus epoch 3 (success: pp; time: s)',
                     **{k:last[k]-first[k] for k in first}}]
        with (root/'checkpoint_comparison.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(comparison)
        lines=['# Checkpoint comparison on the same 360-trial task suite','',
               '| '+' | '.join(fields)+' |','|'+'|'.join(['---']+['---:']*(len(fields)-1))+'|']
        lines.extend('| '+r['Method']+' | '+' | '.join(f'{r[k]:.2f}' for k in fields[1:])+' |' for r in comparison)
        lines+=['','Same task configurations, repetition policy, simulator and hardware; inference latency can affect trajectories.',
                'The later checkpoint is partway through epoch 9. No epoch-10 checkpoint was trained.','']
        (root/'checkpoint_comparison.md').write_text('\n'.join(lines))
    suite.report=report
    try:suite.run(root)
    except BaseException as error:
        suite.state(root,'failed',error=str(error));raise


if __name__=='__main__':main()
