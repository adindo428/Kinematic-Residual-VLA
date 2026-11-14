#!/usr/bin/env python3
"""Publish current validation and verified mini-suite results without resuming W&B."""
import argparse
import csv
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'evaluation/official_dom/scripts'))
from report import verify_folder


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def write(path,data):
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def snapshot(stage,results):
    run=stage/'family/runs/krv100'
    latest=read(run/'latest.json');validation=[]
    for path in sorted(run.glob('validation-epoch-*.json')):
        epoch=int(re.search(r'epoch-(\d+)',path.name).group(1))
        for metric,value in sorted(read(path).items()):
            validation.append({'epoch':epoch,'metric':metric,'value':value})
    status=read(results/'suite_status.json');request=read(results/'request.json')
    verified={}
    for i,dim in enumerate(request['dimension_order']):
        folder=results/'full'/f'{i:02d}_{dim}'
        if (folder/'verification.json').is_file():
            verified[dim]=verify_folder(folder,results,results/'test_lists'/f'{i:02d}_{dim}.txt',20)
    summary=read(results/'summary.json')
    if summary and (not summary.get('complete') or summary['verified_trials']!=360 or len(verified)!=9):
        raise ValueError('Complete task-subset results require 360 verified trials')
    # Timestamp-free signature: upload each new saved checkpoint, validation or
    # completed condition once, rather than every poll/status heartbeat.
    body={'latest_checkpoint':latest,'validation':validation,'evaluation_state':status.get('state'),
          'verified_conditions':verified,'evaluation_summary':summary,
          'selected_evaluation_epoch':request['target_epoch'],
          'selected_checkpoint_label':request.get('checkpoint_label',f"epoch {request['target_epoch']}"),
          'expected_trials':360,'evaluation_scope':request['report_scope'],
          'prior_exposure':request['prior_exposure'],'wandb_id':'2qh0ijb1'}
    signature=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
    return body,signature


def publish(stage,results,out,data,signature,report_prefix='reports/20k-continuation'):
    import wandb
    root=out/report_prefix;root.mkdir(parents=True,exist_ok=True)
    write(root/'progress.json',{'uploaded_at_utc':datetime.now(timezone.utc).isoformat(),**data})
    with (root/'validation.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['epoch','metric','value']);writer.writeheader();writer.writerows(data['validation'])
    mses={row['epoch']:row['value'] for row in data['validation'] if row['metric']=='h20/action_mse'}
    lines=['# KRV100: 20k continuation progress','',
           'Saved training progress: '+json.dumps(data['latest_checkpoint']['progress']),
           '', '| Completed epoch | Validation action MSE |','|---:|---:|']
    lines.extend(f'| {epoch} | {value:.8f} |' for epoch,value in sorted(mses.items()))
    lines+=['','Evaluation status: **'+str(data['evaluation_state'])+'**.',data['evaluation_scope'],
            '', '| Condition | Verified trials | Success (%) | Mean all-trial time (s) |',
            '|---|---:|---:|---:|']
    for dim in ('CR','DA','LS','VU','SR','MP','VG','MG','DR'):
        record=data['verified_conditions'].get(dim)
        lines.append(f"| {dim} | {record['trials']} | {record['success_percent']:.2f} | {record['mean_task_time_s']:.2f} |"
                     if record else f'| {dim} | 0 | pending | pending |')
    lines+=['',data['prior_exposure'],'','Pending cells are not zero. Action MSE is distinct from simulator manipulation success.','']
    (root/'progress.md').write_text('\n'.join(lines))
    for name in ('request.json','training_task_coverage.json','README.md','protocol.json',
                 'summary.json','table1_task_subset.csv','table1_task_subset.md',
                 'checkpoint_comparison.csv','checkpoint_comparison.md'):
        path=results/name
        if path.is_file():shutil.copy2(path,root/name)
    api=wandb.Api(timeout=90)
    run=api.run('aparame-clemson-university/Kinematic-Residual-VLA/2qh0ijb1')
    if run.id!='2qh0ijb1':raise ValueError('Unexpected W&B run')
    uploaded=[]
    for path in sorted(root.iterdir()):
        if path.is_file():
            record=run.upload_file(str(path),root=str(out));uploaded.append(record.name)
    # Files API does not start/finish the run or append competing history steps.
    # Update summary only after the trainer has exited, when no writer competes.
    active=subprocess.run(['systemctl','--user','is-active','krvla-krv100-finish-epoch10-20261024.service'],
                          capture_output=True,text=True,check=False).stdout.strip()
    if active not in ('active','activating','deactivating'):
        prefix=f"TaskSubset/epoch{data['selected_evaluation_epoch']:03d}/"
        metrics={prefix+'evaluation_state':data['evaluation_state'],
                 prefix+'checkpoint_label':data['selected_checkpoint_label'],
                 prefix+'verified_trials':sum(r['trials'] for r in data['verified_conditions'].values()),
                 prefix+'complete':bool(data['evaluation_summary'])}
        for dim,record in data['verified_conditions'].items():
            metrics.update({prefix+f'{dim}/success_percent':record['success_percent'],
                            prefix+f'{dim}/mean_time_s':record['mean_task_time_s']})
        if data['evaluation_summary']:
            metrics.update({prefix+k:v for k,v in data['evaluation_summary']['observed'].items()})
        run.summary.update(metrics)
    # Re-query remote file metadata as a receipt rather than assuming success.
    remote={f.name:f for f in run.files(names=uploaded)}
    if set(remote)!=set(uploaded) or any(remote[name].size<=0 for name in uploaded):
        raise RuntimeError('W&B did not confirm every report file')
    receipt={'phase':'uploaded','updated_at_utc':datetime.now(timezone.utc).isoformat(),
        'signature':signature,'wandb_run':run.url,'uploaded_files':uploaded,
        'latest_completed_validation_epoch':max(mses) if mses else None,
        'evaluation_state':data['evaluation_state'],'evaluation_complete':bool(data['evaluation_summary'])}
    write(out/'upload_status.json',receipt)
    print(json.dumps(receipt),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--results',type=Path)
    parser.add_argument('--state-dir',type=Path)
    parser.add_argument('--report-prefix',default='reports/20k-continuation')
    args=parser.parse_args()
    work=REPO/'scratch/dom45';stage=work/'training/krv100-20k-from-epoch003-20261021'
    override=read(stage/'evaluation_selection_override.json')
    results=(Path(override['active_request']).parent if override.get('active_request')
             else work/'results/krv100-epoch010-task-table1-20261024')
    if args.results:results=args.results.resolve()
    out=args.state_dir.resolve() if args.state_dir else stage/'wandb-progress'
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                data,signature=snapshot(stage,results)
                if read(out/'upload_status.json').get('signature')!=signature:
                    publish(stage,results,out,data,signature,args.report_prefix)
                if not args.watch or data['evaluation_summary']:return
            except Exception as error:
                write(out/'last_error.json',{'updated_at_utc':datetime.now(timezone.utc).isoformat(),'error':str(error)})
                if not args.watch:raise
                print('Upload retry pending: '+str(error),flush=True)
            time.sleep(120)


if __name__=='__main__':main()
