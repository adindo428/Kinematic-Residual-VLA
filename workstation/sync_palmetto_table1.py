#!/usr/bin/env python3
"""Pull verified Palmetto metrics into the open custom Table 1 comparison."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1];WORK=REPO/'scratch/dom45'
STAGE=WORK/'palmetto-table1-20261026'
sys.path.insert(0,str(REPO/'evaluation/official_dom/scripts'))
from report import verify_records
DIMENSIONS=['CR','DA','LS','VU','SR','MP','VG','MG','DR']
LABELS={'pi05_baseline':'π0.5','pi05_kr':'π0.5 + KR','vlash_baseline':'VLASH','vlash_kr':'VLASH + KR',
        'smolvla_baseline':'SmolVLA','smolvla_kr':'SmolVLA + KR','domv100':'DOM + velocity'}


def read(path):return json.loads(path.read_text()) if path.is_file() else {}
def write(path,value):
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def render():
    target=WORK/'results/krv100-best-epoch003-task-table1-20261025/table1_task_subset.md'
    local=[('old KRV100',target.parent),('KRV100 · 8.09 epochs',WORK/'results/krv100-latest-epoch008p09-task-table1-20261025')]
    rows=[];progress=[]
    for label,root in local:
        summary=read(root/'summary.json')
        if not summary.get('complete') or summary.get('verified_trials')!=360:raise ValueError('Local comparison is incomplete')
        rows.append({'Method':label,**summary['observed']});progress.append((label,'360 / 360','local: A6000, Sim 4.5 / Lab 2.2.1'))
    complete=True
    catalog=read(STAGE/'remote-catalog.json')
    for name,label in LABELS.items():
        root=STAGE/'results'/name;protocol=read(root/'protocol.json');values={};count=0
        for i,dim in enumerate(DIMENSIONS):
            folder=root/'full'/f'{i:02d}_{dim}'
            if not (folder/'verification.json').exists():continue
            records=[json.loads(s) for s in (folder/'trial_results.jsonl').read_text().splitlines()]
            tests=[t for t in protocol['tests'] if t['dimension']==dim]
            verified=verify_records(records,[t['name'] for t in tests],20,{t['name']:t['horizon_seconds'] for t in tests})
            if verified['trials']!=40:raise ValueError('Incorrect remote condition denominator')
            values[dim]=verified['success_percent'];count+=40
        summary=read(root/'summary.json');status=read(root/'suite_status.json')
        if summary.get('complete') and summary.get('verified_trials')==count==360:
            values.update(summary['observed'])
        else:complete=False
        rows.append({'Method':label,**values})
        checkpoint=catalog.get('models',{}).get(name,{})
        checkpoint_progress=checkpoint.get('progress',{})
        detail=f"L40S; checkpoint epoch {checkpoint_progress.get('epoch','pending')}, updates {checkpoint_progress.get('updates','pending')}; {status.get('state','queued')}"
        progress.append((label,f'{count} / 360',detail))
    fields=['Method',*DIMENSIONS,'Success','Time']
    lines=['# Table 1: task-aligned VLA comparison','',
        'Same 18 custom scenarios × 20 repeats = 360 scored trials per checkpoint, plus nine separate pilot trials. This is not the official 1,800-trial DOM benchmark.','',
        '| '+' | '.join(fields)+' |','|'+'|'.join(['---']+['---:']*11)+'|']
    for row in rows:
        lines.append('| '+row['Method']+' | '+' | '.join(f'{row[k]:.2f}' if k in row else 'pending' for k in fields[1:])+' |')
    lines+=['','Success columns are percentages; Time is simulated seconds across successes and failures. Pending is not zero. Overall Success/Time appear only after all 360 trials verify.','',
        '## Completion and provenance','', '| Method | Verified main trials | Runtime / checkpoint |','|---|---:|---|']
    lines.extend('| '+' | '.join(row)+' |' for row in progress)
    lines+=['','Local rows use A6000 / Isaac Sim 4.5 / Isaac Lab 2.2.1. Remote rows request L40S only and use the installed Palmetto container (Lab 2.3.2; exact Sim version is recorded in each runtime file). Hardware/software differences can affect latency-sensitive outcomes.',
        '', 'Task families and object categories occur in the subset, but this is a mixed task-aligned/generalization/stress suite, not an entirely in-distribution evaluation. Pick variants do not test container selection. Training budgets/checkpoint progress differ across models; see the pinned catalog.',
        '', 'old KRV100 is the previously evaluated global epoch-3 checkpoint, renamed for display only. Both local KRV checkpoints have two preceding full-data fine-tuning epochs.','']
    text='\n'.join(lines);target.write_text(text)
    (STAGE/'table1_comparison.md').write_text(text)
    with (STAGE/'table1_comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    write(STAGE/'comparison.json',{'rows':rows,'remote_complete':complete,'columns':fields})
    return complete,hashlib.sha256(text.encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
    STAGE.mkdir(exist_ok=True);(STAGE/'results').mkdir(exist_ok=True)
    while True:
        try:
            if not args.render_only:
                transport='ssh -S /tmp/vlcm-palmetto/socket -o BatchMode=yes -o ConnectTimeout=15'
                subprocess.run(['rsync','-a','--timeout=60','--exclude=videos/','--exclude=actions/','--exclude=*.log',
                    '-e',transport,'aparame@slogin.palmetto.clemson.edu:/scratch/aparame/Dynamic_VLA/table1-multivla-20261026/results/',str(STAGE/'results')+'/'],check=True,timeout=90)
                result=subprocess.run(['ssh','-S','/tmp/vlcm-palmetto/socket','-o','BatchMode=yes','aparame@slogin.palmetto.clemson.edu',
                    'cat /scratch/aparame/Dynamic_VLA/table1-multivla-20261026/catalog.json'],check=True,capture_output=True,text=True,timeout=30)
                write(STAGE/'remote-catalog.json',json.loads(result.stdout))
            complete,signature=render();previous=read(STAGE/'sync_status.json')
            if signature!=previous.get('signature') and not args.render_only:
                import wandb
                run=wandb.Api(timeout=60).run('aparame-clemson-university/Kinematic-Residual-VLA/2qh0ijb1')
                upload=STAGE/'reports/multivla-table1';upload.mkdir(parents=True,exist_ok=True)
                import shutil
                for name in ('table1_comparison.md','table1_comparison.csv','comparison.json','remote-catalog.json'):
                    shutil.copy2(STAGE/name,upload/name);run.upload_file(str(upload/name),root=str(STAGE))
            write(STAGE/'sync_status.json',{'phase':'complete' if complete else 'following_remote_evaluations',
                'signature':signature,'updated_at_utc':datetime.now(timezone.utc).isoformat()})
            if args.render_only or not args.watch or complete:return
        except Exception as error:
            write(STAGE/'last_error.json',{'error':str(error),'updated_at_utc':datetime.now(timezone.utc).isoformat()})
            if not args.watch:raise
            print('Sync retry pending: '+str(error),flush=True)
        time.sleep(120)


if __name__=='__main__':main()
