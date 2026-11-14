#!/usr/bin/env python3
"""Render live table status without changing the frozen evaluation controller."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from table1_display_state import status as display_status, low_rows, write_if_changed, provisional

DIMS=['CR','DA','LS','VU','SR','MP','VG','MG','DR']
REPO=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text())


def render(root,active):
 selected=read(root/'catalog.json').get('active_resident_adapter_root')
 adapter=Path(selected)/'request.json' if selected else root/'resident_adapters-20261108/request.json'
 if adapter.exists() and read(adapter).get('enabled'):
  import run_resident_adapter_table1 as runner
  previous=runner.ROOT
  try:
   runner.ROOT=adapter.parent;runner.render_table()
  finally:runner.ROOT=previous
  return
 catalog=read(root/'catalog.json');state=read(root/'status.json')
 fields=['Method',*DIMS,'Success','Time'];lines=['| Method | Status | '+' | '.join(fields[1:])+' |','|---|---|'+'---:|'*(len(fields)-1)]
 running=active and state.get('state')=='evaluating'
 active_rows=low_rows() if running else []
 order=list(catalog['order'])
 if 'krv100_full' in order and (root/'krv_full_two_suites/requested_followup_order.json').exists() and 'domv_full' not in order:
  order.insert(order.index('krv100_full')+1,'domv_full')
 for name in order:
  folder=root/'results'/name;values={}
  for i,dim in enumerate(DIMS):
   path=folder/'full'/f'{i:02d}_{dim}'/'verification.json'
   if path.exists():
    verified=read(path)
    if verified.get('verified') and verified.get('matched_prompts_verified') and verified.get('trajectory_metrics_verified'):
     values[dim]=verified['success_percent']
  path=folder/'summary.json'
  if path.exists():
   summary=read(path)
   if summary.get('complete'):values.update(summary['observed'])
  partial=provisional(folder)
  cells={key:f'{values[key]:.2f}' if key in values else partial.get(key,'pending') for key in fields[1:]}
  label=catalog['models'].get(name,{}).get('label','DOMV_Full (low-clutter, validation-best epoch 3)')
  for row in active_rows:
   if row.get('model')!=name:continue
   dim=row.get('dimension')
   if row.get('phase')=='full' and dim in DIMS and dim not in values:cells[dim]=(partial[dim]+'; running') if dim in partial else 'running'
   elif row.get('phase')=='pilot':label+=' (running pilot)'
  lines.append('| '+label+' | '+display_status(name)+' | '+' | '.join(cells[key] for key in fields[1:])+' |')
 path=root/'table1_comparison.md';text=path.read_text();parts=text.splitlines()
 start=next(i for i,line in enumerate(parts) if line.startswith('| Method |'))
 end=start
 while end<len(parts) and parts[end].startswith('|'):end+=1
 replacement='\n'.join(parts[:start]+lines+parts[end:])+'\n'
 note=('Full-data follow-ups: KRV100_Full and DOMV_Full were trained only on their full retained training splits; '
       'both are outside the primary 20k cohort. Their low-clutter evaluation does not change their training identity. '
       'The mixed-data oldKRV100 checkpoint has been removed; its scores are archived and do not belong to KRV100_Full.\n')
 if not any(name in order for name in ['krv100_full','domv_full']):
  note='Full-data checkpoints are reported separately from this 20k-training comparison.\n'
 replacement='\n'.join(line for line in replacement.splitlines() if not line.startswith(('Full-data historical follow-ups:','Full-data follow-ups:')))+'\n'
 replacement+='\n'+note
 video_note=('Video sampling for newly started phases: save repeat 1 and the first successful repeat per configuration '
             '(at most two videos per configuration, 180 per suite). All trial scores and trajectories are retained. '
             'The phase already running when this change was requested keeps its existing recording policy.\n')
 if 'Video sampling for newly started phases:' not in replacement:replacement+='\n'+video_note
 targets=[path,REPO/'scratch/dom45/results/krv100-best-epoch003-task-table1-20261025/table1_task_subset.md']
 for target in targets:
  if target.exists() and target.read_text()==replacement:continue
  write_if_changed(target,replacement)


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--root',type=Path,required=True);parser.add_argument('--service',required=True)
 parser.add_argument('--once',action='store_true');args=parser.parse_args()
 while True:
  state=subprocess.run(['systemctl','--user','is-active',args.service],capture_output=True,text=True).stdout.strip()
  active=state in ['active','activating','reloading']
  render(args.root,active)
  if args.once or not active:return
  time.sleep(5)


if __name__=='__main__':main()
