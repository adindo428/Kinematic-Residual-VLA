"""Read live evaluation status for tables; never change scoring or scheduling."""
from datetime import datetime, timezone
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
LOW=REPO/'scratch/dom45/evaluations/low-clutter-full-table1-20261102'
FULL=LOW.parent/'krv-full-original-dom-20261106'
DOMV=LOW.parent/'domv-full-original-dom-20261106'
BASE=REPO/'scratch/dom45/results/dom-paper45-a6000-20261010-02'
DIMS=['CR','DA','LS','VU','SR','MP','VG','MG','DR']


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,json.JSONDecodeError):return {}


def fresh(state):
    try:return (datetime.now(timezone.utc)-datetime.fromisoformat(state['updated_utc'])).total_seconds()<180
    except (KeyError,ValueError,TypeError):return False


def low_rows():
    s=read(LOW/'status.json')
    return s.get('active_evaluations',[]) if fresh(s) and s.get('state')=='evaluating' else []


def phase_text(row):
    d=row.get('dimension')
    if d in DIMS:
        text=f"Running {d} ({row.get('completed_records_in_phase',0)}/{row.get('phase_trial_target',200)})"
        return text+(f"; pass {row['pass_index']}/2" if row.get('pass_index') else '')
    return 'Running pilot / timing check'


def status(name,suite='low_clutter'):
    root=(LOW/'results'/name) if suite=='low_clutter' else FULL if name=='krv100_full' else DOMV
    if read(root/'summary.json').get('complete'):return 'Complete'
    if read(LOW/'status.json').get('state')=='paused_by_user':return 'Paused by user'
    live=low_rows()
    if suite=='low_clutter':
        for r in live:
            if r.get('model')==name:
                text=phase_text(r)
                if name=='krv100_full' and read(LOW/'status.json').get('planned_maintenance'):
                    text+='; remainder queued after DOMV_Full'
                return text
        if name=='krv100_full':return 'Queued after DOMV_Full original suite'
        if name=='domv_full':return 'Queued after KRV100_Full low-clutter'
        return 'Paused'
    if not any(r.get('model')==name for r in live):
        s=read(root/'status.json')
        if fresh(s) and s.get('state') in ['evaluating','probing']:
            return phase_text(s)
    return 'Queued after current phase' if name=='krv100_full' else 'Queued after KRV_Full original suite'


def observed(root):
    values={}
    for i,d in enumerate(DIMS):
        v=read(root/'full'/f'{i:02d}_{d}'/'verification.json')
        if v.get('verified') and v.get('trials')==200:values[d]=v['success_percent']
    summary=read(root/'summary.json')
    if summary.get('complete') and summary.get('verified_trials')==1800:values.update(summary['observed'])
    return values


def provisional(root):
    values={};total=successes=0;time_sum=0;counts=[]
    for i,d in enumerate(DIMS):
        blocks=[read(root/'passes'/f'{p:02d}'/'full'/f'{i:02d}_{d}'/'verification.json') for p in [1,2]]
        blocks=[b for b in blocks if b.get('verified')]
        n=sum(b['trials'] for b in blocks);s=sum(b['successes'] for b in blocks)
        counts.append(n)
        if n and n<200:values[d]=f'{100*s/n:.2f} (n={n}, provisional)'
        elif n==200:values[d]=f'{100*s/n:.2f}'
        if n:
            total+=n;successes+=s;time_sum+=sum(b['mean_task_time_s']*b['trials'] for b in blocks)
    # Overall provisional averages only after every condition has evidence.
    if len(values)==9 and len(set(counts))==1:
        suffix=f' (n={total}, provisional)' if counts[0]<200 else ''
        values['Success']=f'{100*successes/total:.2f}'+suffix
        values['Time']=f'{time_sum/total:.2f}'+(' (provisional)' if counts[0]<200 else '')
    return values


def write_if_changed(path,text):
    if path.exists() and path.read_text()==text:return
    # Distinct temporary files allow independent read-only publishers to refresh.
    import os
    tmp=path.with_name(path.name+f'.status-{os.getpid()}.tmp')
    tmp.write_text(text);tmp.replace(path)


def original_tables():
    if read(FULL/'resident_schedule/request.json').get('enabled'):
        from render_resident_table1 import render
        return render()
    if read(FULL/'constant_resume/request.json').get('enabled'):
        from render_constant_table1_resume import render
        return render()
    if read(FULL/'gated_queue/request.json').get('enabled'):
        from render_gated_original_table1 import render
        return render()
    keys=[*DIMS,'Success','Time']
    rows=[('DynamicVLA paper','Reference',read(BASE/'protocol.json')['paper_reference']),
          ('Released DynamicVLA (local original suite)','Complete',read(BASE/'summary.json')['observed']),
          ('KRV100_Full (validation-best epoch 3)',status('krv100_full','original_dom'),observed(FULL)),
          ('DOMV_Full (validation-best, epoch 3)',status('domv_full','original_dom'),observed(DOMV))]
    text='# Original full DOM suite comparison\n\n90 original configurations × 20 repeats = **1,800 scored trials per checkpoint**.\n\n'
    text+='Queue: **KRV_Full original DOM → DOMV_Full original DOM → KRV100_Full low-clutter → DOMV_Full low-clutter**.'
    if read(LOW/'status.json').get('state')=='paused_by_user':
        text+=' **Table 1 evaluations and follow-ups are paused by user request; this order is inactive.**'
    if read(LOW/'status.json').get('planned_maintenance') and low_rows():
        text+=' The active low-clutter CR phase finishes before this order takes over.'
    text+='\n\n'
    schedule=read(FULL/'two_pass_plan.json')
    if schedule:
        counts=schedule['repeats_per_pass']
        text+=f"Two-pass schedule: **{counts[0]} repeats per configuration across all nine columns first, then {counts[1]} remaining repeats**. Each column has ten configurations. Provisional cells show their trial count; only n=200 is final.\n\n"
    text+='| Method | Status | '+' | '.join(keys)+' |\n|---|---|'+'---:|'*len(keys)+'\n'
    for row_index,(label,state,values) in enumerate(rows):
        partial=provisional(FULL if row_index==2 else DOMV) if row_index>=2 else {}
        cells=[f'{values[k]:.2f}' if k in values else partial.get(k,'pending') for k in keys]
        if state.startswith('Running '):
            for i,d in enumerate(DIMS):
                if state.startswith('Running '+d+' ') and d not in values:
                    cells[i]=(partial[d]+'; running') if d in partial else 'running'
        text+='| '+label+' | '+state+' | '+' | '.join(cells)+' |\n'
    text+='\nKRV_Full and DOMV_Full use privileged simulator pose/velocity and have different training histories. The released-checkpoint row is the archived local original-suite result; paper values are a separate reference. The low-clutter results remain in their own table.\n'
    text+='\nVideo sampling for newly started phases: repeat 1 plus the first successful repeat per configuration, at most 180 videos per suite. This affects saved qualitative examples only; all trials remain scored.\n'
    for root in [FULL,DOMV]:
        if root.exists():write_if_changed(root/'table1_comparison.md',text)
