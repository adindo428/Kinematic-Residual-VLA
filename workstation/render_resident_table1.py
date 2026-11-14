"""Live scene-first progress from a verified ledger and a single active policy."""
from pathlib import Path
from collections import Counter

def render():
    from table1_display_state import FULL,DOMV,BASE,DIMS,read,provisional,write_if_changed
    from resident_table1_schedule import coverage
    root=FULL/'resident_schedule';request=read(root/'request.json')
    if not request.get('enabled'):return
    ledger=read(root/'ledger.json');state=read(root/'status.json');job=state.get('job') or {}
    extras=read(root/'additional_models.json')
    extras=extras if isinstance(extras,list) else []
    extra_states={m['key']:read(Path(m['control_root'])/'status.json') for m in extras}
    active_extra=next((extra_states[m['key']] for m in extras if extra_states[m['key']].get('stage') in
                       ('loading_resident_policies','evaluating','block_complete','starting','waiting_for_pilot')),None)
    shown_state=active_extra or state
    pass_index=state.get('pass_index',1)
    second=read(root/'pass02/ledger.json') if pass_index==2 else None
    target_trials=200 if second else 100
    keys=[*DIMS,'Success','Time']
    run=shown_state.get('run')
    profile=read(Path(run)/'execution/resident_profile.json') if run else {}
    capacity=profile.get('gpu_residency_capacity')
    placement=(f" Up to {capacity} policies retain GPU weights; other initialized policies stay in CPU RAM." if capacity else
               ' The startup benchmark selects how many policies retain GPU weights; CPU RAM holds the rest.')
    lines=['# Original full DOM suite comparison','',
           '**Scene-first schedule:** keep Isaac Sim alive, reset the same configuration for each checkpoint, '+
           ('and initialize the added policy once. Only one policy performs inference at a time.' if active_extra else
            'and initialize the three current policies once. Only one policy performs inference at a time.')+placement,'',
           ('**Active budget: 100 trials per column across all nine columns, 900 total, for latest constant gate only.** '
            'The previous three-checkpoint evaluation is paused; completed trials are retained. ' if active_extra else
            '**Budget: 200 trials per column, 1,800 per checkpoint.** Pass two runs repeats 11–20; '
            'all first-pass trials are retained. ' if second else
            '**Budget: 100 trials per column, 900 per checkpoint in pass one.** Pass two is queued for the remaining '
            '100 trials per column. ')+'The simulator resets before every trial '
           'and policy histories/queues are cleared between episodes. Checkpoint switches use completion fences '
           'rather than a fixed ten-second cooldown. Native in-episode inference timing remains unchanged. '
           'Prompts are cached across checkpoints; wording from previously scored trials is recovered from their logs.','',
           f"**Status: {shown_state.get('stage','preparing').replace('_',' ')}.**",'',
           '| Method | Status | '+' | '.join(keys)+' |','|---|---|'+'---:|'*len(keys)]
    for label,status,v in [('DynamicVLA paper','Reference',read(BASE/'protocol.json')['paper_reference']),
                           ('Released DynamicVLA (local original suite)','Complete',read(BASE/'summary.json')['observed'])]:
        lines.append('| '+label+' | '+status+' | '+' | '.join(f'{v[k]:.2f}' for k in keys)+' |')
    for m in ledger['models']:
        values=provisional(Path(m['root']));counts={d:len(coverage(ledger,m['key'],d)) for d in DIMS}
        active_ledger=second or ledger
        if second:
            counts={d:counts[d]+len(coverage(second,m['key'],d)) for d in DIMS}
        status='Pending'
        if all(n==target_trials for n in counts.values()):
            status='Complete (1800/1800)' if second else 'First pass complete; pass two queued'
        elif state.get('stage')=='paused':status='Paused'
        elif state.get('stage')=='waiting_for_pilot':status='Waiting for resident-scene qualification'
        elif job.get('model')==m['key']:
            status=f"Running {job['dimension']}: {Path(job['scene']).stem}"
            if state.get('stage') in ('failed','paused'):status=state['stage'].capitalize()
        cells=[]
        for k in keys:
            if k in values:
                value=values[k]
                if k in DIMS and job.get('model')==m['key'] and job.get('dimension')==k and state.get('stage')=='evaluating':
                    n=counts[k]+(state.get('block_records',0) if job['id'] not in active_ledger['accepted_jobs'] else 0)
                    value+=f'; running ({n}/{target_trials})'
                cells.append(value);continue
            if k in DIMS:
                n=counts[k]
                if job.get('model')==m['key'] and job.get('dimension')==k and job['id'] not in active_ledger['accepted_jobs']:
                    n+=state.get('block_records',0);cells.append(f'running ({n}/{target_trials})')
                else:cells.append(f'{n}/{target_trials} saved; pending' if n else 'pending')
            else:cells.append('pending')
        name=m['label']
        if m['key']=='krv_latest_gated':name=name.replace(', gated)',', causal gate)')
        lines.append('| '+name+' | '+status+' | '+' | '.join(cells)+' |')
    for m in extras:
        control=Path(m['control_root']);progress=extra_states[m['key']];current=progress.get('job') or {}
        phase=progress.get('pass_index',1);own=read(control/'ledger.json');tail=read(control/'pass02/ledger.json')
        deferred=read(control/'request.json').get('remaining_ten_repeats_deferred',False)
        extra_target=100 if deferred else 200
        values=provisional(Path(m['root']));counts={d:len(coverage(own,m['key'],d)) for d in DIMS}
        if tail:counts={d:counts[d]+len(coverage(tail,m['key'],d)) for d in DIMS}
        active_own=tail if phase==2 and tail else own
        label='Pending'
        if progress.get('stage')=='complete':label='Complete (1800/1800)'
        elif progress.get('stage')=='first_pass_complete_deferred':label='First pass complete; rest deferred (900/1800)'
        elif current:label=f"Running {current['dimension']} (pass {phase})"
        elif progress.get('stage') not in ('queued_after_current_pass',None):label=progress['stage'].replace('_',' ').capitalize()
        cells=[]
        for k in keys:
            value=values.get(k,'pending')
            if k in DIMS:
                n=counts[k]
                running=current.get('dimension')==k and progress.get('stage')=='evaluating'
                if running:
                    if current['id'] not in active_own['accepted_jobs']:n+=progress.get('block_records',0)
                    value=(value+'; ' if k in values else '')+f'running ({n}/{extra_target})'
                elif k not in values and n:value=f'{n}/{extra_target} saved; pending'
            cells.append(value)
        lines.append('| '+m['label']+' | '+label+' | '+' | '.join(cells)+' |')
    lines+=['','Constant gate retains KR throughout the episode. The latest constant- and causal-gate variants '
            'share the same weights: four completed epochs plus partial epoch five. '
            'Latest constant gate has a budget of 100 trials per column (900 total); its remaining 900 are deferred. '
            'The 0.20-second timing intervention is disabled. '
            'All cases retain the original physical configurations and success rules. KRV and DOMV use privileged '
            'velocity and have different training histories. Scores with n=100 are provisional. '
            'The unscored resident-policy qualification is recorded under `resident_pilot/verification.json`.','']
    metrics=shown_state.get('latency') or {}
    if metrics.get('samples'):
        lines+= [f"Current block: {metrics['samples']} recent chunks; p50 {metrics['p50_ms']:.1f} ms, "
                 f"p95 {metrics['p95_ms']:.1f} ms. These are live host-call timings, not controlled KR overhead.",'']
    text='\n'.join(lines)
    for target in [FULL,DOMV,*[Path(m['root']) for m in ledger['models']],*[Path(m['root']) for m in extras]]:write_if_changed(target/'table1_comparison.md',text)

if __name__=='__main__':render()
