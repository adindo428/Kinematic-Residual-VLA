"""Render constant-gate resume, saved prefixes and both active evaluator lanes."""
from pathlib import Path

def render():
    from table1_display_state import FULL,DOMV,BASE,DIMS,read,provisional,write_if_changed,fresh
    control=FULL/'constant_resume';plan=read(control/'request.json')
    if not plan.get('enabled'):return
    state=read(control/'status.json');selected=read(control/'task_selection.json')
    mode='two parallel lanes' if state.get('parallel') else 'serial / qualification'
    keys=[*DIMS,'Success','Time']
    lines=['# Original full DOM suite comparison','',
        '**Queue: KRV100_Full epoch 3 (constant gate) → DOMV epoch 3 → latest KRV gated (4 epochs + partial epoch 5).**','',
        'First pass: **100 trials per column, 900 per checkpoint**. The remaining ten repeats per configuration are deferred. '
        'Constant-gate epoch 3 resumes its completed CR (100 trials) and DA prefix (14 trials); gated results remain separate.','',
        f'Runtime mode: **{mode}**. Parallel runs require p50 and p95 latency ≤ 1/0.95 of the isolated reference, '
        '≥95% action-coverage retention, and a measured wall-time benefit. Rolling breaches trigger serial fallback; '
        'affected incomplete attempts are retained and excluded before retry. Qualification trials are unscored.','',
        '| Method | Status | '+' | '.join(keys)+' |','|---|---|'+'---:|'*len(keys)]
    for label,status,v in [('DynamicVLA paper','Reference',read(BASE/'protocol.json')['paper_reference']),
                           ('Released DynamicVLA (local original suite)','Complete',read(BASE/'summary.json')['observed'])]:
        lines.append('| '+label+' | '+status+' | '+' | '.join(f'{v[k]:.2f}' for k in keys)+' |')
    models=plan['models']
    for m in models:
        root=Path(m['root']);v=provisional(root);status='Pending';live=[]
        if read(root/'first_pass_summary.json').get('first_pass_complete'):status='First pass complete (900/1800)'
        elif state.get('model')==m['key']:
            stage=state.get('stage','queued')
            if stage.startswith('qualifying'):status=stage.replace('_',' ').capitalize()
            elif stage=='failed':status='Failed; inspect logs'
            elif stage=='paused':status='Paused'
            elif not fresh(state):status='Status stale; check controller'
            else:
                live=[r for r in state.get('lanes',[]) if r.get('model')==m['key'] and r.get('scored')]
                status='; '.join(f"Running {r['dimension']} (lane {r['lane']})" for r in live) or stage.replace('_',' ')
        cells=[v.get(k,'pending') for k in keys]
        for d in DIMS:
            if d in v:continue
            prefix=14 if m['key']=='krv_best_constant' and d=='DA' else 0
            done=sum(t['target'] for t in plan['tasks'][m['key']] if t['dimension']==d and m['key']+'/'+t['key'] in selected)
            active=sum(r['records'] for r in live if r['dimension']==d)
            if any(r['dimension']==d for r in live):cells[DIMS.index(d)]=f'running ({prefix+done+active}/100)'
            elif prefix+done:cells[DIMS.index(d)]=f'{prefix+done}/100 saved; pending'
        lines.append('| '+m['label']+' | '+status+' | '+' | '.join(cells)+' |')
    gated=FULL/'variants/krv100_epoch003_gated';v=provisional(gated)
    lines.append('| KRV100_Full epoch 3 (causal gate) | Paused; results retained | '+' | '.join(v.get(k,'pending') for k in keys)+' |')
    lines+=['','Provisional scores show their denominators; only n=200 per column completes the full 1,800-trial suite. '
            'Constant gate means analytic KR remains enabled throughout the episode. The 0.20-second timing intervention is disabled. '
            'All rows retain the original 90 physical configurations and success rules. '
            'Completed saved trials are preserved, but restarted blocks do not claim bitwise-continuous prompt/noise replay. '
            'KRV and DOMV have different training histories and use privileged velocity. Low-clutter evaluation remains paused.','']
    text='\n'.join(lines)
    for root in [FULL,DOMV,*[Path(m['root']) for m in models]]:write_if_changed(root/'table1_comparison.md',text)

if __name__=='__main__':render()
