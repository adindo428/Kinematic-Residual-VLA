"""Live view of the explicitly ordered original-DOM gating evaluation queue."""
from pathlib import Path


def render():
    from table1_display_state import FULL, DOMV, BASE, DIMS, read, fresh, observed, provisional, write_if_changed
    control=FULL/'gated_queue'
    request=read(control/'request.json')
    if not request.get('enabled'):return
    state=read(control/'status.json')
    models=request['models']
    keys=[*DIMS,'Success','Time']
    lines=['# Original full DOM suite comparison','',
           '**Active schedule: KRV epoch 3 gated → DOMV epoch 3 → latest KRV gated (4 epochs + partial epoch 5).**','',
           'Each checkpoint runs **100 trials per column now**: ten configurations × ten repeats, '
           'across all nine columns (**900 trials per checkpoint**). The remaining ten repeats per '
           'configuration are deferred. Results with n=100 are provisional; the full suite requires 1,800 trials.','',
           'KRV uses causal grasp gating only, with **no additional 0.20-second timing correction**. '
           'The diagnostic and newly submitted matched-training run remain separate.','',
           '| Method | Status | '+' | '.join(keys)+' |',
           '|---|---|'+'---:|'*len(keys)]
    references=[('DynamicVLA paper','Reference',read(BASE/'protocol.json')['paper_reference']),
                ('Released DynamicVLA (local original suite)','Complete',read(BASE/'summary.json')['observed'])]
    for label,status,values in references:
        lines.append('| '+label+' | '+status+' | '+' | '.join(f'{values[k]:.2f}' if k in values else 'pending' for k in keys)+' |')
    legacy=provisional(FULL)
    lines.append('| KRV100_Full epoch 3 (constant gate; previous run) | Paused; retained | '+' | '.join(legacy.get(k,'pending') for k in keys)+' |')
    for index,model in enumerate(models):
        root=Path(model['root']);values=observed(root);partial=provisional(root)
        status='Pending'
        if read(root/'first_pass_summary.json').get('first_pass_complete'):
            status='First pass complete (900/1800)'
        elif state.get('model')==model['key']:
            if state.get('state') in ('failed','paused_by_user'):
                status='Failed; inspect logs' if state['state']=='failed' else 'Paused by user'
            elif fresh(state) and state.get('state')=='probing':status='Running checkpoint check'
            elif fresh(state) and state.get('state')=='evaluating':
                d=state.get('dimension');n=state.get('completed_records_in_phase',0)
                status=f'Running {d} ({n}/100)' if d else f'Running unscored pilot ({n}/3)'
            else:status='Status stale; check controller'
        cells=[f'{values[k]:.2f}' if k in values else partial.get(k,'pending') for k in keys]
        if status.startswith('Running ') and state.get('dimension') in DIMS:
            i=DIMS.index(state['dimension'])
            if state['dimension'] not in values and state['dimension'] not in partial:
                cells[i]=f"running ({state.get('completed_records_in_phase',0)}/100)"
        lines.append('| '+model['label']+' | '+status+' | '+' | '.join(cells)+' |')
    lines+=['',
            'CR: closed-loop reactivity; DA: dynamic adaptation; LS: long-horizon sequencing; '
            'VU: visual understanding; SR: spatial reasoning; MP: motion perception; '
            'VG: visual generalization; MG: motion generalization; DR: disturbance robustness.','',
            'All new rows retain the 90 original physical configurations, clutter, motion settings and success rules. '
            'KRV and DOMV receive privileged simulator pose/velocity and have different training histories. '
            'The latest KRV checkpoint is u000304483-6ce4fb3d, not an exact end-of-epoch-4 checkpoint. '
            'Previous constant-gate outcomes are preserved separately and are not pooled with gated results.','',
            'Status refreshes every five seconds. Evaluation is serial on separate simulation/inference GPUs. '
            'Video selection retains repeat 1 and the first successful repeat per configuration. '
            'Low-clutter evaluation and other models remain paused.','']
    reuse=read(Path(models[0]['root'])/'cr_trial_reuse.json')
    if reuse:
        lines+=['Epoch-3 gated CR reuses **30 verified diagnostic trials**: avocado 2/10, cup 4/10, '
                'egg 2/10, including all failures. Only the other seven configurations (70 trials) '
                'are newly evaluated. Each configuration contributes exactly ten scored trials. '
                'The final CR verification retains both original source blocks and their client acknowledgements.','']
    text='\n'.join(lines)
    for root in [FULL,DOMV,*[Path(m['root']) for m in models]]:
        if root.exists():write_if_changed(root/'table1_comparison.md',text)


if __name__=='__main__':render()
