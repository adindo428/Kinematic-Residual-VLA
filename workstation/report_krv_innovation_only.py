"""Historical baselines versus new innovation arms, with noise caveat."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]/'scratch/dom45/evaluations'
ROOT = BASE/'closed-loop-krv-innovation-only-20261112'


def read(p):return json.loads(p.read_text())


def rows(root):
    result=[]
    for job in read(root/'plan.json')['jobs']:
        folder=Path(job['folder'])
        receipt=folder/'resident_client_completion.json'
        if not receipt.exists() or not read(receipt)['complete']:continue
        trial, = [json.loads(s) for s in (folder/'trial_results.jsonl').read_text().splitlines()]
        result.append({**trial,**job})
    return result


def render():
    state=read(ROOT/'status.json')
    groups={k:[] for k in ('DOM baseline (existing)','DOM + innovation (completed subset)',
        'KRV baseline (existing)','KRV + innovation')}
    for speed,name in ((.1,'closed-loop-pilot-20261112'),(.5,'closed-loop-speed-mid-20261112'),(1.,'closed-loop-speed-ood-20261112')):
        for r in rows(BASE/name):
            if r['model'] in ('dom_pretrained','krv_best_constant'):
                key='DOM baseline (existing)' if r['model']=='dom_pretrained' else 'KRV baseline (existing)'
                groups[key].append(dict(r,initial_speed_mps=speed))
    groups['DOM + innovation (completed subset)']=[r for r in rows(BASE/'closed-loop-velocity-innovation-20261112') if r['correction_mode']=='innovation']
    groups['KRV + innovation']=rows(ROOT)
    lines=['## KRV + velocity innovation — existing baselines', '',
        f"**{len(groups['KRV + innovation'])}/75 completed · {state['stage']}**. No additional baseline trials.", '',
        '| Policy | 0.10 m/s | 0.50 m/s | 1.00 m/s |','|---|---:|---:|---:|']
    for label,rr in groups.items():
        cells=[]
        for speed in (.1,.5,1.):
            selected=[r for r in rr if r['initial_speed_mps']==speed];n=len(selected);s=sum(r['success'] for r in selected)
            cells.append(f'{s}/{n} ({100*s/n:.1f}%)' if n else 'No completed trials')
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines += ['', 'Existing baseline policy sampling noise is **not paired** with the new run. Scenes, physical seeds, '
        'prompts and checkpoint identity are matched. Partial DOM innovation results cover only completed cases; '
        'their raw rates should not be compared against a different case subset.', '',
        '| Speed | Matched completed KRV cases | Baseline success | Innovation success | Difference (pp) |',
        '|---:|---:|---:|---:|---:|']
    for speed in (.1,.5,1.):
        old={r['scene']:r for r in groups['KRV baseline (existing)'] if r['initial_speed_mps']==speed}
        new={r['scene']:r for r in groups['KRV + innovation'] if r['initial_speed_mps']==speed}
        keys=old.keys()&new.keys();n=len(keys);a=sum(old[k]['success'] for k in keys);b=sum(new[k]['success'] for k in keys)
        delta=f'{100*(b-a)/n:+.1f}' if n else 'Pending'
        lines.append(f'| {speed:.2f} | {n}/25 | {a}/{n} | {b}/{n} | {delta} |')
    lines += ['', '[Protocol](../closed-loop-krv-innovation-only-20261112/README.md) · '
        '[Status](../closed-loop-krv-innovation-only-20261112/status.json)', '']
    temp=ROOT/'comparison.tmp';temp.write_text('\n'.join(lines));temp.replace(ROOT/'comparison.md')
    # Avoid racing the active DOM dashboard writer during handoff.
    if state['stage'] not in ('prepared','waiting_for_gpu_handoff'):
        from watch_closed_loop_combined_markdown import render as old_render, TARGET
        old_render()
        content=TARGET.read_text().replace('## Queue and progress','\n'.join(lines)+'\n## Queue and progress',1)
        temp=TARGET.with_suffix('.only.tmp');temp.write_text(content);temp.replace(TARGET)
    # Keep the user's already-open comparison tab pointing at the new scope.
    previous=BASE/'closed-loop-krv-velocity-innovation-20261112/comparison.md'
    if read(BASE/'closed-loop-krv-velocity-innovation-20261112/status.json')['stage']=='superseded':
        previous.write_text('\n'.join(lines)+'\n')


if __name__=='__main__':render()
