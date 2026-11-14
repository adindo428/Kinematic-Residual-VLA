"""C/D/F checkpoint comparison with matched-case tests and provenance limits."""
import json
import math
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]/'scratch/dom45/evaluations'
ROOT=BASE/'closed-loop-domv-innovation-cdf-20261112'


def read(p):return json.loads(p.read_text())


def records(root, model=None):
    out=[]
    for j in read(root/'plan.json')['jobs']:
        if model and j['model']!=model:continue
        p=Path(j['folder']);receipt=p/'resident_client_completion.json'
        if not receipt.exists() or not read(receipt)['complete']:continue
        row,=[json.loads(s) for s in (p/'trial_results.jsonl').read_text().splitlines()]
        out.append({**row,**j})
    return out


def render(shared=False):
    data={'A':[],'C':records(ROOT),'D':[],'F':records(BASE/'closed-loop-krv-innovation-only-20261112')}
    for speed,name in ((.1,'closed-loop-pilot-20261112'),(.5,'closed-loop-speed-mid-20261112'),(1.,'closed-loop-speed-ood-20261112')):
        for letter,key in [('A','domv_best'),('D','krv_best_constant')]:
            data[letter] += [dict(r,initial_speed_mps=speed) for r in records(BASE/name,key)]
    state=read(ROOT/'status.json')
    lines=['## C vs D vs F — velocity-conditioned checkpoints', '',
        f"C: **{len(data['C'])}/75 completed · {state['stage']}**. D and F reuse completed results.", '',
        '| Arm | 0.10 m/s | 0.50 m/s | 1.00 m/s | Overall |','|---|---:|---:|---:|---:|']
    labels={'A':'A: DOMV, no execution correction','C':'C: DOMV + innovation',
        'D':'D: KRV constant, no execution correction','F':'F: KRV constant + innovation'}
    for k in ('A','C','D','F'):
        cells=[]
        for speed in (.1,.5,1.,None):
            rr=[r for r in data[k] if speed is None or r['initial_speed_mps']==speed]
            n=len(rr);s=sum(r['success'] for r in rr)
            cells.append(f'{s}/{n} ({100*s/n:.1f}%)' if n else 'Pending')
        lines.append('| '+labels[k]+' | '+' | '.join(cells)+' |')
    lines += ['', '**Existing F vs D result:** 17/75 versus 20/75, a -4 percentage-point observed change. '
        'The hypothesized five-point gain is not an observed result.', '',
        '| Comparison | Speed | Matched cases | F-only successes | Other-only successes | Difference (pp) | Exact discordance p (two-sided) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    tests=[]
    for other in ('C','D'):
        for speed in (.1,.5,1.,None):
            a={r['scene']:r for r in data[other] if speed is None or r['initial_speed_mps']==speed}
            b={r['scene']:r for r in data['F'] if speed is None or r['initial_speed_mps']==speed}
            keys=a.keys()&b.keys();n=len(keys)
            wins=sum(b[k]['success'] and not a[k]['success'] for k in keys)
            losses=sum(a[k]['success'] and not b[k]['success'] for k in keys)
            discord=wins+losses
            p=min(1.,2*sum(math.comb(discord,i) for i in range(min(wins,losses)+1))/2**discord) if discord else 1.
            delta=100*(wins-losses)/n if n else None
            tests.append(dict(comparison='F vs '+other,speed_mps=speed,matched_cases=n,
                f_only_successes=wins,other_only_successes=losses,delta_pp=delta,
                exact_mcnemar_p=p if n else None))
            lines.append(f"| F vs {other} | {speed if speed is not None else 'All'} | {n} | {wins} | {losses} | "+
                (f'{delta:+.1f}' if n else 'Pending')+' | '+(f'{p:.3f}' if n else 'Pending')+' |')
    lines += ['', 'Statistical tests are exploratory, unadjusted for multiple comparisons, and use matched scene-seed outcomes. '
        'They do not address training-seed uncertainty. Partial comparisons use only the common completed cases.', '',
        '**Attribution limits:** DOMV uses raw action targets with neural velocity inputs; KRV uses residual targets with neural velocity inputs. '
        'Both use the same physical velocity measurements and identical innovation settings. '
        'However DOMV trained on 197,198 episodes and KRV on 203,151; their velocity normalization and training histories differ. '
        'Thus F vs C is an existing-checkpoint comparison, not a fully controlled residual-training ablation. '
        'D uses historical unpaired policy noise; C and F use the same seed-by-observation schedule but asynchronous timing can differ.', '',
        'B and E (naive compensation on these velocity-conditioned checkpoints) are not run in this focused C/D/F test. '
        'Previously evaluated released DOM + innovation is not C because that checkpoint lacks neural velocity conditioning.', '',
        '[Protocol](../closed-loop-domv-innovation-cdf-20261112/README.md) · [Status](../closed-loop-domv-innovation-cdf-20261112/status.json)', '']
    text='\n'.join(lines)
    temp=ROOT/'comparison.tmp';temp.write_text(text);temp.replace(ROOT/'comparison.md')
    temp=ROOT/'paired_tests.tmp';temp.write_text(json.dumps(tests,indent=2)+'\n');temp.replace(ROOT/'paired_tests.json')
    if shared:
        from watch_closed_loop_combined_markdown import render as render_old,TARGET
        render_old()
        content=TARGET.read_text().replace('## Queue and progress',text+'\n## Queue and progress',1)
        tmp=TARGET.with_suffix('.cdf.tmp');tmp.write_text(content);tmp.replace(TARGET)
        target=BASE/'closed-loop-krv-velocity-innovation-20261112/comparison.md'
        tmp=target.with_suffix('.cdf.tmp');tmp.write_text(text);tmp.replace(target)


if __name__=='__main__':render()
