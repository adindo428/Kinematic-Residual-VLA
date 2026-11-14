"""Compare frozen DOM/KRV arms on completed matched scene-seed pairs."""
from collections import defaultdict
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]/'scratch/dom45/evaluations'
ROOT = BASE/'closed-loop-krv-velocity-innovation-20261112'
DOM = BASE/'closed-loop-velocity-innovation-20261112'


def read(path):
    return json.loads(path.read_text())


def results(root):
    cells = defaultdict(dict)
    if not (root/'plan.json').exists():
        return cells
    for job in read(root/'plan.json')['jobs']:
        folder = Path(job['folder'])
        if not (folder/'resident_client_completion.json').exists():
            continue
        if not read(folder/'resident_client_completion.json')['complete']:
            continue
        row, = [json.loads(line) for line in (folder/'trial_results.jsonl').read_text().splitlines()]
        cells[(job['initial_speed_mps'], job['correction_mode'])][job['scene']] = bool(row['success'])
    return cells


def render(shared=False):
    datasets = [('DOM', results(DOM)), ('KRV full epoch 3', results(ROOT))]
    state = read(ROOT/'status.json') if (ROOT/'status.json').exists() else {}
    lines = ['## KRV + velocity innovation versus DOM', '',
        f"KRV: **{state.get('completed', 0)}/150 completed · {state.get('stage', 'prepared')}**.", '',
        '| Policy | 0.10 m/s | 0.50 m/s | 1.00 m/s |', '|---|---:|---:|---:|']
    for name, cells in datasets:
        for mode in ('none', 'innovation'):
            values = []
            for speed in (.1, .5, 1.):
                rows = cells[(speed, mode)]
                n = len(rows); s = sum(rows.values())
                values.append(f'{s}/{n} ({100*s/n:.1f}%)' if n else 'Pending · 0/25')
            label = name + (' + velocity innovation' if mode == 'innovation' else ' — unchanged')
            lines.append('| '+label+' | '+' | '.join(values)+' |')
    lines += ['', 'Same physical configurations, prompts, and per-observation noise seeds; asynchronous timing can differ. '
        'KRV retains its learned velocity conditioning and native constant KR residual. Only the added execution correction changes between KRV arms. '
        'The correction uses oracle velocity, gain 1, a 5 cm cap, 5 mm/step smoothing, and fades after closing. '
        'Cross-policy differences also include checkpoint training history; this is not an isolated training-method comparison.', '',
        '| Policy | Speed (m/s) | Completed pairs | Innovation-only successes | Baseline-only successes | Paired change (pp) |',
        '|---|---:|---:|---:|---:|---:|']
    paired = []
    for name, cells in datasets:
        for speed in (.1, .5, 1.):
            a, b = cells[(speed, 'none')], cells[(speed, 'innovation')]
            common = a.keys() & b.keys()
            win = sum(b[k] and not a[k] for k in common)
            loss = sum(a[k] and not b[k] for k in common)
            delta = 100*(win-loss)/len(common) if common else None
            paired.append(dict(policy=name, speed_mps=speed, pairs=len(common),
                innovation_only_successes=win, baseline_only_successes=loss, paired_delta_pp=delta))
            lines.append(f'| {name} | {speed:.2f} | {len(common)}/25 | {win} | {loss} | '+
                (f'{delta:+.1f}' if delta is not None else 'Pending')+' |')
    lines += ['', 'Partial results are provisional; paired changes use only cases completed by both arms. '
        'Final interpretation requires completion verification. These are previously inspected development cases.', '',
        '[KRV protocol](../closed-loop-krv-velocity-innovation-20261112/README.md) · '
        '[KRV status](../closed-loop-krv-velocity-innovation-20261112/status.json)', '']
    ROOT.mkdir(parents=True, exist_ok=True)
    temp = ROOT/'comparison.tmp'
    temp.write_text('\n'.join(lines));temp.replace(ROOT/'comparison.md')
    temp = ROOT/'paired_comparison.tmp'
    temp.write_text(json.dumps(paired, indent=2)+'\n');temp.replace(ROOT/'paired_comparison.json')
    if shared:
        # Only called after acquiring the GPU queue lock and DOM verification.
        from watch_closed_loop_combined_markdown import render as render_base, TARGET
        render_base()
        content = TARGET.read_text()
        content = content.replace('## Queue and progress', '\n'.join(lines)+'\n## Queue and progress', 1)
        temp = TARGET.with_suffix('.krv.tmp');temp.write_text(content);temp.replace(TARGET)


if __name__ == '__main__':
    render()
