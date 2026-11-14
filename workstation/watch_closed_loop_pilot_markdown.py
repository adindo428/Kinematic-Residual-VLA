"""Publish completed pilot episodes to Markdown without touching evaluation."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]/'scratch/dom45/evaluations/closed-loop-pilot-20261112'


def read(path):
    return json.loads(path.read_text())


def render():
    plan = read(ROOT/'plan.json')
    state = read(ROOT/'status.json')
    models = plan['models']
    rows = defaultdict(list)
    completed = []
    for job in plan['jobs']:
        folder = Path(job['folder'])
        if not (folder/'resident_client_completion.json').exists():
            continue
        receipt = read(folder/'resident_client_completion.json')
        if not receipt.get('complete'):
            continue
        records = [json.loads(line) for line in (folder/'trial_results.jsonl').read_text().splitlines()]
        assert len(records) == 1 and records[0]['status'] in (0, 1)
        row = {**records[0], **job}
        rows[job['model']].append(row)
        completed.append(row)
    now = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S %Z')
    lines = ['# Closed-loop pilot — live results', '',
        f"Updated: **{now}** · Refresh: **10 seconds** · Completed: **{len(completed)}/{len(plan['jobs'])}**", '',
        f"Evaluation status: **{state['stage']}**. Status timestamp: {state.get('updated_utc', 'unknown')}.", '',
        f"Five layouts × five simulator seeds per model. Initial object speed: **{plan['protocol']['initial_speed_mps']:.2f} m/s**. "
        'Native closed-loop streaming; no added disturbances or velocity sweep.', '',
        '## Placement success', '',
        '| Model | Successes / completed | Success rate | Progress |',
        '|---|---:|---:|---:|']
    for model in models:
        rr = rows[model['key']]; n = len(rr); s = sum(r['success'] for r in rr)
        rate = f'{100*s/n:.1f}%' if n else '—'
        lines.append(f"| {model['label']} | {s}/{n} | {rate} | {n}/25 |")
    lines += ['', 'Partial counts may cover different seeds. Pending episodes are not failures. '
        'These small-sample results do not establish overfitting. Layouts reuse existing DOM assets.', '',
        '## Successes / completed by layout', '',
        '| Layout | '+ ' | '.join(m['key'] for m in models)+' |',
        '|---|'+'---:|'*len(models)]
    for layout in range(1, 6):
        cells = []
        for model in models:
            rr = [r for r in rows[model['key']] if r['layout'] == layout]
            cells.append(f"{sum(r['success'] for r in rr)}/{len(rr)}")
        lines.append(f'| {layout} | '+' | '.join(cells)+' |')
    lines += ['', '## Completed episodes and videos', '',
        '| Layout | Seed | Model | Placement | Video |', '|---|---:|---|---|---|']
    for row in completed:
        videos = sorted((Path(row['folder'])/'videos').rglob('*.mp4'))
        links = ' · '.join(f'[video {i+1}]({v.relative_to(ROOT).as_posix()})' for i, v in enumerate(videos)) or 'Encoding / unavailable'
        lines.append(f"| {row['layout']} | {row['seed']} | {row['model']} | {'Success' if row['success'] else 'Failure'} | {links} |")
    lines += ['', '[Machine-readable status](status.json) · [Frozen protocol](protocol.json)', '']
    if plan['protocol'].get('ood_scope'):
        lines += ['OOD interpretation: '+plan['protocol']['ood_scope'], '']
    target = ROOT/'live_results.md'
    temporary = target.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines))
    temporary.replace(target)
    return state['stage'] in ('complete', 'failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.root.resolve()
    while True:
        done = render()
        if args.once or done:
            break
        time.sleep(10)
