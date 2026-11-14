"""One live Markdown dashboard for baseline, causal gate, and OOD runs."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parents[1]/'scratch/dom45/evaluations'
TARGET = BASE/'closed-loop-pilot-20261112/live_results.md'
SUITES = [
    ('Baseline / IID-style', 'closed-loop-pilot-20261112'),
    ('Causal gate / IID-style', 'closed-loop-causal-gate-20261112'),
    ('OOD initial speed', 'closed-loop-speed-ood-20261112'),
    ('Causal gate / OOD', 'closed-loop-ood-causal-gate-20261112'),
    ('Intermediate speed', 'closed-loop-speed-mid-20261112'),
]
MODELS = [('dom_pretrained', 'Pretrained DOM'), ('domv_best', 'DOMV full epoch 3'),
          ('krv_best_constant', 'KRV full epoch 3 — constant'),
          ('krv_best_causal', 'KRV full epoch 3 — causal gate')]

def read(path): return json.loads(path.read_text())

def link(path): return Path(os.path.relpath(path, TARGET.parent)).as_posix()

def render():
    suites = []
    for label, name in SUITES:
        root = BASE/name; plan = read(root/'plan.json'); state = read(root/'status.json')
        rows = []; by_model = defaultdict(list)
        for job in plan['jobs']:
            folder = Path(job['folder']); receipt = folder/'resident_client_completion.json'
            if not receipt.exists() or not read(receipt).get('complete'): continue
            records = [json.loads(s) for s in (folder/'trial_results.jsonl').read_text().splitlines()]
            assert len(records) == job['repeats'] == 1 and records[0]['status'] in (0, 1)
            row = {**records[0], **job}; rows.append(row); by_model[job['model']].append(row)
        suites.append(dict(label=label, root=root, plan=plan, state=state, rows=rows, by_model=by_model))
    count = sum(len(s['rows']) for s in suites); total = sum(len(s['plan']['jobs']) for s in suites)
    extra = BASE/'closed-loop-velocity-innovation-20261112'
    extra_count = 0
    if (extra/'plan.json').exists():
        extra_plan = read(extra/'plan.json')
        extra_count = sum((Path(j['folder'])/'resident_client_completion.json').exists() for j in extra_plan['jobs'])
        count += extra_count
        total += len(extra_plan['jobs'])
    now = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S %Z')
    lines = ['# Closed-loop evaluation — all live results', '',
        f'Updated: **{now}** · Refresh: **10 seconds** · Completed: **{count}/{total}**', '',
        'Queue: baseline (75 episodes) → causal gate (25 episodes) → OOD speed (75 episodes) → OOD causal gate (25 episodes) → intermediate speed, all four variants (100 episodes).', '',
        '## Placement success', '',
        'Cells show **successes/completed (success rate)**; each scheduled model has 25 episodes.', '',
        '| Model | Baseline / IID-style · 0.10 m/s | Intermediate · 0.50 m/s | OOD initial speed · 1.00 m/s |',
        '|---|---:|---:|---:|']
    for key, label in MODELS:
        cells = []
        for s in [suites[1] if key == 'krv_best_causal' else suites[0], suites[4], suites[3] if key == 'krv_best_causal' else suites[2]]:
            if key not in {m['key'] for m in s['plan']['models']}: cells.append('Not scheduled'); continue
            rr = s['by_model'][key]; n = len(rr); successes = sum(r['success'] for r in rr)
            cells.append(f'{successes}/{n} ({100*successes/n:.1f}%)' if n else 'Pending · 0/25 completed')
        lines.append(f'| {label} | '+' | '.join(cells)+' |')
    lines += ['', 'Baseline is called “IID-style” for convenience; its new layouts reuse DOM assets and are not certified IID to training. '
        'OOD refers to initial speed above the saved 0.75 m/s collection limit, supported by the 4,096-episode sample; '
        'it is not a bound on all contact/fall velocities. Causal gating is inference-only on the same KRV weights.', '',
        'Partial counts may cover different seeds. Pending episodes are not failures; rates use completed episodes only.', '',
        '## Queue and progress', '', '| Test | Completed | State | Status last updated (UTC) |', '|---|---:|---|---|']
    from render_velocity_innovation import section
    addition = section(BASE)
    if addition:
        position = lines.index('## Queue and progress')
        lines[position:position] = addition
    for s in suites:
        lines.append(f"| [{s['label']}]({link(s['root']/'status.json')}) | {len(s['rows'])}/{len(s['plan']['jobs'])} | {s['state']['stage']} | {s['state'].get('updated_utc', 'unknown')} |")
    if (extra/'status.json').exists():
        st = read(extra/'status.json')
        lines.append(f"| [Training-free velocity innovation]({link(extra/'status.json')}) | {extra_count}/225 | {st['stage']} | {st['updated_utc']} |")
    for s in suites:
        models = s['plan']['models']
        lines += ['', f"## {s['label']} — by layout", '',
            '| Layout | '+' | '.join(m['key'] for m in models)+' |', '|---|'+'---:|'*len(models)]
        for layout in range(1, 6):
            cells = []
            for m in models:
                rr = [r for r in s['by_model'][m['key']] if r['layout'] == layout]
                cells.append(f"{sum(r['success'] for r in rr)}/{len(rr)}")
            lines.append(f'| {layout} | '+' | '.join(cells)+' |')
        lines += ['', f"### {s['label']} — completed episodes and videos", '',
            '| Layout | Seed | Model | Placement | Video |', '|---|---:|---|---|---|']
        for row in s['rows']:
            videos = sorted((Path(row['folder'])/'videos').rglob('*.mp4'))
            links = ' · '.join(f'[video {i+1}]({link(v)})' for i, v in enumerate(videos)) or 'Encoding / unavailable'
            lines.append(f"| {row['layout']} | {row['seed']} | {row['model']} | {'Success' if row['success'] else 'Failure'} | {links} |")
        if not s['rows']: lines.append('| — | — | — | Pending | — |')
    lines += ['', 'All runs use matched layouts, simulator seeds, and prompts. Policy sampling noise is not paired. '
        'Completion verification remains separate from these live counts.', '']
    temporary = TARGET.with_suffix('.combined.tmp'); temporary.write_text('\n'.join(lines)); temporary.replace(TARGET)
    extra_state = BASE/'closed-loop-velocity-innovation-20261112/status.json'
    extra_done = not extra_state.exists() or read(extra_state).get('stage') in ('complete', 'failed')
    return all(s['state']['stage'] == 'complete' for s in suites) and extra_done

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--once', action='store_true'); args = parser.parse_args()
    while True:
        done = render()
        if args.once or done: break
        time.sleep(10)
