"""Live summary for the frozen training-free comparison."""
import json
from pathlib import Path


def section(base):
    root = base/'closed-loop-velocity-innovation-20261112'
    if not (root/'plan.json').exists():
        return []
    plan = json.loads((root/'plan.json').read_text())
    state = json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    cells = {}
    count = 0
    details = []
    for job in plan['jobs']:
        folder = Path(job['folder'])
        if not (folder/'resident_client_completion.json').exists():
            continue
        receipt = json.loads((folder/'resident_client_completion.json').read_text())
        if not receipt.get('complete'):
            continue
        row, = [json.loads(s) for s in (folder/'trial_results.jsonl').read_text().splitlines()]
        key = (job['initial_speed_mps'], job['correction_mode'])
        scores = cells.setdefault(key, [0, 0])
        scores[0] += int(row['success']); scores[1] += 1; count += 1
        videos = sorted((folder/'videos').rglob('*.mp4'))
        video = ' · '.join(f'[video]({v.relative_to(root).as_posix()})' for v in videos) or 'Encoding'
        summary = json.loads((folder/'velocity_correction_summary.json').read_text())
        details.append(f"| {job['initial_speed_mps']:.2f} | {job['layout']} | {job['seed']} | {job['correction_mode']} | "
            f"{'Success' if row['success'] else 'Failure'} | {100*summary['max_offset_m']:.3f} | {video} |")
    lines = ['', '## Training-free velocity innovation — released DOM', '',
        f"**{count}/225 completed · {state.get('stage', 'prepared')}**. Fresh unchanged-DOM, innovation and naive arms; same 25 cases at each speed.", '',
        '| Method | 0.10 m/s | 0.50 m/s | 1.00 m/s |', '|---|---:|---:|---:|']
    for mode, label in [('none', 'Unchanged DOM (fresh baseline)'), ('innovation', 'DOM + velocity innovation'), ('naive', 'DOM + naive velocity × age')]:
        values = []
        for speed in (.1, .5, 1.):
            success, total = cells.get((speed, mode), [0, 0])
            values.append(f'{success}/{total} ({100*success/total:.1f}%)' if total else 'Pending · 0/25')
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    lines += ['', 'Oracle robot-frame object velocity; correction uses no object or destination position. '
        'Gain 1, 5 cm norm cap, 5 mm/step smoothing, fade after first gripper closing command. '
        'Same scene seeds and per-observation policy noise; asynchronous chunk timing can still differ. '
        'No injected perturbations or tuning. Partial arms may cover different cases.', '',
        '[Protocol and provenance](../closed-loop-velocity-innovation-20261112/README.md) · '
        '[Status](../closed-loop-velocity-innovation-20261112/status.json) · '
        '[Episode results and videos](../closed-loop-velocity-innovation-20261112/live_results.md)', '']
    own = ['# Velocity innovation — individual trials', '', f'{count}/225 completed.', '',
        '| Speed (m/s) | Layout | Seed | Method | Placement | Max correction (cm) | Video |',
        '|---:|---:|---:|---|---|---:|---|', *details, '',
        'Maximum correction includes the bounded fade after gripper closing. Pending trials are not failures.', '']
    temporary = root/'live_results.tmp'
    temporary.write_text('\n'.join(own))
    temporary.replace(root/'live_results.md')
    return lines
