#!/usr/bin/env python3
"""Render the diagnostic's live comparison without changing evaluation records."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'scratch/dom45/evaluations/krv100-full-causal-gate-cr-20261106'
TIMING_ROOT = REPO / 'scratch/dom45/evaluations/krv100-full-causal-gate-timing-cr-20261106'


def rows(path):
    result = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                break  # A live append may be incomplete.
    return result


def render(state=None):
    current = TIMING_ROOT if (TIMING_ROOT / 'status.json').exists() else ROOT
    status_path = current / 'status.json'
    if not status_path.exists():
        return
    status = json.loads(status_path.read_text())
    state = state or status['state']
    gated = rows(ROOT / 'full/trial_results.jsonl')
    timing = rows(TIMING_ROOT / 'full/trial_results.jsonl')
    timing_enabled = current == TIMING_ROOT
    active = timing if timing_enabled else gated
    original = rows(REPO / 'scratch/dom45/evaluations/krv-full-original-dom-20261106/passes/01/full/00_CR/trial_results.jsonl')
    released = rows(REPO / 'scratch/dom45/results/dom-paper45-a6000-20261010-02/full/00_CR/trial_results.jsonl')
    updated = status.get('updated_utc', '')
    if state in ('evaluating', 'running') and updated:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(updated)).total_seconds()
        if age > 180:
            state = 'status stale; check controller'
    lines = [
        '# CR causal-gate diagnostic comparison', '',
        f'**Status: {state} — {len(active)}/30 trials completed' + (' in the +0.20 s target-lead diagnostic.**' if timing_enabled else '.**'), '',
        f'Controller update (UTC): {updated}. Display refreshes every 5 seconds.', '',
        ('| CR scenario | Gating + 0.20 s lead | Progress | Gating only | Original constant gating | Released DynamicVLA |'
         if timing_enabled else '| CR scenario | New gating | Progress | Original constant gating | Released DynamicVLA |'),
        '|---|---:|---|---:|---:|---:|' if timing_enabled else '|---|---:|---|---:|---:|',
    ]
    active_found = False
    for key, label in [('avocado', 'Avocado'), ('cup', 'Cup'), ('egg', 'Egg')]:
        selected = [r for r in active if key in r.get('env_name', '')]
        progress = 'Complete' if len(selected) == 10 else 'Pending'
        if len(selected) < 10 and not active_found:
            active_found = True
            if state in ('evaluating', 'running'):
                progress = 'Running'
        def score(records):
            selected_rows = [r for r in records if key in r.get('env_name', '') and r.get('trial', 0) <= 10]
            n = len(selected_rows)
            successes = sum(bool(r.get('success')) for r in selected_rows)
            return f'{successes}/{n} ({100*successes/n:.1f}%)' if n else '—'
        if timing_enabled:
            lines.append(f'| {label} | {score(timing)} | {progress} | {score(gated)} | {score(original)} | {score(released)} |')
        else:
            lines.append(f'| {label} | {score(gated)} | {progress} | {score(original)} | {score(released)} |')
    lines += ['',
        'Each diagnostic scenario targets 10 trials. Baseline columns use their first 10 recorded trials.', '',
        'This is an inference-only intervention on a checkpoint trained with constant gating. '
        'Comparisons are descriptive; identical prompts and policy noise across runs have not been verified. '
        'These diagnostic scores are excluded from the main Table 1 and paper.', '',
    ]
    if timing_enabled:
        lines += ['The timing intervention adds `gate * 0.20 * (current velocity - saved training velocity mean)` '
                  'to physical XYZ actions once. Reconstruction uses horizons 0.20–0.96 s; '
                  'grasp-gate timing, neural inputs and checkpoint weights are unchanged. '
                  'The expert was trained with the original 0.00–0.76 s residual horizon. '
                  'Native policy-call latency excludes the added wrapper computation; '
                  'its host duration is recorded separately in the worker trace.', '']
    for directory in ([ROOT, TIMING_ROOT] if timing_enabled else [ROOT]):
        path = directory / 'comparison.md'
        temporary = path.with_name(f'.comparison.{os.getpid()}.tmp')
        temporary.write_text('\n'.join(lines))
        temporary.replace(path)


if __name__ == '__main__':
    render()
