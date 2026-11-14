"""Generate an offline collection target schedule; never connects to hardware."""
import argparse
import csv
import json
from pathlib import Path
import random


def build_schedule(plan):
    rng = random.Random(plan['seed'])
    rows = []
    blocks = []
    for task in plan['tasks']:
        for split, count in plan['split_blocks_per_task'].items():
            for block in range(count):
                blocks.append((task, split, block))
    rng.shuffle(blocks)
    for task, split, block in blocks:
        # A block is a dedicated recording session; it must never span splits.
        session = f"{task['id']}-{split}-{block:02d}"
        speeds = [name for name, spec in plan['speed_bins'].items()
                  for _ in range(spec['slots_per_block'])]
        rng.shuffle(speeds)
        for slot, speed in enumerate(speeds):
            spec = plan['speed_bins'][speed]
            rows.append(dict(target_id=f'{session}-{slot:02d}', session_id=session,
                             task_id=task['id'], split=split, speed_bin=speed,
                             min_mps=spec['min_mps'], max_mps=spec['max_mps'],
                             launch_direction='left_to_right' if slot % 2 == 0 else 'right_to_left',
                             instruction=task['instruction']))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path(__file__).with_name('collection_plan.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = build_schedule(json.loads(args.plan.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite a schedule that may already have collection annotations.
    with args.output.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} accepted-demonstration targets written to {args.output}')


if __name__ == '__main__':
    main()
