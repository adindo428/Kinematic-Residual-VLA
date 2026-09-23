#!/usr/bin/env python3
"""Rebuild the historical DOM split from cached parquets, without downloads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

def mean_speed(item):
    episode, path = item
    table = pq.read_table(path, columns=['observation.environment_state'], use_threads=False)
    values = np.stack(table.column('observation.environment_state').to_pylist()).astype(np.float32)
    speeds = np.linalg.norm(values[..., 6:9], axis=-1)
    speeds = speeds[np.isfinite(speeds)]
    if not len(speeds):
        raise ValueError(f'No finite speeds for episode {episode}')
    return episode, float(np.mean(speeds))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/scratch/aparame/.cache/huggingface/lerobot/hzxie/DOM'))
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/dom_speed_ood_0p10_0p18_split.json')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'Refusing to overwrite existing split: {args.output}')
    if args.workers < 1:
        parser.error('--workers must be positive')
    jobs = [(i, args.root / 'data' / f'chunk-{i//1000:03d}' / f'episode_{i:06d}.parquet') for i in range(207306)]
    train, test = [], []
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for offset in range(0, len(jobs), 1000):
            for episode, speed in pool.map(mean_speed, jobs[offset:offset+1000]):
                if speed <= 0.10:train.append(episode)
                if speed >= 0.18:test.append(episode)
            if offset % 10000 == 0:
                print(f'{min(offset+1000,len(jobs))}/{len(jobs)} episodes; {time.monotonic()-start:.0f}s', flush=True)
    if (len(train), len(test)) != (3810, 119778):
        raise SystemExit(f'Historical count mismatch: train={len(train)}, test={len(test)}; expected 3810/119778. No split written.')
    result = {'metadata': {
        'dataset':'hzxie/DOM', 'candidate_episode_range':[0,207305],
        'speed_key':'observation.environment_state', 'velocity_slice':[6,9],
        'speed_stat':'mean', 'train_max_mps':0.10, 'test_min_mps':0.18,
        'train_episode_count':len(train), 'test_episode_count':len(test),
        'notes':'Recomputed 2026-09-22 using the historical float32 mean-speed algorithm; historical counts verified.'
    }, 'train':train, 'test':test}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(args.output)
    print(f'Wrote {args.output}: train={len(train)} test={len(test)}', flush=True)

if __name__ == '__main__':
    main()
