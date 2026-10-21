"""Prepare a compact DOM experiment from frozen audited parent receipts.

Uses existing data and pinned model files in place. Never downloads data or
loads trained parent weights. Run heavy statistics work as a CPU batch job.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

from common import atomic, sha, fingerprint
from subset import DEFAULT_QUOTAS, build_subset, require


def read(path):
    return json.loads(Path(path).read_text())


def frozen_json(path, value):
    if path.exists():
        require(read(path) == value, f'Conflicting frozen preparation: {path}')
    else:
        atomic(path, value)


def select(root, parent, quotas, seed, epochs):
    parent_prepared = read(parent / 'prepared.json')
    require(parent_prepared['complete'], 'Parent preparation incomplete')
    for key, name in [('statistics_sha256', 'statistics.json'), ('split_sha256', 'split.json'),
                      ('checkpoints_sha256', 'checkpoints.json'), ('implementation_sha256', 'implementation-lock.json')]:
        require(sha(parent / name) == parent_prepared[key], f'Parent artifact changed: {name}')
    provenance = read(parent / 'data-provenance.json')
    audited = Path(provenance['parent'])
    episodes = read(audited / 'episodes.json')
    split, report, selected = build_subset(episodes, read(parent / 'split.json'), quotas, seed)
    report.update(parent=str(parent), parent_prepared_sha256=sha(parent / 'prepared.json'),
                  parent_split_sha256=sha(parent / 'split.json'),
                  audited_episodes_sha256=sha(audited / 'episodes.json'),
                  dataset_revision=provenance['dataset_revision'], min_epochs=epochs)
    frozen_json(root / 'split.json', split)
    frozen_json(root / 'subset-report.json', report)
    frozen_json(root / 'selected-episodes.json', selected)
    for name in ('parquet_hashes.json', 'video_hashes.json', 'checkpoints.json', 'implementation-lock.json'):
        target = root / name
        if target.exists():
            require(sha(target) == sha(parent / name), f'Inherited artifact changed: {name}')
        else:
            shutil.copyfile(parent / name, target)
    (root / 'recipes').mkdir(exist_ok=True)
    for file in sorted((parent / 'recipes').glob('*.json')):
        recipe = read(file)
        count, batch = report['train_anchors'], recipe['global_batch']
        steps = epochs * (count // batch)
        recipe.update(training_anchors=count, training_episodes=len(selected), min_epochs=epochs,
                      steps=steps, schedule_steps=steps, schedule_policy='fresh_subset_cosine',
                      warmup_updates=min(1000, max(1, steps // 20)),
                      initialization='released_pretrained', subset_sha256=sha(root / 'split.json'),
                      seed=seed, budget_authorization='20k subset requested 2026-10-20; five subset epochs')
        frozen_json(root / 'recipes' / file.name, recipe)
    for folder in ('profiles', 'runs', 'logs'):
        (root / folder).mkdir(exist_ok=True)
    for name in ('env', 'upstream-vlash'):
        path = root / name
        if not path.exists():
            path.symlink_to((parent / name).resolve(), target_is_directory=True)
    return provenance, report, selected


def statistics(root, provenance, selected, seed, sample_count, workers):
    import numpy as np
    import pyarrow.parquet as pq

    data = Path(provenance['dataset_root'])
    lengths = {r['episode_index']: r['length'] for r in selected}
    eps = sorted(lengths)
    ends = np.cumsum([lengths[ep] for ep in eps])
    chosen = np.random.default_rng(seed).choice(int(ends[-1]), min(sample_count, int(ends[-1])), replace=False)
    groups = {}
    for idx in chosen:
        local = int(np.searchsorted(ends, idx, side='right'))
        groups.setdefault(eps[local], []).append(int(idx - (ends[local-1] if local else 0)))
    hashes = read(root / 'parquet_hashes.json')
    # Velocity mean uses every retained training frame, not the full-data mean.
    # Action/state statistics use the same deterministic anchor estimator as
    # the parent, restricted to the newly selected training episodes.
    def episode(ep):
        file = data / f'data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet'
        require(sha(file) == hashes[str(ep)], f'Parquet changed: {ep}')
        table = pq.read_table(file, columns=['action', 'observation.state', 'observation.environment_state'])
        require(len(table) == lengths[ep], f'Episode length changed: {ep}')
        action, state, environment = [np.asarray(table[k].to_pylist(), dtype=np.float32) for k in table.column_names]
        require(np.isfinite(action).all() and np.isfinite(state).all() and np.isfinite(environment[:, 6:9]).all(), f'Nonfinite data: {ep}')
        actions, states = [], []
        for frame in groups.get(ep, []):
            delay = (ep + frame) % 4
            reference = action[min(frame + delay - 1, len(action)-1), :6].copy() if delay else state[frame].copy()
            chunk = action[frame+delay:min(frame+delay+20, len(action))].copy()
            chunk[:, :6] -= reference[:6]
            actions.append(chunk)
            states.append(reference)
        return (np.concatenate(actions) if actions else np.empty((0, 7), dtype=np.float32),
                np.stack(states) if states else np.empty((0, 6), dtype=np.float32),
                environment[:, 6:9].sum(axis=0, dtype=np.float64), len(table))

    actions, states, velocity, frames = [], [], np.zeros(3, dtype=np.float64), 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # A bounded submission window avoids retaining thousands of completed
        # episode arrays behind a slow filesystem read on Python <3.14.
        for start in range(0, len(eps), workers * 4):
            for a, s, v, n in pool.map(episode, eps[start:start+workers*4]):
                if len(a):
                    actions.append(a)
                if len(s):
                    states.append(s)
                velocity += v
                frames += n
            if start % (workers * 4 * 25) == 0:
                print(json.dumps({'statistics_episodes': min(start+workers*4, len(eps)), 'of': len(eps)}), flush=True)
    require(frames == int(ends[-1]), 'Statistics frame count differs')
    stats = {}
    for key, arrays in [('action', actions), ('observation.state', states)]:
        x = np.concatenate(arrays)
        require(len(x) and np.isfinite(x).all(), f'Invalid statistics: {key}')
        stats[key] = {'mean': x.mean(axis=0).tolist(), 'std': x.std(axis=0).clip(1e-6).tolist(),
                      'q01': np.quantile(x, .01, axis=0).tolist(), 'q99': np.quantile(x, .99, axis=0).tolist()}
    frozen_json(root / 'statistics.json', stats)
    return (velocity / frames).tolist(), len(chosen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--quotas', default=json.dumps(DEFAULT_QUOTAS), help='JSON task-to-episode quota; zero explicitly excludes a family')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--statistics-anchors', type=int, default=200_000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--select-only', action='store_true')
    args = parser.parse_args()
    require(args.epochs >= 5 and args.statistics_anchors > 0 and 1 <= args.workers <= 16, 'Invalid preparation budget')
    root, parent = args.root.resolve(), args.parent.resolve()
    require(root != parent, 'Subset preparation must use a fresh experiment directory')
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'prepared.json').exists():
        prepared = read(root / 'prepared.json')
        for name, expected in prepared['files'].items():
            require(sha(root / name) == expected, f'Prepared artifact changed: {name}')
        report = read(root / 'subset-report.json')
        require(report['task_quotas'] == json.loads(args.quotas) and report['seed'] == args.seed and
                report['min_epochs'] == args.epochs and report['parent'] == str(parent), 'Preparation parameters differ')
        print(json.dumps({'already_prepared': True, 'report': report}), flush=True)
        return
    provenance, report, selected = select(root, parent, json.loads(args.quotas), args.seed, args.epochs)
    print(json.dumps({'selection_complete': True, **report}), flush=True)
    if args.select_only:
        return
    mean, sample_count = statistics(root, provenance, selected, args.seed, args.statistics_anchors, args.workers)
    frozen_json(root / 'data-provenance.json', {**provenance, 'parent': str(parent),
        'split_sha256': sha(root / 'split.json'), 'velocity_mean': mean,
        'normalization': 'subset train only; deterministic uniform anchors; original physical actions',
        'statistics_anchors': sample_count, 'statistics_seed': args.seed,
        'velocity_mean_source': 'all selected training frames',
        'initialization': 'released_pretrained; no full-DOM trained checkpoint inheritance'})
    files = ['statistics.json', 'split.json', 'checkpoints.json', 'implementation-lock.json',
             'data-provenance.json', 'parquet_hashes.json', 'video_hashes.json',
             'subset-report.json', 'selected-episodes.json']
    files += [str(p.relative_to(root)) for p in sorted((root / 'recipes').glob('*.json'))]
    source = {p.name: sha(p) for p in sorted(root.glob('*.py'))}
    frozen_json(root / 'source-manifest.json', {'files': source, 'fingerprint': fingerprint(source)})
    atomic(root / 'prepared.json', {'complete': True, 'files': {name: sha(root / name) for name in files},
                                  'source_sha256': sha(root / 'source-manifest.json')})
    print(json.dumps({'preparation_complete': True, 'train_episodes': len(selected),
                      'train_anchors': report['train_anchors'], 'velocity_mean': mean}), flush=True)


if __name__ == '__main__':
    main()
