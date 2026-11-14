#!/usr/bin/env python3
"""Freeze a small procedural collection-distribution evaluation, without policy outcomes."""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / 'scratch/dom45'
ROOT = WORK / 'evaluations/collection-matched20k-20261102'
FAMILY = WORK / 'training/fresh20k-20261025/family'
DATA = WORK / 'cache/huggingface/lerobot/hzxie/DOM'
QUOTAS = {'pick': 9, 'place': 9, 'long_horizon': 2}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rank(value):
    return hashlib.sha256(('collection-matched20k:42:' + str(value)).encode()).hexdigest()


def allocate(histogram, budget):
    total = sum(histogram.values())
    values = {key: budget * count / total for key, count in histogram.items()}
    quotas = {key: int(value) for key, value in values.items()}
    for key in sorted(values, key=lambda k: (-(values[k] - quotas[k]), rank(k)))[:budget-sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def plan():
    import pyarrow.parquet as pq
    import numpy as np
    split = read(FAMILY / 'split.json')['splits']
    train, heldout = set(split['train']), set(split['test'])
    assert len(train) == 20000 and train.isdisjoint(heldout)
    episodes = {e['episode_index']: e for e in read(FAMILY / 'episodes.json')}
    cameras = {}
    selected_ids = train | heldout
    for line in (DATA / 'meta/camera.jsonl').open():
        row = json.loads(line)
        if row['episode_index'] in selected_ids:
            match = re.search(r'_franka_(.+)([ds])_O(\d+)_(\d+)_', row['filename'])
            if match:
                cameras[row['episode_index']] = {**row, 'asset': match[1], 'dynamic': match[2] == 'd',
                                                 'rigid_count': int(match[3])}
    support = defaultdict(set)
    counts = defaultdict(Counter)
    for i in sorted(train):
        row, ep = cameras[i], episodes[i]
        support[ep['task']].add(row['asset'])
        counts[ep['task']][(row['rigid_count'], ep['speed_bin'])] += 1
    available = {p.stem: p for p in (WORK / 'assets/objects').glob('*/*.usd') if p.parent.name != 'unseen'}
    tasks = {}
    for line in (DATA / 'meta/episodes.jsonl').open():
        row = json.loads(line)
        if row['episode_index'] in train:
            tasks[row['episode_index']] = row['tasks']
    metadata = read(WORK / 'assets/objects/metadata.json')
    # Container IDs are absent from episode filenames. Retain only assets whose
    # intrinsic descriptions actually occur in training container instructions.
    train_container_tags = set()
    for values in tasks.values():
        if isinstance(values, str):
            values = [values]
        for value in values:
            value = ast.literal_eval(value) if isinstance(value, str) and value.startswith('{') else value
            if isinstance(value, dict):
                train_container_tags.update(value.get('containers', []))
    containers = sorted(name for name, path in available.items()
                        if path.parent.name in ('bowl', 'box', 'plate', 'placemat', 'tray')
                        and set(metadata.get(name + '.usd', {}).get('tags', [])).intersection(train_container_tags))
    assert containers, 'No training-supported container assets'
    selected, used_categories = [], defaultdict(set)
    for task, budget in QUOTAS.items():
        for (rigid_count, speed_bin), quota in sorted(allocate(counts[task], budget).items()):
            candidates = [i for i in heldout if i in cameras and episodes[i]['task'] == task
                          and cameras[i]['asset'] in support[task] & available.keys()
                          and cameras[i]['rigid_count'] == rigid_count and episodes[i]['speed_bin'] == speed_bin]
            for _ in range(quota):
                assert candidates, (task, rigid_count, speed_bin)
                i = min(candidates, key=lambda k: (available[cameras[k]['asset']].parent.name in used_categories[task], rank(k)))
                candidates.remove(i)
                row = cameras[i]
                category = available[row['asset']].parent.name
                used_categories[task].add(category)
                parquet = DATA / f'data/chunk-{i//1000:03d}/episode_{i:06d}.parquet'
                state = pq.read_table(parquet, columns=['observation.environment_state']).slice(0, 1).to_pylist()[0]['observation.environment_state']
                initial_speed = float(np.linalg.norm(state[6:9]))
                assert np.isfinite(initial_speed) and initial_speed > 0
                ordinal = sum(s['task'] == task for s in selected)
                name = f'{task}_{ordinal:02d}'
                # New deterministic seeds: these are NOT reconstructions of held-out episodes.
                seed = int(rank(name)[:8], 16) % (2**31-1)
                selected.append({'name': name, 'task': task, 'anchor_episode': i, 'anchor_duplicate_id': episodes[i]['duplicate_id'],
                                 'target_asset': row['asset'], 'category': category, 'rigid_count': rigid_count,
                                 'speed_bin': speed_bin, 'anchor_mean_speed': episodes[i]['mean_speed'],
                                 'initial_speed': initial_speed, 'seed': seed,
                                 'n_objects': rigid_count-1 if task != 'place' else 1,
                                 'n_containers': 1 if task != 'place' else rigid_count-1})
    assert len(selected) == 20 and Counter(x['task'] for x in selected) == Counter(QUOTAS)
    train_groups = {episodes[i]['duplicate_id'] for i in train}
    assert train_groups.isdisjoint(x['anchor_duplicate_id'] for x in selected)
    value = {'version': 1, 'selection_seed': 42, 'scored_trials': 100, 'repeats': 5, 'task_quotas': QUOTAS,
             'cases': selected, 'target_asset_support': {k: sorted(v & available.keys()) for k, v in support.items()},
             'container_asset_support': containers,
             'training_joint_histogram': {t: {str(k): v for k, v in c.items()} for t, c in counts.items()},
             'files_sha256': {str(p): sha(p) for p in [FAMILY/'split.json', FAMILY/'episodes.json',
                                DATA/'meta/camera.jsonl', WORK/'official/source/simulations/configs/sim_cfg.yaml', Path(__file__)]},
             'limitations': ['Procedural collection-supported cases, not exact training or held-out episode replays.',
                 'House membership and distractor/container identities and poses were not retained in the LeRobot export.',
                 'Rigid counts and trajectory-speed bins use training proportions rounded to 20 cases; tiny strata may be absent.',
                 'Pick uses one container and remaining rigid bodies as objects; place uses one object and remaining bodies as containers. This allocation is a design assumption.',
                 'Initial target speeds come from deterministic held-out anchors with training-supported task/count/speed-bin/asset.',
                 'All geometry is generated before evaluating any policy; no successful-policy-case selection.',
                 'Five repeats share frozen initial geometry and measure execution repeatability, not 100 independent scenarios.']}
    ROOT.mkdir(parents=True, exist_ok=True)
    destination = ROOT / 'selection.json'
    if destination.exists():
        assert read(destination) == value, 'Refusing to change an existing frozen selection'
    else:
        write(destination, value)
    print(json.dumps({'root': str(ROOT), 'cases': len(selected), 'trials': 100, 'tasks': QUOTAS,
                      'containers': len(containers), 'initial_speed_range': [min(x['initial_speed'] for x in selected), max(x['initial_speed'] for x in selected)]}))


def generate():
    import copy
    import logging
    import random
    import sys
    import types
    from unittest.mock import patch
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True, enable_cameras=True, device='cuda:0', kit_args=__import__('os').environ['KRVLA_KIT_ARGS'])
    import numpy as np
    import torch
    import yaml
    source = WORK / 'official/source'
    sys.path[:0] = [str(source), str(source / 'simulations')]
    import simulations.simulate as sim
    selection = read(ROOT / 'selection.json')
    cfg = yaml.safe_load((source / 'simulations/configs/sim_cfg.yaml').read_text())
    cfg.update(device='cuda:0', num_envs=1, disable_fabric=False, enable_cameras=True, debug=False, disable_sm=False, path_tracing=False)
    objects_dir = WORK / 'assets/objects'
    available = {p.stem: p for p in objects_dir.glob('*/*.usd') if p.parent.name != 'unseen'}
    meta_tags = read(objects_dir / 'metadata.json')
    size_cache = {}
    original_states = sim._get_object_states
    original_listdir = __import__('os').listdir
    # Freeze directory ordering, which the upstream generator leaves unspecified.
    def sorted_listdir(path):
        return sorted(original_listdir(path))
    tests = []
    try:
        for case in selection['cases']:
            destination = ROOT / 'configs' / (case['name'] + '.json')
            if destination.exists():
                raw = read(destination)
            else:
                local = copy.deepcopy(cfg)
                local['scene']['objects']['n_objects'] = case['n_objects']
                local['scene']['containers']['n_containers'] = case['n_containers']
                local['scene']['objects']['moving_speed'] = [case['initial_speed']] * 2
                asset_names = sorted(selection['target_asset_support'][case['task']] + selection['container_asset_support'])
                metadata = {}
                for name in asset_names:
                    path = available[name]
                    if name not in size_cache:
                        size_cache[name] = sim._get_object_size(str(path))
                    metadata[name + '.usd'] = {'file_path': str(path), 'size': size_cache[name],
                        'category': path.parent.name, 'tags': meta_tags.get(name + '.usd', {}).get('tags', [path.parent.name])}
                # Keep upstream placement logic, while ensuring the first object is the selected supported target.
                def states(*args, **kwargs):
                    choose = random.choice
                    first = True
                    def choice(values):
                        nonlocal first
                        if first and values and isinstance(values[0], dict) and 'file_path' in values[0]:
                            first = False
                            return next(v for v in values if Path(v['file_path']).stem == case['target_asset'])
                        return choose(values)
                    with patch.object(random, 'choice', side_effect=choice):
                        return original_states(*args, **kwargs)
                random.seed(case['seed']); np.random.seed(case['seed']); torch.manual_seed(case['seed'])
                with patch.object(sim, '_get_object_states', states), patch('os.listdir', side_effect=sorted_listdir):
                    env, tags, objects, _ = sim.get_env_cfg(local, case['task'].replace('_', '-'), 'franka', metadata, str(WORK / 'assets/full-scene-release'))
                assert hasattr(env.scene, 'container'), 'Geometry generation did not fit a target container'
                raw = sim.get_object_without_numpy(env.to_dict())
                raw.update(seed=case['seed'], instruction={'task': case['task'].replace('_', '-'), **tags})
                assert Path(raw['scene']['object']['spawn']['usd_path']).stem == case['target_asset']
                n = sum(isinstance(v, dict) and v.get('class_type') == 'isaaclab.assets.rigid_object.rigid_object:RigidObject' for v in raw['scene'].values())
                assert n == case['rigid_count'], (case['name'], n, case['rigid_count'])
                write(destination, raw)
            tests.append({'name': case['name'], 'path': str(destination), 'config_sha256': sha(destination),
                          'task': case['task'], 'horizon_seconds': raw['episode_length_s']})
            print('FROZEN', case['name'], flush=True)
        for task in QUOTAS:
            listing = ROOT / 'test_lists' / (task + '.txt')
            listing.parent.mkdir(parents=True, exist_ok=True)
            listing.write_text(''.join(t['path'] + '\n' for t in tests if t['task'] == task))
        (ROOT / 'test_lists/pilot.txt').write_text(''.join(next(t['path'] for t in tests if t['task'] == task) + '\n' for task in QUOTAS))
        assets = set()
        for test in tests:
            for value in read(test['path'])['scene'].values():
                if isinstance(value, dict):
                    path = Path(value.get('spawn', {}).get('usd_path', ''))
                    if path.is_file() and str(path).startswith(str(WORK / 'assets')):
                        assets.add(path)
        write(ROOT / 'protocol.json', {'benchmark': 'Procedural collection-supported 20k evaluation v1',
            'tests': tests, 'task_quotas': QUOTAS, 'repeats': 5, 'scored_trials': 100, 'pilot_trials': 3,
            'selection_sha256': sha(ROOT / 'selection.json'), 'asset_sha256': {str(p): sha(p) for p in sorted(assets)},
            'files_sha256': {str(source/p): sha(source/p) for p in ['simulations/simulate.py', 'simulations/configs/scene_cfg.py',
                'simulations/configs/sim_cfg.yaml', 'simulations/configs/termination_cfg.py']},
            'limitations': selection['limitations'], 'scoring': 'Unchanged upstream manipulation completion; no-action trials invalidate a phase.'})
        print('PROTOCOL_COMPLETE', flush=True)
    finally:
        launcher.app.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    generate() if args.generate else plan()
