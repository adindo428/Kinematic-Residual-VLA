#!/usr/bin/env python3
"""Match procedural evaluation to training semantics and initial-state support."""
import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import sys

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / 'scratch/dom45'
ROOT = WORK / 'evaluations/prompt-matched20k-20261102'
BASE = WORK / 'evaluations/collection-matched20k-20261102'
FAMILY = WORK / 'training/fresh20k-20261025/family'
DATA = WORK / 'cache/huggingface/lerobot/hzxie/DOM'
SOURCE = WORK / 'official/source'
GENERATOR = WORK / 'training/krv100-fresh20k-h200-20261029/source/utils/instruction_generator.py'
CONTAINERS = {'bowl', 'box', 'plate', 'placemat', 'tray'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def rank(value):
    return hashlib.sha256(('prompt-matched-v2:42:' + str(value)).encode()).hexdigest()


def training():
    ids = set(read(FAMILY/'split.json')['splits']['train'])
    eps = {e['episode_index']: e for e in read(FAMILY/'episodes.json') if e['episode_index'] in ids}
    for line in (DATA/'meta/camera.jsonl').open():
        c = json.loads(line); i = c['episode_index']
        if i in ids:
            m = re.search(r'_franka_(.+)[ds]_O(\d+)_', c['filename'])
            eps[i].update(asset=m[1], rigid_count=int(m[2]), cameras=c['cameras'])
    for line in (DATA/'meta/episodes.jsonl').open():
        c = json.loads(line); i = c['episode_index']
        if i in ids: eps[i]['instruction'] = json.loads(c['tasks'])
    return eps


def container_match(tags, aliases):
    # Favor specific jointly observed descriptions over bare category overlap.
    shared = set(tags) & set(aliases)
    if not shared: return -1
    return sum(1 + len(x.split()) for x in shared) / (1 + len(set(tags) - set(aliases)))


def infer_counts(inst, task, total):
    def bounds(aliases, long_set=False):
        if long_set: return 2, None
        relative = [s for s in aliases if s.startswith('the ')]
        if not relative: return 1, 1
        text = ' '.join(relative)
        if re.search(r'\b(closer|farther|larger|smaller|taller|shorter)\b', text): return 2, 2
        if re.search(r'\b(closest|farthest|largest|smallest|tallest|shortest|middle)\b', text): return 3, None
        return 2, None  # Direction/velocity descriptions also imply multiple objects.
    omin, oexact = bounds(inst['objects'], task == 'long_horizon')
    cmin, cexact = bounds(inst.get('containers', []))
    possible = [(n,total-n) for n in range(1,total) if n>=omin and total-n>=cmin
                and (oexact is None or n==oexact) and (cexact is None or total-n==cexact)]
    return possible


def plan():
    import pyarrow.parquet as pq
    eps = training(); original = read(BASE/'selection.json')
    tags = read(WORK/'assets/objects/metadata.json')
    available = {p.stem: p for p in (WORK/'assets/objects').glob('*/*.usd') if p.parent.name in CONTAINERS}
    cases = []
    for old in original['cases']:
        candidates = [i for i,e in eps.items() if e['task'] == old['task'] and e['asset'] == old['target_asset']
                      and e['rigid_count'] == old['rigid_count'] and e['speed_bin'] == old['speed_bin']]
        assert candidates, ('No exact task/asset/count/bin training witness', old['name'])
        candidates = [i for i in candidates if infer_counts(eps[i]['instruction'],old['task'],old['rigid_count'])]
        assert candidates, ('No structurally compatible training witness', old['name'])
        witness = min(candidates, key=rank)
        allocations = infer_counts(eps[witness]['instruction'],old['task'],old['rigid_count'])
        allocation = min(allocations, key=lambda x:(abs(x[0]-old['n_objects']),x))
        candidates = [i for i in candidates if allocation in infer_counts(eps[i]['instruction'],old['task'],old['rigid_count'])]
        aliases = eps[witness]['instruction'].get('containers', [])
        scores = {a: container_match(tags.get(a+'.usd', {}).get('tags', []), aliases) for a in available}
        best = max(scores.values()); assert best > 0
        container = min((a for a,v in scores.items() if v == best), key=rank)
        states = {}
        for i in candidates:
            path = DATA/f'data/chunk-{i//1000:03d}/episode_{i:06d}.parquet'
            state = pq.read_table(path, columns=['observation.environment_state']).slice(0,1).to_pylist()[0]['observation.environment_state']
            states[str(i)] = state
        cases.append({**old, 'n_objects':allocation[0], 'n_containers':allocation[1],
            'count_inference':'Relational prompt forms plus total rigid count; unobserved identities remain uncertain.',
            'possible_count_allocations':allocations, 'container_asset': container, 'design_witness': witness,
            'witness_instruction': eps[witness]['instruction'],
            'training_witnesses': {str(i): eps[i]['instruction'] for i in candidates},
            'training_witness_lengths': {str(i): eps[i]['length'] for i in candidates},
            'training_first_states': states, 'geometry_candidates': 8})
    value = {**original, 'version': 2, 'cases': cases, 'training_prompt_generator_sha256': sha(GENERATOR),
        'selection_policy': 'Fixed held-out task/asset/count/speed-bin anchors. Exact training-stratum semantic witnesses; collection geometry ranked by prompt overlap and robot-relative initial position/velocity, never policy success.',
        'limitations': ['Approximately IID / training-distribution-matched; house membership is unverified.',
            'Initial-state comparison uses pre-physics generated states versus recorded first training states; it is an approximate diagnostic.',
            'Container identities are inferred from intrinsic descriptions, not verified original USD identities.',
            'Rigid-object/container allocation is inferred from task and total rigid count, not recovered original geometry.',
            'Five repeated executions share geometry. No claim of independent scene draws or expert-verified solvability.']}
    destination = ROOT/'selection.json'
    if destination.exists(): assert read(destination) == value, 'Frozen selection changed'
    else: write(destination, value)
    print('PLANNED', len(cases), 'cases; every case has exact task/asset/count/speed-bin training witnesses', flush=True)


def compatible(raw, case):
    result = []
    obj = set(raw['instruction']['objects']); con = set(raw['instruction'].get('containers', []))
    for i, inst in case['training_witnesses'].items():
        objects = sorted(obj & set(inst['objects']))
        containers = sorted(con & set(inst.get('containers', [])))
        if objects and containers:
            result.append((i, objects, containers))
    return result


def physical_match(raw, case):
    import numpy as np
    from scipy.spatial.transform import Rotation
    scene = raw['scene']; robot = scene['robot']['init_state']; obj = scene['object']['init_state']
    rotation = Rotation.from_quat(np.roll(robot['rot'], -1)).inv()
    pos = rotation.apply(np.array(obj['pos']) - robot['pos'])
    vel = rotation.apply(obj['lin_vel'])
    vec = np.r_[pos[:2], vel[:2]]
    distances = {}
    for i, state in case['training_first_states'].items():
        reference = np.array(state)[[0,1,6,7]]
        distances[i] = float(np.linalg.norm((vec - reference) / .25))
    nearest = min(distances, key=lambda k: (distances[k], rank(k)))
    ref = case['training_first_states'][nearest]
    return {'normalized_distance': distances[nearest], 'nearest_training_episode': int(nearest),
            'position_xy_m': pos[:2].tolist(), 'velocity_xy_m_s': vel[:2].tolist(),
            'position_xy_distance_m': float(np.linalg.norm(pos[:2]-np.array(ref[:2]))),
            'velocity_xy_distance_m_s': float(np.linalg.norm(vel[:2]-np.array(ref[6:8]))),
            'coordinate_scales': [.25,.25,.25,.25]}


def prompts(raw, case):
    spec = importlib.util.spec_from_file_location('frozen_training_instructions', GENERATOR)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    matches = compatible(raw, case); assert matches
    records = []
    for repeat in range(5):
        rng = random.Random(int(rank(f'{case["name"]}:{repeat}')[:8],16))
        match = rng.choices(matches, weights=[case['training_witness_lengths'][m[0]] for m in matches])[0]
        i, objects, containers = match
        metadata = {'task': case['task'], 'objects': [rng.choice(objects)], 'containers': [rng.choice(containers)]}
        state = random.getstate()
        random.seed(rng.randrange(2**31))
        try: text = module.InstructionGenerator.generate_instruction(metadata)
        finally: random.setstate(state)
        records.append({'trial': repeat+1, 'text': text, 'training_witness_episode': int(i), 'metadata': metadata})
    return records


def generate():
    from unittest.mock import patch
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True, enable_cameras=True, device='cuda:0', kit_args=os.environ['KRVLA_KIT_ARGS'])
    import numpy as np
    import torch
    import yaml
    sys.path[:0] = [str(SOURCE), str(SOURCE/'simulations')]
    import simulations.simulate as sim
    selection = read(ROOT/'selection.json'); cfg = yaml.safe_load((SOURCE/'simulations/configs/sim_cfg.yaml').read_text())
    cfg.update(device='cuda:0', num_envs=1, disable_fabric=False, enable_cameras=True, debug=False, disable_sm=False, path_tracing=False)
    assets = {p.stem: p for p in (WORK/'assets/objects').glob('*/*.usd') if p.parent.name != 'unseen'}
    alltags = read(WORK/'assets/objects/metadata.json'); sizes = {}; tests = []; evidence = []
    original_states = sim._get_object_states; original_dynamic = sim._get_dynamic_object_state; listdir = os.listdir
    try:
        for case in selection['cases']:
            destination = ROOT/'configs'/f'{case["name"]}.json'
            if destination.exists():
                raw = read(destination); audit = read(ROOT/'case_evidence'/f'{case["name"]}.json')
            else:
                container_category = assets[case['container_asset']].parent.name
                names = selection['target_asset_support'][case['task']] + [a for a,p in assets.items() if p.parent.name == container_category]
                metadata = {}
                for name in sorted(set(names)):
                    path = assets[name]
                    if name not in sizes: sizes[name] = sim._get_object_size(str(path))
                    metadata[name+'.usd'] = {'file_path': str(path), 'size': sizes[name], 'category': path.parent.name,
                        'tags': alltags.get(name+'.usd', {}).get('tags', [path.parent.name])}
                candidates = []
                for attempt in range(case['geometry_candidates']):
                    local = copy.deepcopy(cfg)
                    local['scene']['objects'].update(n_objects=case['n_objects'])
                    local['scene']['containers'].update(n_containers=case['n_containers'], categories=[container_category])
                    def states(*args, **kwargs):
                        choose = random.choice; used = set()
                        first_dynamic = True
                        def dynamic(*values, **named):
                            nonlocal first_dynamic
                            if first_dynamic:
                                first_dynamic = False
                                values = list(values); values[2] = [case['initial_speed']]*2
                            return original_dynamic(*values, **named)
                        def choice(values):
                            if values and isinstance(values[0], dict) and 'file_path' in values[0]:
                                kind = 'container' if values[0]['category'] in CONTAINERS else 'object'
                                if kind not in used:
                                    used.add(kind)
                                    target = case['container_asset'] if kind == 'container' else case['target_asset']
                                    return next(v for v in values if Path(v['file_path']).stem == target)
                            return choose(values)
                        with patch.object(random,'choice',side_effect=choice), patch.object(sim,'_get_dynamic_object_state',side_effect=dynamic):
                            return original_states(*args, **kwargs)
                    seed = int(rank(f'{case["name"]}:geometry:{attempt}')[:8],16) % (2**31-1)
                    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                    with patch.object(sim,'_get_object_states',states), patch('os.listdir',side_effect=lambda p:sorted(listdir(p))):
                        env, tags, _, _ = sim.get_env_cfg(local,case['task'].replace('_','-'),'franka',metadata,str(WORK/'assets/full-scene-release'))
                    if not hasattr(env.scene,'container'): continue
                    raw = sim.get_object_without_numpy(env.to_dict())
                    raw.update(seed=seed, instruction={'task':case['task'].replace('_','-'), **tags})
                    n = sum(isinstance(v,dict) and v.get('class_type')=='isaaclab.assets.rigid_object.rigid_object:RigidObject' for v in raw['scene'].values())
                    if n != case['rigid_count'] or not compatible(raw,case): continue
                    match = physical_match(raw,case)
                    candidates.append((match['normalized_distance'], attempt, raw, match))
                assert candidates, ('No geometry with training-supported joint prompt',case['name'])
                _, attempt, raw, match = min(candidates,key=lambda x:(x[0],x[1]))
                schedule = prompts(raw,case)
                raw.update(evaluation_case_id=case['name'],evaluation_prompts=[p['text'] for p in schedule])
                audit = {'name':case['name'],'task':case['task'],'target_asset':case['target_asset'],
                    'container_asset':case['container_asset'],'training_stratum_witnesses':len(case['training_witnesses']),
                    'compatible_prompt_witnesses':len(compatible(raw,case)),'candidate_count':case['geometry_candidates'],
                    'compatible_candidates':len(candidates),'selected_candidate':attempt,'physical_match':match,'prompts':schedule}
                write(destination,raw); write(ROOT/'case_evidence'/f'{case["name"]}.json',audit)
            tests.append({'name':case['name'],'path':str(destination),'config_sha256':sha(destination),
                          'task':case['task'],'horizon_seconds':raw['episode_length_s'],'condition':'approx_iid'})
            evidence.append(audit)
            print('FROZEN',case['name'],'container',case['container_asset'],'prompt witnesses',audit['compatible_prompt_witnesses'],flush=True)
        for task in ['pick','place','long_horizon']:
            p=ROOT/'test_lists'/f'{task}.txt';p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(''.join(t['path']+'\n' for t in tests if t['task']==task))
        (ROOT/'test_lists/pilot.txt').write_text(''.join(next(t['path'] for t in tests if t['task']==task)+'\n' for task in ['pick','place','long_horizon']))
        files = {str(p):sha(p) for p in [Path(__file__),GENERATOR,SOURCE/'simulations/simulate.py',SOURCE/'simulations/configs/sim_cfg.yaml']}
        scene_assets = set()
        for t in tests:
            for v in read(t['path'])['scene'].values():
                if isinstance(v,dict):
                    p=Path(v.get('spawn',{}).get('usd_path',''))
                    if p.is_file() and str(p).startswith(str(WORK/'assets')): scene_assets.add(p)
        write(ROOT/'protocol.json',{'benchmark':'Training-prompt-matched approximate IID v2','tests':tests,
            'task_quotas':{'pick':9,'place':9,'long_horizon':2},'repeats':5,'scored_trials':100,'pilot_trials':3,
            'selection_sha256':sha(ROOT/'selection.json'),'asset_sha256':{str(p):sha(p) for p in sorted(scene_assets)},
            'files_sha256':files,'limitations':selection['limitations'],
            'scoring':'Upstream manipulation completion, tolerance 0.07 m; no-action trials invalidate the phase.',
            'prompt_schedule':'Five frozen training-supported prompts per case, matched across methods; reset artifacts do not advance prompt index.',
            'auxiliary_target_semantics':read(BASE/'protocol.json')['auxiliary_target_semantics']})
        write(ROOT/'distribution_evidence.json',{'cases':evidence,'all_joint_prompts_training_supported':True,
            'all_strata_training_supported':True,'house_membership_verified':False,'policy_results_used_for_selection':False,
            'candidate_ranking':'Closest robot-relative XY position/velocity to training-stratum first states, scales 0.25 m and 0.25 m/s; pre-physics comparison.',
            'task_mix':{'train_episodes':{'pick':9000,'place':9000,'long_horizon':2000},'eval_cases':{'pick':9,'place':9,'long_horizon':2}}})
        print('PROTOCOL_COMPLETE',flush=True)
    finally: launcher.app.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--generate',action='store_true');args=p.parse_args()
    generate() if args.generate else plan()
