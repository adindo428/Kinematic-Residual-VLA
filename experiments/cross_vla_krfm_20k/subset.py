"""Deterministic task/speed selection from an already audited grouped split.

No tensor/data-science dependencies: selection is inspectable before GPU work.
Duplicate trajectories are indivisible, including when meeting exact quotas.
"""
from collections import Counter, defaultdict
from functools import reduce
from hashlib import sha256
from math import gcd, sqrt
import json

DEFAULT_QUOTAS = {'pick': 9000, 'place': 9000, 'long_horizon': 2000}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ranked(seed, key):
    return sha256(f'{seed}:{key}'.encode()).hexdigest()


def allocate(capacities, total):
    """Capped square-root allocation gives scarce speed bins useful coverage."""
    require(0 <= total <= sum(capacities.values()), 'Insufficient task capacity')
    result = {k: 0 for k in capacities}
    active = {k: v for k, v in capacities.items() if v}
    remaining = total
    while active and remaining:
        weight = sum(sqrt(v) for v in active.values())
        capped = [k for k, v in active.items() if remaining * sqrt(v) / weight >= v]
        if capped:
            for k in capped:
                result[k] = active[k]
                remaining -= active.pop(k)
            continue
        shares = {k: remaining * sqrt(v) / weight for k, v in active.items()}
        for k, share in shares.items():
            result[k] = int(share)
        left = remaining - sum(result[k] for k in active)
        for k in sorted(active, key=lambda k: (-(shares[k] - int(shares[k])), str(k)))[:left]:
            result[k] += 1
        remaining = 0
    return result


def select_groups(groups, target, seed):
    """Exact subset sum of duplicate-group sizes; hash order resolves ties.

    Repeated sizes need only target//size candidates. Python integer bitsets
    keep the reachability calculation small even for 20k episode quotas.
    """
    groups = sorted(groups, key=lambda g: ranked(seed, g[0]['duplicate_id']))
    reachable, snapshots = 1, []
    mask = (1 << (target + 1)) - 1
    used = Counter()
    candidates = []
    for group in groups:
        size = len(group)
        if used[size] >= target // size:
            continue
        used[size] += 1
        candidates.append(group)
        snapshots.append(reachable)
        reachable |= (reachable << size) & mask
        if (reachable >> target) & 1:
            break
    require((reachable >> target) & 1, f'Quota {target} cannot preserve duplicate groups')
    selected, remaining = [], target
    for group, before in reversed(list(zip(candidates, snapshots))):
        if (before >> remaining) & 1:
            continue
        selected.extend(group)
        remaining -= len(group)
    require(remaining == 0, 'Duplicate-group selection failed')
    return selected


def build_subset(episodes, parent_split, quotas=None, seed=42):
    quotas = dict(DEFAULT_QUOTAS if quotas is None else quotas)
    require(quotas and all(isinstance(v, int) and v >= 0 for v in quotas.values()), 'Invalid quotas')
    split = parent_split['splits']
    require(all(k in split for k in ('train', 'validation', 'test')), 'Missing parent split')
    owner = {}
    for name, ids in split.items():
        require(len(ids) == len(set(ids)), f'Duplicate episode IDs in {name}')
        for ep in ids:
            require(ep not in owner, 'Parent partitions overlap')
            owner[ep] = name
    rows = {r['episode_index']: r for r in episodes}
    require(len(rows) == len(episodes), 'Duplicate metadata episode IDs')
    require(set(owner) <= set(rows), 'Missing audited episode metadata')
    group_owner, groups = {}, defaultdict(list)
    for ep, name in owner.items():
        row = rows[ep]
        duplicate = row['duplicate_id']
        require(group_owner.setdefault(duplicate, name) == name, 'Duplicate trajectory crosses partitions')
        if name == 'train':
            require(row['length'] > 0, 'Empty training episode')
            groups[duplicate].append(row)
    strata = defaultdict(list)
    for group in groups.values():
        labels = {(r['task'], r['speed_bin']) for r in group}
        require(len(labels) == 1, 'Duplicate group crosses task/speed strata')
        strata[next(iter(labels))].append(group)
    selected = []
    for task, quota in sorted(quotas.items()):
        bins = {speed: gs for (family, speed), gs in strata.items() if family == task}
        capacities = {speed: sum(map(len, gs)) for speed, gs in bins.items()}
        require(bins or quota == 0, f'Unknown or unavailable task: {task}')
        allocation = allocate(capacities, quota)
        # Audited DOM has singleton or paired duplicates in each bin. A gcd
        # catches impossible parity quotas before attempting subset selection.
        for speed, target in allocation.items():
            divisor = reduce(gcd, (len(g) for g in bins[speed]))
            require(target % divisor == 0, 'Speed quota incompatible with duplicate grouping')
            selected.extend(select_groups(bins[speed], target, seed))
    ids = sorted(r['episode_index'] for r in selected)
    require(len(ids) == len(set(ids)) == sum(quotas.values()), 'Training quota mismatch')
    counts, frames = Counter(), Counter()
    for row in selected:
        key = json.dumps([row['task'], row['speed_bin']])
        counts[key] += 1
        frames[key] += row['length']
    result = {'seed': seed, 'policy': 'task quotas; capped square-root speed allocation; whole duplicate groups',
              'splits': {**split, 'train': ids}}
    report = {'seed': seed, 'task_quotas': quotas, 'train_episodes': len(ids),
              'train_anchors': sum(r['length'] for r in selected),
              'parent_train_episodes': len(split['train']),
              'parent_train_anchors': sum(rows[ep]['length'] for ep in split['train']),
              'task_speed_episodes': dict(sorted(counts.items())),
              'task_speed_anchors': dict(sorted(frames.items())),
              'validation_episodes': len(split['validation']), 'test_episodes': len(split['test']),
              'excluded_task_families': sorted({r['task'] for r in groups_flat(groups)} - {k for k,v in quotas.items() if v}),
              'duplicate_groups_preserved': True, 'held_out_membership_preserved': True}
    return result, report, [rows[ep] for ep in ids]


def groups_flat(groups):
    return (row for group in groups.values() for row in group)
