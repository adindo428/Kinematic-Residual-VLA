"""Freeze a conservative DOM low-clutter split without changing trajectories.

Run dataset work on an allocated CPU node. Oxx is the upstream count of ALL
RigidObject entries, including containers. Nonempty container aliases prove
at least one container exists, so O<=3 bounds task objects by two. Alias list
length is deliberately never interpreted as a physical object count.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build(metadata, family, subset_module, output, parent_split, comparison_split):
    spec = importlib.util.spec_from_file_location('audited_subset', subset_module)
    subset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(subset)
    rows = json.loads((family / 'episodes.json').read_text())
    parent = json.loads(parent_split.read_text())
    comparison = json.loads(comparison_split.read_text())
    for name in ('validation', 'test'):
        assert set(parent['splits'][name]) == set(comparison['splits'][name]), 'Wrong comparison parent partition'
    tasks = {}
    for line in (metadata / 'episodes.jsonl').open():
        entry = json.loads(line)
        task = entry['tasks']
        tasks[entry['episode_index']] = json.loads(task) if isinstance(task, str) else task
    counts = {}
    for line in (metadata / 'camera.jsonl').open():
        entry = json.loads(line)
        match = re.search(r'_O(\d+)_', entry['filename'])
        if not match:
            raise ValueError(f"Missing rigid-object count: {entry['episode_index']}")
        if entry['episode_index'] in counts:
            raise ValueError('Duplicate camera episode metadata')
        counts[entry['episode_index']] = int(match[1])
    groups = defaultdict(list)
    for row in rows:
        ep = row['episode_index']
        assert tasks[ep]['task'].replace('-', '_') == row['task']
        groups[row['duplicate_id']].append(ep)
    eligible = {ep for ep, n in counts.items()
                if 2 <= n <= 3 and bool(tasks[ep].get('containers'))}
    # Reject the entire duplicate group if any member fails the clutter rule.
    group_safe = {ep for ids in groups.values() if set(ids) <= eligible for ep in ids}
    filtered_parent = {**parent, 'splits': {
        name: [ep for ep in ids if ep in group_safe]
        for name, ids in parent['splits'].items()}}
    split, report, selected = subset.build_subset(rows, filtered_parent, seed=42)
    lookup = {row['episode_index']: row for row in rows}
    partitions = {}
    owners = {}
    for name, ids in split['splits'].items():
        for ep in ids:
            assert ep in group_safe
            assert owners.setdefault(lookup[ep]['duplicate_id'], name) == name
        partitions[name] = {
            'episodes': len(ids),
            'frames': sum(lookup[ep]['length'] for ep in ids),
            'task_episodes': dict(Counter(lookup[ep]['task'] for ep in ids)),
            'total_rigid_objects': dict(Counter(counts[ep] for ep in ids)),
        }
    assert len(split['splits']['train']) == 20000
    held_out = set(comparison['splits']['validation']) | set(comparison['splits']['test'])
    assert not set(split['splits']['train']) & held_out, 'Comparison held-out leakage'
    split['policy'] += '; O<=3 with nonempty container aliases; <=2 task objects'
    report.update(
        name='DOM low-clutter 20k v1', dataset_revision='bbf6baa15f7253746d271a6e958e037afa537b1e',
        clutter_rule={'maximum_total_rigid_objects': 3, 'minimum_containers_proven': 1,
                      'maximum_task_objects_proven': 2,
                      'proof': 'Upstream get_episode_name counts all rigid objects; nonempty container aliases require a container.',
                      'alias_lengths_are_not_object_counts': True,
                      'scope': 'Conservative subset; scenes with more containers may be excluded even if task-object count is low.'},
        eligible_audited_episodes=len(group_safe),
        groups_rejected_if_any_member_ineligible=True,
        partitions=partitions, inputs_sha256={
            'episodes.jsonl': digest(metadata / 'episodes.jsonl'),
            'camera.jsonl': digest(metadata / 'camera.jsonl'),
            'audited_episodes.json': digest(family / 'episodes.json'),
            'parent_split.json': digest(parent_split),
            'comparison_split.json': digest(comparison_split),
            'subset.py': digest(subset_module),
            'prepare_low_clutter_dataset.py': digest(__file__)},
        selection_uses_policy_results=False,
        comparison_held_out_overlap_with_training=0,
        trajectories_modified=False,
        held_out_policy='Filter within existing partitions; never promote held-out episodes into training.',
        evaluator_alignment='Same <=2 task-object cap and pick/place/long_horizon families; not identical scene or condition distribution.',
    )
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {'split.json': split, 'subset-report.json': report,
                 'selected-episodes.json': [{**row, 'total_rigid_objects': counts[row['episode_index']],
                                             'task_objects_upper_bound': counts[row['episode_index']] - 1}
                                            for row in selected]}
    for name, value in artifacts.items():
        text = json.dumps(value, indent=2, sort_keys=True) + '\n'
        path = output / name
        if path.exists() and path.read_text() != text:
            raise ValueError(f'Refusing to overwrite a different frozen artifact: {name}')
        path.write_text(text)
    hashes = {name: digest(output / name) for name in artifacts}
    (output / 'sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'partitions': partitions, 'sha256': hashes}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('metadata', 'family', 'subset-module', 'output', 'parent-split', 'comparison-split'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    build(args.metadata, args.family, args.subset_module, args.output, args.parent_split, args.comparison_split)
