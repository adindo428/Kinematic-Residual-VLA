"""Merge completed object runs and split entire paired trajectories, never individual modes."""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import tempfile
import numpy as np
from .config import run_fingerprint
from .storage import atomic_json


def validate_groups(rows, splits, count):
    if sorted(r['episode_index'] for r in rows) != list(range(count)):
        raise ValueError('Paired group inventory is incomplete')
    assignment = {eid: name for name, ids in splits.items() for eid in ids}
    groups = defaultdict(list)
    for row in rows:
        groups[row['pair_group']].append(row)
    for group in groups.values():
        if len(group) != 2 or {r['mode'] for r in group} != {-45,45}:
            raise ValueError('Every trajectory must have exactly two orientation modes')
        if len({assignment[r['episode_index']] for r in group}) != 1:
            raise ValueError('Trajectory pair leaked across dataset splits')
        if group[0]['trajectory'] != group[1]['trajectory'] or group[0]['object'] != group[1]['object']:
            raise ValueError('Paired episodes disagree in trajectory/object')


def grouped_splits(rows, seed):
    objects = defaultdict(lambda: defaultdict(list))
    for row in rows:
        objects[row['object']][row['pair_group']].append(row['episode_index'])
    result = {'train':[], 'validation':[], 'test':[]}
    rng = np.random.default_rng(seed)
    for obj in sorted(objects):
        keys = sorted(objects[obj])
        keys = [keys[i] for i in rng.permutation(len(keys))]
        a,b = int(.8*len(keys)),int(.9*len(keys))
        for name, subset in zip(result,(keys[:a],keys[a:b],keys[b:])):
            for key in subset:result[name].extend(objects[obj][key])
    result = {name:sorted(ids) for name,ids in result.items()}
    validate_groups(rows,result,len(rows))
    return result


def export_paired(runs, destination, allow_smoke=False):
    from .export import export_dom
    runs = [Path(p).resolve() for p in runs]
    provenances = [json.loads((p/'provenance.json').read_text()) for p in runs]
    reports = [json.loads((p/'report.json').read_text()) for p in runs]
    assert all(p['orientation_policy']=='paired_fixed_modes' for p in provenances)
    assert all(r['complete'] and r['balanced'] for r in reports)
    reference = provenances[0]
    for p in provenances[1:]:
        for k in ('robot_position','robot_quaternion','scene_usd_sha256'):
            assert p['layout'][k] == reference['layout'][k],k
        for k in ('camera','controller','trajectory'):
            assert p['config'][k] == reference['config'][k],k
        for k in ('purpose','seconds','fps','physics_hz','future_frames'):
            assert p['config']['collection'][k] == reference['config']['collection'][k],k
        assert p['software'] == reference['software'], 'Collection code/runtime changed between object runs'
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.paired-export-',dir=destination.parent) as temp:
        staging = Path(temp)
        (staging/'episodes').mkdir()
        provenance = copy.deepcopy(reference)
        provenance['source_runs'] = [{'path':str(p),'provenance':prov,'report':report} for p,prov,report in zip(runs,provenances,reports)]
        provenance['config']['collection']['instruction'] = 'Hover above the moving object while maintaining either a +45 or -45 degree tilt.'
        eid = 0
        for run in runs:
            for pair in sorted((run/'episodes').glob('episode_*')):
                for mode in ('minus45','plus45'):
                    (staging/f'episodes/episode_{eid:06d}').symlink_to(pair/mode,target_is_directory=True)
                    eid += 1
        provenance['config']['collection']['episodes'] = eid
        provenance['config_sha256'] = run_fingerprint(provenance['config'])
        atomic_json(staging/'provenance.json',provenance)
        atomic_json(staging/'report.json',{'complete':True,'balanced':True,'episodes':eid,
                    'orientation_policy':'paired_fixed_modes','mode_order':[-45,45], 'mode_frequency':[.5,.5],
                    'object_reports':reports,'sampling_note':'Both modes pass unchanged quality gates; all attempted pairs and failures remain in the raw runs.'})
        (staging/'asset_manifest.json').write_bytes((runs[0]/'asset_manifest.json').read_bytes())
        result = export_dom(staging,destination,allow_smoke=allow_smoke)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-smoke',action='store_true')
    args=parser.parse_args()
    print(json.dumps(export_paired(args.runs,args.output,args.allow_smoke),indent=2))
