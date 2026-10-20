#!/usr/bin/env python3
"""Add new Coordinate+Velocity trials to retained four-model plots and videos."""
import argparse
import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from prepare_coordinate_velocity_eval import MATCHED_FIELDS, LEGACY_MODELS
from summarize_tracking_online_three import main as summarize, joint_match
from tracking_online_common import write_json, sha256
from tracking_variants import VARIANT_LABELS


def merge(previous, new, output, no_videos=False):
    previous=Path(previous).resolve();new=Path(new).resolve();output=Path(output).resolve()
    old_manifest=json.loads((previous/'manifest.json').read_text())
    new_manifest=json.loads((new/'manifest.json').read_text())
    for field in MATCHED_FIELDS:
        assert old_manifest[field]==new_manifest[field], f'Experimental setting differs: {field}'
    assert list(old_manifest['models'])==list(LEGACY_MODELS)
    assert list(new_manifest['models'])==['coordinate_velocity'], 'Only the new model may have been evaluated'
    completion=json.loads((new/'completion.json').read_text())
    assert completion['completed'] and completion['rollouts']==30
    old_runtime=json.loads((previous/'runtime.json').read_text())
    new_runtime=json.loads((new/'runtime.json').read_text())
    assert old_runtime==new_runtime, 'Simulator software or execution contract differs'
    manifest=copy.deepcopy(old_manifest)
    for key in ('models','model_labels','checkpoint_metadata'):
        manifest[key].update(new_manifest[key])
    manifest['reused_evaluation']=new_manifest['reused_evaluation']
    manifest['rollout_sources']={m:str(previous) for m in LEGACY_MODELS}|{'coordinate_velocity':str(new)}
    output.mkdir(parents=True,exist_ok=False)
    rows=[];source_hashes={}
    for model,source in manifest['rollout_sources'].items():
        source=Path(source)
        records=sorted(source.glob(f'{model}/*/episode_*/metrics.json'))
        assert len(records)==30, f'Expected 30 trials for {model}'
        for path in records:
            row=json.loads(path.read_text());assert row['model']==model
            assert row['requested_seconds']==10
            assert row['checkpoint']['weights_sha256']==manifest['checkpoint_metadata'][model]['weights_sha256']
            destination=output/path.relative_to(source).parent
            destination.mkdir(parents=True)
            # Copy small metric records and link original telemetry/cameras. No inference or simulation here.
            for item in path.parent.iterdir():
                if item.name!='metrics.json': (destination/item.name).symlink_to(item.resolve())
            row['videos']={camera:str((path.parent/Path(video).name).resolve()) for camera,video in row['videos'].items()}
            write_json(destination/'metrics.json',row)
            rows.append(row)
            source_hashes[str(path)]=sha256(path)
    expected=set(itertools.product(list(manifest['models']),manifest['speed_multipliers'],[x['episode'] for x in manifest['episodes']]))
    keys=[(r['model'],r['speed_multiplier'],r['episode']) for r in rows]
    assert len(keys)==150 and len(set(keys))==150 and set(keys)==expected
    lookup={key:row for key,row in zip(keys,rows)}
    for speed in manifest['speed_multipliers']:
        for item in manifest['episodes']:
            ep=item['episode'];new_row=lookup['coordinate_velocity',speed,ep]
            for model in LEGACY_MODELS:
                assert joint_match(lookup[model,speed,ep],new_row), f'Initial state differs: {model}, {speed}x, episode {ep}'
    write_json(output/'manifest.json',manifest);write_json(output/'rollouts.json',rows)
    write_json(output/'runtime.json',new_runtime)
    write_json(output/'completion.json',{'completed':True,'rollouts':150,'new_rollouts':30,'reused_rollouts':120,
        'policy_terminations':sum(r.get('failure_type')=='policy_termination' for r in rows)})
    write_json(output/'provenance.json',{'old_evaluation_job':16741254,'old_results':str(previous),
        'new_results':str(new),'source_metrics_sha256':source_hashes,
        'new_rollouts':30,'reused_rollouts':120,'other_models_rerun':False})
    summarize(SimpleNamespace(output=output,no_videos=no_videos))
    result=json.loads((output/'summary.json').read_text());old_result=json.loads((previous/'summary.json').read_text())
    assert result['complete'] and result['pairing_valid'] and result['recorded_rollouts']==150
    for key,value in old_result['groups'].items():assert result['groups'][key]==value, f'Original result changed: {key}'
    if not no_videos:
        assert len(result['comparison_videos'])==30
        assert all(v['frames']==250 and len(v['model_order'])==5 for v in result['comparison_videos'])
    for path,digest in source_hashes.items():assert sha256(path)==digest
    (output/'README.md').write_text(
        '# Five-method closed-loop tracking comparison\n\n'
        'Adds KR-VLA-Coordinate+Velocity (75 epochs; selected at epoch 57) to the saved four-method evaluation. '
        'Only this new model was evaluated: 30 ten-second trials on the same ten held-out trajectories at 1x, 1.5x and 2x speed. '
        'All 120 old trials and camera recordings are reused from job 16741254.\n\n'
        'The old purple KR-VLA-Coordinate curve is the checkpoint without dedicated neural velocity input; '
        'the new red KR-VLA-Coordinate+Velocity curve uses both neural velocity conditioning and analytic residual displacement.\n\n'
        'tracking_comparison.png adds the fifth method to tracking-error and completion plots. '
        'comparison_videos/ contains 30 five-panel videos; each retains all five labels and the original early-stop card. '
        'Per-trial camera videos and telemetry are linked in each method directory.\n\n'
        'Simulation pauses for inference, matching the previous experiment. Mean position errors use completed trials only; '
        'completion is distinct from strict task success. Current simulator velocity is supplied in robot-base coordinates. '
        'Speed scaling also scales acceleration by its square. No baseline, Field, velocity baseline, or old Coordinate inference was rerun.\n')
    print('VERIFIED: 150 trials, 120 reused, 30 new; old group results unchanged; paired resets match')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',required=True,type=Path);p.add_argument('--new',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--no-videos',action='store_true')
    a=p.parse_args();merge(a.previous,a.new,a.output,a.no_videos)
