#!/usr/bin/env python3
"""Reuse the previous closed-loop manifest, selecting only the new Coordinate model."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from tracking_checkpoint import checkpoint_files
from tracking_online_common import sha256, write_json
from tracking_variants import VARIANT_LABELS, validate_variant


MATCHED_FIELDS = ('dataset_revision','config','layout','episodes','speed_multipliers','train_max_speed_mps','collector_commit')
LEGACY_MODELS = ('baseline','velocity','residual','coordinate')


def prepare(previous, checkpoint, collector, output):
    previous=Path(previous).resolve()
    original=json.loads((previous/'manifest.json').read_text())
    report=json.loads((previous/'summary.json').read_text())
    assert list(original['models'])==list(LEGACY_MODELS)
    assert report['complete'] and report['pairing_valid'] and report['recorded_rollouts']==120
    assert len(original['episodes'])==10 and original['speed_multipliers']==[1,1.5,2]
    assert original['config']['collection']['seconds']==10
    # Require retained source videos; metadata may still contain paths from before relocation.
    for model in LEGACY_MODELS:
        records=list(previous.glob(f'{model}/*/episode_*/metrics.json'))
        assert len(records)==30, f'Missing original trials for {model}'
        for record in records:
            row=json.loads(record.read_text())
            assert row['model']==model and row['requested_seconds']==10
            assert (record.parent/'initial_state.json').is_file()
            for camera,count in row['video_frames'].items():
                if count: assert (record.parent/f'{camera}.mp4').is_file()
    root,config,weights,payload=checkpoint_files(checkpoint,require_complete=True)
    validate_variant(SimpleNamespace(**payload),'coordinate_velocity')
    assert payload['n_obs_steps']==2 and payload['n_action_steps']==20 and payload['chunk_size']==20
    assert payload['use_delta_action']
    sys.path.insert(0,str(collector))
    from dom_tracking.config import fingerprint,file_hash
    files=Path(collector)/'dom_tracking'
    collector_hash=fingerprint({p.name:file_hash(p) for p in sorted(files.glob('*.py'))})
    runtime=json.loads((previous/'runtime.json').read_text())
    assert collector_hash==runtime['software']['collector_source_sha256'], 'Collector source changed since the previous evaluation'
    for asset in ('scene_usd','object_usd'):
        assert sha256(original['layout'][asset])==original['layout'][asset+'_sha256']
    manifest=copy.deepcopy(original)
    manifest.pop('baseline',None);manifest.pop('residual',None)
    manifest['models']={'coordinate_velocity':str(root)}
    manifest['model_labels']={'coordinate_velocity':VARIANT_LABELS['coordinate_velocity']}
    manifest['checkpoint_metadata']={'coordinate_velocity':{
        'weights_sha256':sha256(weights),'config_sha256':sha256(config),
        'chunk_size':20,'selected_weights':str(weights),'neural_velocity_conditioning':True}}
    manifest['reused_evaluation']={'job_id':16741254,'path':str(previous),
        'manifest_sha256':sha256(previous/'manifest.json'),'runtime_sha256':sha256(previous/'runtime.json'),
        'collector_source_sha256':collector_hash,'models':list(LEGACY_MODELS),
        'new_rollouts_only':30}
    write_json(output,manifest)
    print('Prepared Coordinate+Velocity only: 10 episodes x 3 speeds x 10 seconds; reused prior job 16741254')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',required=True,type=Path);p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--collector',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();prepare(a.previous,a.checkpoint,a.collector,a.output)
