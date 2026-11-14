#!/usr/bin/env python3
"""Prepare unqueued velocity-only pairs of an existing frozen IID protocol."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    protocol=json.loads((root/'protocol.json').read_text())
    out=root/'velocity_shift_prepared_only';out.mkdir(exist_ok=True);entries=[]
    for test in protocol['tests']:
        assert test['condition']=='approx_iid'
        original=json.loads(Path(test['path']).read_text());shift=copy.deepcopy(original);changed=[]
        for name,obj in shift['scene'].items():
            if not name.startswith('object') or not isinstance(obj,dict) or 'init_state' not in obj:continue
            velocity=obj['init_state']['lin_vel'];speed=math.sqrt(sum(x*x for x in velocity))
            if speed>1e-8:
                obj['init_state']['lin_vel']=[x*(speed+.8)/speed for x in velocity]
                changed.append('scene.'+name+'.init_state.lin_vel')
        assert changed
        shift['evaluation_case_id']=test['name']+'_velocity_shift'
        neutral=copy.deepcopy(shift);neutral['evaluation_case_id']=original['evaluation_case_id']
        for key in changed:
            name=key.split('.')[1]
            neutral['scene'][name]['init_state']['lin_vel']=original['scene'][name]['init_state']['lin_vel']
        assert neutral==original, 'Nonvelocity scenario change'
        path=out/'configs'/(shift['evaluation_case_id']+'.json');path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(shift,indent=2)+'\n')
        entries.append({'name':shift['evaluation_case_id'],'path':str(path),'paired_iid_case':test['name'],
            'paired_iid_config_sha256':test['config_sha256'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'changed_physical_fields':changed,'prompts_identical':True,'reset_seed_identical':True})
    (out/'manifest.json').write_text(json.dumps({'state':'prepared_not_queued','executed_trials':0,
        'user_instruction':'100 IID trials now; prepare velocity-only OOD but do not run it',
        'speed_transform':'Add 0.8 m/s to every moving object speed, preserving directions and moving-object speed differences/ranks; static objects stay static.',
        'claim_limit':'Outside nominal 0.15-0.75 m/s collection range; no claim of exclusion from all fine-tuning frames or released-model pretraining.',
        'cases':entries},indent=2)+'\n')
    print('Prepared',len(entries),'velocity-only pairs; no simulation launched')


if __name__=='__main__':main()
