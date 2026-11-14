#!/usr/bin/env python3
"""Check frozen scene physical feasibility with the original collection expert."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
WORK = REPO/'scratch/dom45'


def check_scene(raw, scene_path, sim, evaluate, settings, sizes):
    import torch
    metadata = {}
    for name, value in raw['scene'].items():
        if not name.startswith(('object','container')) or not isinstance(value,dict): continue
        path = Path(value.get('spawn',{}).get('usd_path',''))
        if not path.is_file(): continue
        if str(path) not in sizes: sizes[str(path)] = sim._get_object_size(str(path))
        metadata[path.name] = {'size':sizes[str(path)]}
    cfg = evaluate._get_env_cfg(copy.deepcopy(raw),1,str(WORK/'assets/full-scene-release'),str(WORK/'assets/objects'),.015,'cuda:0',False)
    cfg.episode_length_s = raw['episode_length_s']
    for name in list(vars(cfg.scene)):
        if name.endswith('_cam'): delattr(cfg.scene,name)
    task = raw['instruction']['task']
    objects = list(raw['terminations']['objects_placed']['params']['objects']) if task=='long-horizon' else ['object']
    object_sizes = {name:sim.get_object_size(Path(raw['scene'][name]['spawn']['usd_path']).name,metadata,device='cuda:0') for name in objects}
    tags = {k:v for k,v in raw['instruction'].items() if k!='task'}
    # Reuse the frozen config and the exact collection controller, without
    # rendering cameras. Physics, reset state and 0.015 m collection goal stay.
    captured = []
    convert = sim.get_env_states
    def capture(states, n_envs=1):
        converted = convert(states,n_envs)
        captured[:] = converted
        return converted
    with patch.object(sim,'get_env_cfg',return_value=(cfg,tags,objects,object_sizes)), patch.object(sim,'get_env_states',side_effect=capture):
        _, _, states = sim.simulate(settings,task,'franka',str(WORK/'assets/full-scene-release'),metadata,raw['seed'])
    success = bool(states)
    stopped = bool(sim.is_object_stopped(raw['scene'],states[0]['object_vel'])) if success else None
    direction_changed = bool(sim.is_object_direction_changed(raw['scene'],states[0]['object_vel'])) if success else None
    return {'case':scene_path.stem,'expert_completed':success,'early_target_stop':stopped,
            'direction_changed':direction_changed,
            'collection_eligible':success and not stopped,'camera_rendering':False,'goal_tolerance_m':.015,
            'steps':len(captured[0]['action']) if captured else None,
            'first_object_position':captured[0]['object_pos'][0].tolist() if captured else None,
            'first_object_velocity':captured[0]['object_vel'][0].tolist() if captured else None}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=WORK/'evaluations/prompt-matched20k-20261102')
    parser.add_argument('--case');args=parser.parse_args()
    from isaaclab.app import AppLauncher
    launcher=AppLauncher(headless=True,enable_cameras=False,device='cuda:0',kit_args=os.environ['KRVLA_KIT_ARGS'])
    import yaml
    source=WORK/'official/source';sys.path[:0]=[str(source),str(source/'simulations')]
    import simulations.simulate as sim
    import simulations.evaluate as evaluate
    settings=yaml.safe_load((source/'simulations/configs/sim_cfg.yaml').read_text())
    settings.update(device='cuda:0',num_envs=1,disable_fabric=False,enable_cameras=False,path_tracing=False,disable_sm=False,debug=False)
    paths=[args.root/'configs'/(args.case+'.json')] if args.case else sorted((args.root/'configs').glob('*.json'))
    records=[];sizes={};out=args.root/'expert_checks';out.mkdir(exist_ok=True)
    try:
        for path in paths:
            raw=json.loads(path.read_text())
            record=check_scene(raw,path,sim,evaluate,settings,sizes)
            records.append(record);(out/(path.stem+'.json')).write_text(json.dumps(record,indent=2)+'\n')
            print('EXPERT',json.dumps(record),flush=True)
        (out/'summary.json').write_text(json.dumps({'checked':len(records),'eligible':sum(r['collection_eligible'] for r in records),'records':records},indent=2)+'\n')
    finally:launcher.app.close()


if __name__=='__main__':main()
