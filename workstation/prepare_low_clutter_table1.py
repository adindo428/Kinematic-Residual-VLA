#!/usr/bin/env python3
"""Freeze all 90 released Table 1 cases with at most two task objects."""
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys

REPO=Path(__file__).resolve().parents[1];WORK=REPO/'scratch/dom45'
ROOT=WORK/'evaluations/low-clutter-full-table1-20261102'
SOURCE=WORK/'official/source'
DIMS=['CR','DA','LS','VU','SR','MP','VG','MG','DR']


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def object_names(cfg):
 return sorted(k for k,v in cfg['scene'].items() if k.startswith('object') and isinstance(v,dict) and v.get('class_type')=='isaaclab.assets.rigid_object.rigid_object:RigidObject')
def remove_term_references(value,removed):
 if isinstance(value,dict):return {k:remove_term_references(v,removed) for k,v in value.items() if k not in removed}
 if isinstance(value,list):return [remove_term_references(v,removed) for v in value if not (isinstance(v,str) and v in removed)]
 return value


def main():
 from isaaclab.app import AppLauncher
 launcher=AppLauncher(headless=True,enable_cameras=False,device='cuda:0',kit_args=os.environ['KRVLA_KIT_ARGS'])
 import numpy as np
 import yaml
 sys.path[:0]=[str(SOURCE),str(SOURCE/'simulations')]
 import simulations.simulate as sim
 from simulations import helpers
 original=read(WORK/'results/dom-paper45-a6000-20261010-02/protocol.json')
 tags=read(WORK/'assets/objects/metadata.json');settings=yaml.safe_load((SOURCE/'simulations/configs/sim_cfg.yaml').read_text())
 generator=WORK/'training/krv100-fresh20k-h200-20261029/source/utils/instruction_generator.py'
 spec=importlib.util.spec_from_file_location('frozen_table1_instructions',generator);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 sizes={};tests=[];changes=[]
 try:
  for test in original['tests']:
   src=read(test['path']);cfg=copy.deepcopy(src);names=object_names(cfg);removed=set(names[2:])
   for name in removed:del cfg['scene'][name]
   if removed:
    cfg['terminations']=remove_term_references(cfg['terminations'],removed)
    if cfg['instruction']['task']!='long-horizon':
     states={'objects':[],'containers':[]}
     for name,value in cfg['scene'].items():
      if not isinstance(value,dict) or not name.startswith(('object','container')) or 'init_state' not in value:continue
      asset=Path(value['spawn']['usd_path']);path=WORK/'assets/objects'/asset.parent.name/asset.name
      if str(path) not in sizes:sizes[str(path)]=sim._get_object_size(str(path))
      row={'category':asset.parent.name,'size':sizes[str(path)],'tags':copy.deepcopy(tags.get(asset.name,{}).get('tags',[asset.parent.name])),
       'pos':np.array(value['init_state']['pos']),'quat':np.array(value['init_state']['rot']),'lin_vel':np.array(value['init_state']['lin_vel'])}
      states['containers' if name.startswith('container') else 'objects'].append(row)
     robot=cfg['scene']['robot']['init_state'];pose={'pos':np.array(robot['pos']),'quat':np.array(robot['rot'])}
     for kind in ['objects','containers']:
      aliases=helpers.get_object_tags(kind,states[kind],pose,['VELOCITY'] if kind=='containers' else None,settings['scene']['objects']['tag_thresholds'])
      assert aliases,(test['name'],kind)
      cfg['instruction'][kind]=sorted(set(aliases))
   # All physical settings of retained entities, all houses, cameras, forces,
   # speeds and episode horizons remain exactly as in the source release.
   assert 1<=len(object_names(cfg))<=2
   for k,v in cfg['scene'].items():assert v==src['scene'][k]
   for k,v in cfg.items():
    if k not in ['scene','instruction','terminations']:assert v==src[k]
   if cfg['instruction']['task']=='long-horizon':
    assert len(cfg['terminations']['objects_placed']['params']['objects'])==2
   if not removed:assert cfg==src
   prompts=[]
   for trial in range(20):
    seed=int(hashlib.sha256(f'low-clutter90:42:{test["name"]}:{trial}'.encode()).hexdigest()[:8],16)
    state=random.getstate();random.seed(seed)
    try:prompts.append(module.InstructionGenerator.generate_instruction(cfg['instruction']))
    finally:random.setstate(state)
   cfg.update(evaluation_case_id=test['name'],evaluation_prompts=prompts)
   destination=ROOT/'configs'/(test['name']+'.json')
   if destination.exists():assert read(destination)==cfg,'Frozen suite changed'
   else:write(destination,cfg)
   entry={**test,'path':str(destination),'config_sha256':sha(destination),'source_config':test['path'],
    'source_sha256':test['config_sha256'],'original_task_objects':len(names),'task_objects':len(object_names(cfg)),
    'removed_objects':sorted(removed),'condition':'released_table1_low_clutter','modified_physics':bool(removed)}
   tests.append(entry)
   changes.append({'name':test['name'],'dimension':test['dimension'],'removed_objects':sorted(removed),
    'prompts_regenerated_for_retained_scene':bool(removed and cfg['instruction']['task']!='long-horizon'),
    'termination_targets_trimmed':bool(removed and cfg['instruction']['task']=='long-horizon')})
  assert len(tests)==90 and Counter(t['dimension'] for t in tests)==Counter(dict.fromkeys(DIMS,10))
  assert sum(bool(t['removed_objects']) for t in tests)==18
  for dim in DIMS:
   p=ROOT/'test_lists'/(dim+'.txt');p.parent.mkdir(parents=True,exist_ok=True)
   p.write_text(''.join(t['path']+'\n' for t in tests if t['dimension']==dim))
  # Three representative pilots: unchanged placement, capped distractor case,
  # and capped long-horizon case. Selected by condition, never success.
  pilot=[next(t for t in tests if t['dimension']=='CR'),next(t for t in tests if t['dimension']=='DA' and t['removed_objects']),next(t for t in tests if t['dimension']=='LS' and t['removed_objects'])]
  (ROOT/'test_lists/pilot.txt').write_text(''.join(t['path']+'\n' for t in pilot))
  selection={'source_protocol':str(WORK/'results/dom-paper45-a6000-20261010-02/protocol.json'),
   'source_protocol_sha256':sha(WORK/'results/dom-paper45-a6000-20261010-02/protocol.json'),'object_cap':2,'containers_excluded_from_cap':True,
   'retention_rule':'Keep object and the first named distractor; remove subsequent task objects. Preserve all destination containers.',
   'all_cases_included':True,'cases_chosen_using_policy_results':False,'repeats':20,'scored_trials_per_checkpoint':1800,
   'changes':changes,'method_pairs':[['dom100_best','krv100_best'],['dom100_latest','krv100_latest'],['smolvla_baseline','smolvla_kr']],
   'posthoc':'After both complete: all-trial success transitions; time/path comparisons conditioned on both succeeding. No outcome-selected episodes replace aggregate scores.'}
  write(ROOT/'selection.json',selection)
  files=[Path(__file__),generator,SOURCE/'simulations/helpers.py',SOURCE/'simulations/configs/sim_cfg.yaml']
  write(ROOT/'protocol.json',{'benchmark':'Low-clutter full Table 1 variant','dimension_order':DIMS,'tests':tests,'repeats':20,
   'scored_trials':1800,'pilot_trials':3,'selection_sha256':sha(ROOT/'selection.json'),
   'asset_sha256':original['asset_sha256'],'files_sha256':{str(p):sha(p) for p in files},
   'scoring':'Original placement/all-objects completion, upstream tolerance 0.07 m; no-action trials invalidate a phase.',
   'limitations':['18 object-capped configurations differ from the original benchmark. 72 retain the entire original physical configuration.',
    'All nine conditions include intentional generalization/disturbance tests; this suite is not labeled IID.',
    'Long horizon remains two task objects plus a container. Container distractors are retained.',
    'Repeated resets share original seeds; fixed prompt schedules are matched across methods, not model noise.',
    'DOM/KRV checkpoints differ in neural velocity conditioning and residualization; that pair does not isolate KR alone. SmolVLA baseline/KR has the same neural inputs.']})
  write(ROOT/'preparation_complete.json',{'complete':True,'cases':90,'modified_cases':18,'unchanged_physical_cases':72,'scored_trials_per_checkpoint':1800})
  print('PREPARED 90 cases: 72 physical configurations unchanged, 18 capped; 1800 trials/checkpoint',flush=True)
 finally:launcher.app.close()


if __name__=='__main__':main()
