#!/usr/bin/env python3
"""Append the explicitly requested historical checkpoint, preserving primary eligibility."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import run_low_clutter_table1 as original


def main():
 root=original.ROOT;out=root/'historical_oldKRV100';out.mkdir(exist_ok=True)
 catalog=original.read(root/'catalog.json')
 if 'old_krv100' in catalog['models']:
  print('oldKRV100 already registered');return
 original.validate(catalog)
 previous=original.WORK/'results/krv100-best-epoch003-task-table1-20261025'
 old=original.read(previous/'protocol.json');checkpoint=Path(old['checkpoint']);weights=checkpoint/'model.safetensors'
 actual=original.sha(weights);assert actual==old['checkpoint_weights_sha256']
 config_sha=original.sha(checkpoint/'config.json');assert config_sha==old['checkpoint_config_sha256']
 cfg=original.read(checkpoint/'config.json');assert cfg['enable_residual_flow'] and cfg['condition_on_object_velocity']
 source=old['policy_source'];assert source==catalog['models']['dom100_best']['source']
 policy=original.read(original.WORK/'evaluations/training_history_policy.json')
 assert policy['models']['krv100_old']['prior_full_data_epochs']==2
 stat=weights.stat()
 entry={'label':'oldKRV100','client_kind':'native','kr':True,'source':source,'checkpoint':str(checkpoint),
  'weight_path':str(weights),'weight_bytes':stat.st_size,'weight_mtime_ns':stat.st_mtime_ns,
  'checkpoint_sha256':actual,'config_sha256':config_sha,'selection':'historical global epoch-3 checkpoint, explicitly requested',
  'generation':old['checkpoint_revision'],'progress':{'epoch':3,'updates':137976},
  'historical_user_exception':True,'prior_full_data_epochs':2.0,'primary_20k_cohort':False}
 request={'authorized_utc':datetime.now(timezone.utc).isoformat(),'explicit_user_request':True,
  'model':'old_krv100','display_label':'oldKRV100','checkpoint_sha256':actual,'config_sha256':config_sha,
  'checkpoint':str(checkpoint),'prior_full_data_epochs':2.0,'primary_20k_cohort':False,
  'scope':'This one historical checkpoint on the frozen low-clutter 90x20 suite; no other training-history exclusions relaxed.',
  'queued_after_existing_models':list(catalog['order']),'scored_trials':1800,'excluded_pilot_trials':3}
 original.write(root/'historical_checkpoint_request.json',request)
 shutil.copy2(root/'catalog.json',out/'catalog.before-extension.json')
 target=root/'results/old_krv100'
 shutil.copytree(root/'test_lists',target/'test_lists',dirs_exist_ok=True)
 original.write(target/'protocol.json',{**original.read(root/'protocol.json'),'checkpoint':entry,
  'historical_comparison':True,'training_history_note':'Two prior full-data epochs; outside the primary 20k comparison.'})
 original.write(target/'suite_status.json',{'state':'queued','scored_trials_target':1800,'historical_comparison':True})
 catalog['primary_order']=list(catalog['order']);catalog['historical_order']=['old_krv100']
 catalog['order']=[*catalog['order'],'old_krv100'];catalog['models']['old_krv100']=entry
 original.write(root/'catalog.json',catalog)
 original.validate(catalog)
 print(json.dumps({'registered':True,'label':'oldKRV100','weight_sha256':actual,'scored_trials':1800,
  'queued_total_trials':1800*len(catalog['order']),'primary_total_trials':1800*len(catalog['primary_order'])},indent=2))


if __name__=='__main__':main()
