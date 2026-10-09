#!/usr/bin/env python3
"""Copy a historical tracking recipe, changing only workstation paths/logging."""
import argparse
from pathlib import Path
import yaml

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--recipe',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
p.add_argument('--run-root',required=True,type=Path);p.add_argument('--normalization-reference',type=Path)
p.add_argument('--initial-checkpoint',type=Path);p.add_argument('--wandb',action='store_true')
a=p.parse_args();cfg=yaml.safe_load(a.recipe.read_text())
if a.output.exists():raise FileExistsError(a.output)
cfg['DIR']['OUTPUT']=str(a.run_root.expanduser().resolve())
cfg['WANDB']['ENABLED']=a.wandb
if not a.wandb:cfg['WANDB']['MODE']='disabled'
cfg['DATASET']['EPISODE_SPLIT_FILE']=str(Path(__file__).resolve().parents[1]/'configs/dom_franka_tracking_split.json')
if cfg['DATASET'].get('NORMALIZATION_REFERENCE_CHECKPOINT'):
    if not a.normalization_reference:p.error('--normalization-reference is required for this matched recipe')
    cfg['DATASET']['NORMALIZATION_REFERENCE_CHECKPOINT']=str(a.normalization_reference.expanduser().resolve())
if a.initial_checkpoint:cfg['POLICY']['CHECKPOINT']=str(a.initial_checkpoint.expanduser().resolve())
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(yaml.safe_dump(cfg,sort_keys=False))
print(a.output.resolve())
