#!/usr/bin/env python3
"""Strict-load and finite-action validation for a native velocity-only policy."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser()
    for k in ['source','checkpoint','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.source))
    import torch
    from safetensors.torch import load_model
    from utils.helpers import get_policy_cfg
    from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy
    torch.set_num_threads(2);torch.manual_seed(42)
    cfg=get_policy_cfg(cfg_file=a.checkpoint/'config.json')
    assert cfg.condition_on_object_velocity and not cfg.enable_residual_flow
    assert cfg.chunk_size==cfg.n_action_steps==20 and cfg.n_obs_steps==2 and cfg.use_delta_action
    cfg.enable_streaming=False
    model=DynamicVLAPolicy.from_pretrained(a.checkpoint,config=cfg).cuda().eval()
    load_model(model,str(a.checkpoint/'model.safetensors'),strict=True)
    batch={k:torch.zeros(1,2,*f.shape,device='cuda') for k,f in cfg.input_features.items()}
    batch['task']=['Pick up the moving object.']
    batch['observation.environment_state']=torch.zeros(1,2,9,device='cuda');batch['observation.environment_state'][...,6]=.1
    with torch.inference_mode():actions=model.predict_action_chunk(batch);torch.cuda.synchronize()
    assert tuple(actions.shape)==(1,20,7) and torch.isfinite(actions).all()
    a.output.write_text(json.dumps({'strict_checkpoint_loaded':True,'velocity_conditioning':True,'residual_flow':False,
        'finite_action_shape':list(actions.shape),'gpu':torch.cuda.get_device_name(0)},indent=2)+'\n')


if __name__=='__main__':main()
