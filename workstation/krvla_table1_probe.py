#!/usr/bin/env python3
"""Strictly load the trained KRVLA+Velocity policy before the simulator pilot."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','checkpoint','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.source))
    import torch
    import safetensors.torch
    from utils.helpers import get_policy_cfg
    from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy
    cfg=get_policy_cfg(cfg_file=args.checkpoint/'config.json')
    if (not cfg.enable_residual_flow or not cfg.condition_on_object_velocity
            or cfg.residual_flow_formulation!='coordinate' or cfg.residual_phase_gate_mode!='constant'
            or cfg.chunk_size!=20 or cfg.n_action_steps!=20 or not cfg.use_delta_action):
        raise ValueError('Checkpoint does not describe the trained KRVLA+Velocity variant')
    cfg.enable_streaming=False
    torch.cuda.reset_peak_memory_stats()
    policy=DynamicVLAPolicy.from_pretrained(args.checkpoint,config=cfg).cuda().eval()
    # Use the same strict loader as training resume; it also understands any
    # tied-parameter aliases omitted by safetensors.save_model.
    safetensors.torch.load_model(policy,str(args.checkpoint/'model.safetensors'),strict=True)
    saved=safetensors.torch.load_file(str(args.checkpoint/'model.safetensors'))
    for key,tensor in saved.items():
        actual=policy.state_dict()[key]
        torch.testing.assert_close(actual.cpu(),tensor,rtol=0,atol=0)
    batch={key:torch.zeros(1,cfg.n_obs_steps,*feature.shape,device='cuda')
           for key,feature in cfg.input_features.items()}
    batch['task']=['Pick up the moving object and place it in the container.']
    # This is an auxiliary analytic input; adding it to input_features would
    # incorrectly change the checkpoint's pretrained normalization modules.
    env=torch.zeros(1,cfg.n_obs_steps,9,device='cuda')
    batch['observation.environment_state']=env
    env[...,6]=.1
    with torch.inference_mode():
        policy.predict_action_chunk(batch)
        torch.cuda.synchronize();start=time.perf_counter()
        actions=policy.predict_action_chunk(batch)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
    if list(actions.shape)!=[1,20,7] or not torch.isfinite(actions).all():
        raise ValueError('Invalid predicted action chunk')
    report={'validated':True,'all_weights_exactly_match_checkpoint':True,'live_velocity_input':True,
            'finite_action_shape':list(actions.shape),'warmed_chunk_time_s':elapsed,
            'gpu':torch.cuda.get_device_name(0),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'checkpoint':str(args.checkpoint),'policy_source':str(args.source)}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
