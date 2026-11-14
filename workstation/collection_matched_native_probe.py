#!/usr/bin/env python3
"""Strict load and finite-action probe for the native DOM/KRV evaluation client."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--kr', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    import torch
    from safetensors.torch import load_model
    from utils.helpers import get_policy_cfg
    from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = get_policy_cfg(cfg_file=args.checkpoint / 'config.json')
    assert cfg.chunk_size == cfg.n_action_steps == 20 and cfg.use_delta_action
    assert bool(getattr(cfg, 'enable_residual_flow', False)) == args.kr
    assert bool(getattr(cfg, 'condition_on_object_velocity', False)) == args.kr
    cfg.enable_streaming = False
    policy = DynamicVLAPolicy.from_pretrained(args.checkpoint, config=cfg).cuda().eval()
    load_model(policy, str(args.checkpoint / 'model.safetensors'), strict=True)
    batch = {key: torch.zeros(1, cfg.n_obs_steps, *feature.shape, device='cuda') for key, feature in cfg.input_features.items()}
    batch['task'] = ['Pick up the moving object.']
    batch['observation.environment_state'] = torch.zeros(1, cfg.n_obs_steps, 9, device='cuda')
    batch['observation.environment_state'][..., 6] = .1
    with torch.inference_mode():
        policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        start = time.perf_counter()
        actions = policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
    assert tuple(actions.shape) == (1, 20, 7) and torch.isfinite(actions).all()
    args.output.write_text(json.dumps({'strict_checkpoint_loaded': True, 'kr': args.kr,
        'finite_action_shape': list(actions.shape), 'warmed_chunk_time_s': time.perf_counter()-start,
        'gpu': torch.cuda.get_device_name(0)}, indent=2) + '\n')


if __name__ == '__main__':
    main()
