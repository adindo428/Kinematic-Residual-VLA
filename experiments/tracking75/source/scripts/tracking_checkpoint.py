"""Strict loading of validation-selected tracking weights, independent of training code."""
import dataclasses
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tracking_online_common import sha256


def checkpoint_files(root, require_complete=False):
    root=Path(root).resolve()
    config=root/'config.json'
    weights=root/'model.best_validation_action_mse.safetensors'
    if not config.is_file() or not weights.is_file():
        raise FileNotFoundError(f'Expected config.json and validation-selected weights in {root}')
    payload=json.loads(config.read_text())
    if payload.get('delta_timestamps',{}).get('observation') != [-2,0]:
        raise ValueError(f'{root}: expected observation offsets [-2, 0]')
    if require_complete:
        ready=root/'training_ready.json'
        if not ready.is_file():
            raise ValueError(f'{root}: training_ready.json is missing; the 75-epoch run is not verified complete')
        marker=json.loads(ready.read_text())
        if marker.get('status')!='ready' or marker.get('trained_epochs')!=75:
            raise ValueError(f'{root}: the 75-epoch run is not ready')
        if marker['weights_sha256']!=sha256(weights) or marker['config_sha256']!=sha256(config):
            raise ValueError(f'{root}: checkpoint changed after training verification')
    return root,config,weights,payload


def load_checkpoint(root, device='cuda'):
    import draccus
    import safetensors.torch
    import torch
    from policies.dynamicvla.configuration_dynamicvla import DynamicVLAConfig
    from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy
    root,config,weights,payload=checkpoint_files(root)
    fields={f.name for f in dataclasses.fields(DynamicVLAConfig)}
    cfg=draccus.decode(DynamicVLAConfig,{k:v for k,v in payload.items() if k in fields})
    cfg.device=device
    cfg.enable_streaming=False
    policy=DynamicVLAPolicy(cfg)
    # Load normalizers as well as model parameters; no pretrained fallback.
    policy.load_state_dict(safetensors.torch.load_file(str(weights)),strict=True)
    for name,value in policy.state_dict().items():
        if ('normalize' in name) and not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite checkpoint normalizer: {name}')
    return policy.to(device).eval()
