"""Restore a native checkpoint's recorded parameter and inference precision."""
import json
from pathlib import Path
import torch


def apply_checkpoint_precision(policy,checkpoint):
    root=Path(checkpoint)
    if (root/'selection.json').exists():
        root=Path(json.loads((root/'selection.json').read_text())['source'])
    path=root/'provenance.json'
    record=json.loads(path.read_text()) if path.exists() else {}
    parameters=record.get('parameter_dtype','native')
    autocast=record.get('autocast_dtype','bfloat16')
    if parameters not in ('native','float32') or autocast not in ('bfloat16','float16'):
        raise ValueError('Unsupported saved checkpoint precision contract')
    if parameters=='float32':policy.float()
    policy.execution_autocast_dtype=getattr(torch,autocast)
    return dict(parameter_dtype=parameters,autocast_dtype=autocast)
