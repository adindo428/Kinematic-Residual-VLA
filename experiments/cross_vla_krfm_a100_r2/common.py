"""Shared, physical-coordinate KRFM and provenance for DOM backbone experiments."""
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def drift(velocity, mean, horizon, offset=0, dt=.04):
    """Only current observed velocity; no future environment labels enter KR.

    Trajectory actions are displacements from the chunk's state reference.
    In VLASH, the reference state rolls forward but vision remains at t, so
    the target object's displacement is evaluated at (offset+j)/25 seconds.
    Rotation and gripper have exactly zero analytic drift. Centering is frozen
    using training episodes; lambda=1. Inputs may be NumPy or torch tensors.
    """
    import torch
    velocity = torch.as_tensor(velocity)
    mean = torch.as_tensor(mean, device=velocity.device, dtype=velocity.dtype)
    steps = torch.arange(horizon, device=velocity.device, dtype=velocity.dtype)
    offsets = torch.as_tensor(offset, device=velocity.device, dtype=velocity.dtype)
    times = (steps + offsets[..., None]) * dt
    result = velocity.new_zeros((*velocity.shape[:-1], horizon, 7))
    result[..., :3] = (velocity - mean)[..., None, :] * times[..., None]
    return result


def coordinates(raw, correction, inverse=False):
    return raw + correction if inverse else raw - correction


def masked_loss(losses, padding, dimensions=7):
    valid = (~padding.bool()).unsqueeze(-1)
    losses = losses[..., :dimensions]
    return (losses * valid).sum() / (valid.sum() * dimensions).clamp_min(1)
