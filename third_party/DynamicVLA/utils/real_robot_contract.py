"""Explicit real-robot dataset contracts; legacy DOM checkpoints stay unchanged."""
import json
import math
import hashlib


def base_calibration_status(root, policy):
    if policy.get('base_calibration_validated') is True:
        return 'validated'
    acceptance = policy.get('base_calibration_acceptance') or {}
    camera_path = root/'meta/camera.jsonl'
    if (acceptance.get('schema') == 'krvla_operator_transform_acceptance_v1'
            and acceptance.get('operator_accepted') is True
            and camera_path.exists()
            and acceptance.get('camera_metadata_sha256') == hashlib.sha256(camera_path.read_bytes()).hexdigest()):
        return 'operator_accepted_existing_transform'
    raise ValueError('Object/robot base calibration needs validation or explicit operator acceptance bound to this dataset')


def delta_pose(action, state, convention='legacy'):
    result = action.clone()
    result[..., :-1] -= state[..., :action.shape[-1]-1]
    if convention == 'wrapped_xyz_v1':
        if action.shape[-1] != 7:
            raise ValueError('wrapped_xyz_v1 requires Cartesian XYZ/Euler/gripper actions')
        result[..., 3:6] = (result[..., 3:6] + math.pi) % (2*math.pi) - math.pi
    elif convention != 'legacy':
        raise ValueError('Unknown Euler delta convention: '+convention)
    return result


def absolute_pose(delta, state, convention='legacy'):
    result = delta.clone()
    result[..., :-1] += state[..., :delta.shape[-1]-1]
    if convention == 'wrapped_xyz_v1':
        if delta.shape[-1] != 7:
            raise ValueError('wrapped_xyz_v1 requires seven action dimensions')
        # Export uses X/Z in [0, 2pi), Y in [-pi, pi). Rotation-equivalent
        # representations are preserved; this is componentwise Euler wrapping.
        result[..., 3] %= 2*math.pi
        result[..., 4] = (result[..., 4]+math.pi) % (2*math.pi)-math.pi
        result[..., 5] %= 2*math.pi
    elif convention != 'legacy':
        raise ValueError('Unknown Euler delta convention: '+convention)
    return result


def explicit_split(root, split, total):
    path = root/'meta/splits.json'
    if not path.exists():
        return None
    splits = json.loads(path.read_text())
    if sorted(ep for ids in splits.values() for ep in ids) != list(range(total)):
        raise ValueError('Dataset split file must cover each episode exactly once')
    if not splits.get(split):
        raise ValueError(f'Dataset has no {split} episodes; create an explicit held-out split')
    return splits[split]


def usable_anchor_indices(episodes, lengths, usable_frames, offsets):
    """Filter anchors without compressing the original physical frame timeline."""
    result, start = [], 0
    for ep, length in zip(episodes, lengths):
        valid = set(usable_frames[str(ep)])
        result.extend(start+i for i in range(length)
                      if all(max(0,min(length-1,i+offset)) in valid for offset in offsets))
        start += length
    if not result:
        raise ValueError('No usable tracking anchors for the requested observation history')
    return result


def velocity_input(velocity, *, held_confirmed=False, valid=False):
    """Same live/offline rule: neutral input only for a confirmed held object."""
    if held_confirmed:
        return velocity * 0
    if not valid:
        raise ValueError('Current pre-grasp velocity unavailable; zero is not a stationary measurement')
    return velocity


def checked_velocity(value, batch, require_quality=False):
    """Validate current inputs and neutralize only positively confirmed holding.

    An unrequired velocity means a confirmed held phase, never merely lost
    tracking. These flags must come from reviewed offline intervals or live
    grasp confirmation using wrist evidence and measured telemetry.
    """
    import torch
    required = batch.get('dynamic_object.velocity_required')
    valid = batch.get('dynamic_object.velocity_valid')
    if required is None or valid is None:
        if require_quality:
            raise ValueError('Real-robot policies require velocity_required and velocity_valid quality flags')
        return value
    masks = []
    for mask in (required, valid):
        mask = torch.as_tensor(mask, device=value.device, dtype=torch.bool)
        if mask.ndim == 2:
            mask = mask[:, -1]
        if mask.shape != value.shape[:1]:
            raise ValueError('Velocity quality flags must be [batch] or [batch, observation_history]')
        masks.append(mask)
    required, valid = masks
    if (required & ~valid).any():
        raise ValueError('Pre-grasp object velocity is unavailable; missing tracking cannot default to zero')
    return torch.where(required[:,None], value, torch.zeros_like(value))
