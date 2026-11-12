"""Physical action losses and deterministic validation for real Cartesian policies.

The auxiliary objective backpropagates through the complete deployment sampler.
It is deliberately separate from the no-grad inference unnormalizer.
"""
import math
import random
from contextlib import contextmanager

import numpy as np
import torch


def horizon_weights(recipe, horizon, device):
    weights = torch.ones(horizon, device=device)
    weights[:recipe.get('early_steps', 4)] = recipe.get('early_weight', 4.)
    return weights


def weighted_mean(values, valid, weights):
    """Equal weight per anchor, with masked, normalized horizon weights."""
    w = valid.to(values.dtype) * weights
    return ((values * w).sum(1) / w.sum(1).clamp_min(1)).mean()


def xyz_rotation_matrix(euler):
    """Extrinsic XYZ, matching scipy Rotation.from_euler('xyz')."""
    x, y, z = euler.unbind(-1)
    cx, cy, cz = x.cos(), y.cos(), z.cos()
    sx, sy, sz = x.sin(), y.sin(), z.sin()
    return torch.stack((cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx,
                        sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx,
                        -sy, cy*sx, cy*cx), -1).reshape(*euler.shape[:-1], 3, 3)


def physical_errors(prediction, target):
    delta = prediction - target
    r1 = xyz_rotation_matrix(prediction[..., 3:6])
    r2 = xyz_rotation_matrix(target[..., 3:6])
    # Chordal SO(3) error: theta^2 locally, with finite gradients at equality.
    rotation = ((r1-r2)**2).sum((-1, -2)) / 2
    return dict(position_mse_m2=delta[..., :3].square().mean(-1),
                rotation_chordal=rotation, gripper_mse=delta[..., 6].square(),
                position_distance_m=delta[..., :3].norm(dim=-1),
                rotation_deg=torch.rad2deg(torch.acos((1-rotation/2).clamp(-1, 1))),
                gripper_mae=delta[..., 6].abs())


def physical_loss(prediction, target, valid, recipe):
    errors = physical_errors(prediction.float(), target.float())
    weights = horizon_weights(recipe, prediction.shape[1], prediction.device)
    terms = dict(position=weighted_mean(errors['position_mse_m2'], valid, weights) / recipe['position_scale_m']**2,
                 rotation=weighted_mean(errors['rotation_chordal'], valid, weights) / math.radians(recipe['rotation_scale_deg'])**2,
                 gripper=weighted_mean(errors['gripper_mse'], valid, weights) / recipe['gripper_scale']**2)
    total = sum(recipe.get(name+'_weight', 1.)*value for name, value in terms.items())
    return total, {name:float(value.detach()) for name,value in terms.items()}


def sampled_absolute_actions(policy, batch, noise=None):
    """Differentiable full sampler, including KRV restoration and physical decode."""
    from utils.real_robot_contract import absolute_pose
    from lerobot.configs.types import NormalizationMode
    cfg = policy.config
    if cfg.adapt_to_pi_aloha or not cfg.use_delta_action or cfg.action_feature.shape[0] != 7:
        raise ValueError('Real action objective requires seven-dimensional Cartesian delta actions')
    state = batch['observation.state'][:, -1:, :].clone()
    batch = policy._prepare_batch(batch)
    images, masks = policy.prepare_images(batch)
    lang, lang_masks = policy.prepare_language(batch)
    state_input = policy.prepare_state(batch)
    # sample_actions restores KRV coordinate drift exactly once.
    actions = policy.model.sample_actions(images, masks, lang, lang_masks, state_input, noise=noise,
        residual_position=policy.prepare_residual_xyz(batch, cfg.residual_position_key, state.device, state.dtype),
        residual_velocity=policy.prepare_residual_velocity(batch, state.device, state.dtype),
        residual_lookahead_position=policy.prepare_residual_xyz(batch, cfg.residual_lookahead_position_key, state.device, state.dtype),
        residual_lookahead_horizon=policy.prepare_residual_horizon(batch, cfg.residual_lookahead_horizon_key, state.device, state.dtype),
        residual_action_std=policy.get_action_normalization_std(state.device, state.dtype),
        residual_phase_gate=policy.prepare_phase_gate(batch, state.device))
    actions = actions[..., :7].float()
    mode = cfg.normalization_mapping[cfg.action_feature.type]
    if mode != NormalizationMode.MEAN_STD:
        raise ValueError('Audited real-data objective requires mean/std action normalization')
    buffer = policy.unnormalize_outputs.buffer_action
    # Unnormalize.forward is decorated no_grad; explicit affine decode keeps gradients.
    actions = actions*buffer['std'] + buffer['mean']
    return absolute_pose(actions, state, cfg.euler_delta_convention)


@contextmanager
def fixed_rng(seed, device):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[device]):
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state); np.random.set_state(numpy_state)


def action_validation(policy, dataset, indices, recipe, batch_size=8):
    """Three seeded samples on fixed anchors; no best-of-N selection."""
    from utils.real_robot_contract import absolute_pose
    from real_robot.training_augmentation import model_inputs
    device = next(policy.parameters()).device
    policy.eval(); sums={}; counts={}
    subset = torch.utils.data.Subset(dataset, indices)
    with fixed_rng(recipe.get('validation_seed', 12345), device), torch.no_grad():
        loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
        for batch in loader:
            ids=batch['episode_index'].tolist(); frames=batch['frame_index'].tolist()
            batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            valid=~batch['action_is_pad']; target=absolute_pose(batch['action'],batch['observation.state'][:,-1:,:],policy.config.euler_delta_convention)
            for seed in recipe.get('validation_noise_seeds',[12345,23456,34567]):
                noise=torch.stack([torch.randn((policy.config.chunk_size,policy.config.max_action_dim),device=device,
                    generator=torch.Generator(device=device).manual_seed(seed+ep*100000+fr)) for ep,fr in zip(ids,frames)])
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    pred=sampled_absolute_actions(policy,model_inputs(batch,policy.config),noise)
                if not torch.isfinite(pred).all():raise RuntimeError('Nonfinite sampled validation actions')
                loss,_=physical_loss(pred,target,valid,recipe)
                sums['selection_score']=sums.get('selection_score',0.)+float(loss)*len(ids)
                counts['selection_score']=counts.get('selection_score',0)+len(ids)
                errors=physical_errors(pred.float(),target.float())
                for scope,mask in [('first',valid & (torch.arange(valid.shape[1],device=device)[None,:]==0)),('full',valid)]:
                    for name,value in errors.items():
                        key=scope+'/'+name
                        sums[key]=sums.get(key,0.)+float(value[mask].sum())
                        counts[key]=counts.get(key,0)+int(mask.sum())
    return {key:sums[key]/counts[key] for key in sums}
