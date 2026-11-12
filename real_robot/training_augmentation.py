"""Train-only input perturbations; preserve expert absolute Cartesian targets."""
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from third_party.DynamicVLA.utils.real_robot_contract import absolute_pose,delta_pose


def augment_training_batch(batch,recipe,*,training=True):
    """Input is unnormalized, already delta-encoded loader output.

    Images remain camera-consistent: photometric parameters are shared across
    history/views. Proprioceptive bias is one small translation/rotation-vector
    draw per sample, shared across history. It is sensor-noise augmentation,
    not a substitute for physical recovery demonstrations with matching images.
    """
    if not training or not recipe:return batch,{}
    result=dict(batch);state=batch['observation.state'];metrics={}
    photo=recipe.get('photometric',{})
    if photo.get('enabled',False):
        from torchvision.transforms import functional as F
        keys=[k for k,v in batch.items() if k.startswith('observation.images.')
            and torch.is_tensor(v) and v.ndim>=4 and v.shape[-3]==3]
        for k in keys:result[k]=batch[k].clone()
        for i in range(len(state)):
            factors={k:float(torch.empty((),device=state.device).uniform_(*photo.get(k,default)))
                for k,default in [('brightness',[.9,1.1]),('contrast',[.9,1.1]),('saturation',[.95,1.05])]}
            for k in keys:
                image=F.adjust_brightness(result[k][i],factors['brightness'])
                image=F.adjust_contrast(image,factors['contrast'])
                result[k][i]=F.adjust_saturation(image,factors['saturation'])
        metrics['photometric_samples']=len(state)
    noise=recipe.get('proprioception',{})
    if noise.get('enabled',False):
        if state.ndim!=3 or state.shape[-1]!=6:raise ValueError('Expected B x history x six-dimensional EE state')
        probability=float(noise.get('sample_probability',.3));sigma=float(noise.get('translation_std_m',.002))
        angle=float(noise.get('rotation_std_deg',.5));clip=float(noise.get('clip_std',3.))
        if not 0<=probability<=1 or not np.isfinite([sigma,angle,clip]).all() or min(sigma,angle,clip)<0:
            raise ValueError('Invalid proprioception augmentation settings')
        selected=torch.rand(len(state),device=state.device)<probability
        shift=torch.randn((len(state),3),device=state.device).clamp(-clip,clip)*sigma
        rotvec=torch.randn((len(state),3),device=state.device).clamp(-clip,clip)*np.deg2rad(angle)
        shift[~selected]=0;rotvec[~selected]=0
        noisy=state.clone();noisy[:,:,:3]+=shift[:,None,:]
        # Compose physical rotations, then use the same export Euler branch.
        original=state[:,:,3:6].detach().cpu().numpy();bias=rotvec.detach().cpu().numpy()
        rotations=Rotation.from_euler('xyz',original.reshape(-1,3)).as_matrix().reshape(*original.shape[:2],3,3)
        perturbation=Rotation.from_rotvec(bias).as_matrix()
        changed=Rotation.from_matrix((perturbation[:,None]@rotations).reshape(-1,3,3)).as_euler('xyz').reshape(original.shape)
        changed[:,:,[0,2]]%=2*np.pi
        noisy[:,:,3:6]=torch.as_tensor(changed,device=state.device,dtype=state.dtype)
        noisy[~selected]=state[~selected]
        absolute=absolute_pose(batch['action'],state[:,-1:,:],'wrapped_xyz_v1')
        result['action']=delta_pose(absolute,noisy[:,-1:,:],'wrapped_xyz_v1')
        result['action'][~selected]=batch['action'][~selected]
        result['observation.state']=noisy
        metrics['proprioception_samples']=int(selected.sum())
    return result,metrics


def model_inputs(batch,cfg):
    """Plain DOM gets no environment/velocity/residual fields, even if the
    loader uses those fields to enforce the common quality/sample mask.
    """
    batch={k:v for k,v in batch.items() if not k.startswith('dynamic_object.future_')}
    if cfg.condition_on_object_velocity or cfg.enable_residual_flow:return batch
    return {k:v for k,v in batch.items() if k!='observation.environment_state' and not k.startswith('dynamic_object.')}
