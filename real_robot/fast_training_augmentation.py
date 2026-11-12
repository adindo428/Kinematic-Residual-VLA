"""Batched form of the existing photometric recipe, with no per-example GPU sync."""
import torch
from real_robot.training_augmentation import augment_training_batch


def photometric_factors(images,factors):
    """Match torchvision brightness -> contrast -> saturation and clipping."""
    if images.shape[-3]!=3 or factors.shape!=(images.shape[0],3):raise ValueError('Expected RGB images and Bx3 factors')
    shape=(images.shape[0],)+(1,)*(images.ndim-1)
    brightness,contrast,saturation=[factors[:,i].reshape(shape) for i in range(3)]
    image=(images*brightness).clamp(0,1)
    def gray(value):
        return .2989*value[...,0:1,:,:]+.587*value[...,1:2,:,:]+.114*value[...,2:3,:,:]
    mean=gray(image).mean(dim=(-3,-2,-1),keepdim=True)
    image=(contrast*image+(1-contrast)*mean).clamp(0,1)
    image=(saturation*image+(1-saturation)*gray(image)).clamp(0,1)
    return image


def augment_family_batch(batch,recipe,training=True):
    if not training or recipe.get('proprioception',{}).get('enabled',False):
        return augment_training_batch(batch,recipe,training=training)
    photo=recipe.get('photometric',{})
    if not photo.get('enabled',False):return batch,{}
    count=len(batch['observation.state']);device=batch['observation.state'].device
    bounds=torch.tensor([photo.get('brightness',[.9,1.1]),photo.get('contrast',[.9,1.1]),photo.get('saturation',[.95,1.05])],device=device)
    factors=torch.rand(count,3,device=device)*(bounds[:,1]-bounds[:,0])+bounds[:,0]
    result=dict(batch)
    for key,value in batch.items():
        if key.startswith('observation.images.') and torch.is_tensor(value) and value.ndim>=4 and value.shape[-3]==3:
            result[key]=photometric_factors(value,factors)
    return result,{'photometric_samples':count}
