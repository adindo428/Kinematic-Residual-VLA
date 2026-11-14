"""Guided real-time chunking for the pinned native SmolVLA sampler.

Inference-time prefix guidance following Physical Intelligence RTC and LeRobot:
https://huggingface.co/docs/lerobot/rtc
https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/rtc/modeling_rtc.py
No checkpoint parameters or robot-control limits are modified.
"""
from contextlib import contextmanager
import math
from unittest.mock import patch
import numpy as np
import torch


def prefix_weights(delay,end,length,device=None):
    delay=min(max(int(delay),0),length);end=min(max(int(end),delay),length)
    weights=torch.zeros(length,device=device)
    weights[:delay]=1.
    if end>delay:weights[delay:end]=torch.linspace(1.,0.,end-delay+2,device=device)[1:-1]
    return weights


def guided_velocity(x,time,denoise,previous,weights,max_weight=10.):
    """VJP prefix guidance in native noise-at-one flow coordinates."""
    t=float(torch.as_tensor(time).flatten()[0])
    if not 0<t<=1 or not math.isfinite(max_weight) or max_weight<=0:raise ValueError('Invalid RTC guidance time/weight')
    with torch.enable_grad():
        current=x.detach().clone().requires_grad_(True)
        velocity=denoise(current)
        endpoint=current-t*velocity
        error=(previous-endpoint)*weights
        correction=torch.autograd.grad(endpoint,current,error.detach(),retain_graph=False)[0]
    tau=1-t
    gain=max_weight if tau<=1e-8 else min(max_weight,(t/tau)*(t*t+tau*tau)/(t*t))
    result=(velocity-gain*correction).detach()
    if not torch.isfinite(result).all():raise ValueError('Nonfinite RTC-guided velocity')
    return result


def normalize_prefix(model,batch,absolute):
    from utils.real_robot_contract import delta_pose
    values=np.asarray(absolute,dtype=np.float32)
    if values.ndim!=2 or values.shape[1]!=7 or len(values)>20 or not np.isfinite(values).all():
        raise ValueError('RTC prefix must be finite Nx7 physical actions, at most one horizon')
    device=batch['reference_state'].device
    absolute=torch.as_tensor(values,device=device)[None]
    delta=delta_pose(absolute,batch['reference_state'][:,None,:],'wrapped_xyz_v1')
    _,correction=model.prepare(batch)
    delta=delta-correction[:,:len(values)]
    normalized=model.native.affine(delta,'action')
    return torch.nn.functional.pad(normalized,(0,model.native.policy.config.max_action_dim-7)).detach()


@contextmanager
def guide_smolvla(model,batch,absolute_prefix,*,inference_delay=4,execution_horizon=10,max_guidance_weight=10.):
    if model.method!='smolvla':raise ValueError('Guided RTC currently supports the native SmolVLA family')
    if not isinstance(inference_delay,int) or inference_delay<0 or not 1<=execution_horizon<=20:
        raise ValueError('Invalid RTC inference delay or execution horizon')
    if not math.isfinite(max_guidance_weight) or max_guidance_weight<=0:raise ValueError('Invalid RTC maximum guidance weight')
    prefix=normalize_prefix(model,batch,absolute_prefix)
    count=prefix.shape[1]
    diagnostics=dict(enabled=True,mode='guided',prefix_actions=count,inference_delay_steps=inference_delay,
        execution_horizon=execution_horizon,max_guidance_weight=max_guidance_weight,guided_denoise_steps=0,
        action_coordinates='current-source wrapped pose delta, residual correction removed, native normalized')
    if not count:
        yield diagnostics;return
    target=torch.zeros((prefix.shape[0],20,prefix.shape[2]),device=prefix.device,dtype=prefix.dtype);target[:,:count]=prefix
    weights=torch.zeros_like(target);weights[:,:count,:7]=prefix_weights(inference_delay,execution_horizon,count,device=prefix.device)[None,:,None]
    native=model.native.policy.model;original=native.denoise_step
    def denoise(prefix_masks,cache,x,time):
        result=guided_velocity(x,time,lambda current:original(prefix_masks,cache,current,time),target,weights,max_guidance_weight)
        diagnostics['guided_denoise_steps']+=1
        return result
    with patch.object(native,'denoise_step',side_effect=denoise):yield diagnostics
