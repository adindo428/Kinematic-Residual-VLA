"""Training-only utilities around the released, strictly loaded native policies."""
import ast
from contextlib import contextmanager,nullcontext
import hashlib
import mmap
import inspect
import json
from pathlib import Path
import sys
import textwrap
import types
from unittest.mock import patch

import torch
from torch import nn
from velocity_conditioning import velocity_prompt_batch, UntruncatedTokenizer


def signature(model):
    rows=[(n,list(p.shape)) for n,p in model.named_parameters(remove_duplicate=False)]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


@contextmanager
def sampling_cache_context(model):
    """HF checkpointing disables KV caches in training mode; sampling needs them.

    Flow training retains activation checkpointing. Only the full-sampler forward
    temporarily disables these flags, preserving its inference conditioning and
    the action-expert gradient graph. No train/eval modes or parameters change.
    """
    previous=[]
    for module in model.modules():
        for name in ['gradient_checkpointing','gradient_checkpointing_enabled']:
            if getattr(module,name,False) is True:
                previous.append((module,name,True));setattr(module,name,False)
    try:yield
    finally:
        for module,name,value in previous:setattr(module,name,value)


def enable_sampler_gradients(model):
    """Remove inference-only no-grad wrappers; leave neural layers/parameters intact."""
    for name in ['sample_actions','denoise_step']:
        method=getattr(model,name,None)
        if method is not None and hasattr(method,'__wrapped__'):
            setattr(model,name,types.MethodType(method.__wrapped__,model))
    # SmolVLA's Euler update is mathematically identical out of place, but its
    # in-place update invalidates tensors needed by autograd.
    method=model.sample_actions
    source=textwrap.dedent(inspect.getsource(method))
    if 'x_t += dt * v_t' in source:
        tree=ast.parse(source);tree.body[0].decorator_list=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.AugAssign) and isinstance(node.target,ast.Name) and node.target.id=='x_t':
                node.__class__=ast.Assign
                node.targets=[ast.Name(id='x_t',ctx=ast.Store())]
                node.value=ast.BinOp(left=ast.Name(id='x_t',ctx=ast.Load()),op=node.op,right=node.value)
                del node.target;del node.op
        ast.fix_missing_locations(tree);namespace=dict(method.__func__.__globals__)
        exec(compile(tree,inspect.getsourcefile(method),'exec'),namespace)
        model.sample_actions=types.MethodType(namespace['sample_actions'],model)


def load_released_policy(root,entry,stats,device='cuda'):
    root=Path(root).resolve();sys.path.insert(0,str(root/'hub'))
    sys.path.insert(0,str(root/'hub/code/vlash'))
    family=entry['family'];method=family.split('_')[0]
    if method=='pi05':
        from lerobot.policies.pi05.configuration_pi05 import PI05Config as Config
    elif method=='vlash':
        sys.path.insert(0,str(root/'hub'/entry['checkpoint']['code']))
        from vlash_runtime import activate
        activate()
        from vlash.policies.pi05.configuration_pi05 import PI05Config as Config
    else:Config=None
    from tools.checkpoint_tools import load_model
    # Dtype is a precision setting, not an architecture/configuration change.
    if Config is not None:
        original=Config.__post_init__
        def post_init(self):self.dtype='bfloat16';original(self)
        context=patch.object(Config,'__post_init__',post_init)
    else:
        context=nullcontext()
    # The 40 GB PI05 files include unused optimizer state. MAP_PRIVATE reserves
    # the whole file as a potential anonymous copy on this host and fails before
    # loading. Shared file-backed mapping avoids that reservation. The audited
    # loader copies tensors into existing modules (assign=False); mapped source
    # tensors are never trained or mutated.
    mapping=torch.serialization.set_default_mmap_options(mmap.MAP_SHARED) if method=='pi05' else nullcontext()
    with context,mapping:native,_=load_model(root/'hub',entry['checkpoint']['id'],device='cpu',train=True)
    fingerprint=signature(native)
    # Training-set-only normalization, preserving each family's native affine convention.
    native.stats=stats['statistics']
    native.velocity_mean.copy_(torch.tensor(stats['velocity_mean']))
    for key,names in native._affine_buffers.items():
        s=stats['statistics'][key]
        if method=='smolvla':offset,scale=s['mean'],s['std']
        else:
            offset=[(a+b)/2 for a,b in zip(s['q01'],s['q99'])]
            scale=[max((b-a)/2,1e-6) for a,b in zip(s['q01'],s['q99'])]
        for name,value in zip(names,[offset,scale]):getattr(native,name).copy_(torch.tensor(value))
    # Keep existing vision/language weights frozen for the large families on
    # this 24 GiB workstation. Identical masks apply to each baseline/KR pair.
    # This changes the trainable set only; no modules or adapters are inserted.
    if method!='smolvla':
        native.requires_grad_(False)
        p=native.policy.model
        trainable=[p.paligemma_with_expert.gemma_expert,p.action_in_proj,p.action_out_proj,p.time_mlp_in,p.time_mlp_out] if method=='pi05' else [p.action_expert,p.suffix_embedder,p.action_out_proj]
        for module in trainable:module.requires_grad_(True)
        expert=p.paligemma_with_expert.gemma_expert if method=='pi05' else p.action_expert
        # These inherited language-vocabulary tensors are not used by native
        # action forwards, which consume continuous suffix embeddings.
        for module in [getattr(expert,'lm_head',None),getattr(expert.model,'embed_tokens',None)]:
            if isinstance(module,nn.Module):module.requires_grad_(False)
        # Keep trainable parameters and optimizer moments in FP32; frozen
        # backbone uses the published BF16 compute precision.
        for param in native.parameters():
            if param.requires_grad:param.data=param.data.float()
    native.to(device)
    native.policy.config.device=str(device)
    if signature(native)!=fingerprint:raise RuntimeError('Architecture changed during training setup')
    reference_sampler=native.policy.model.sample_actions
    enable_sampler_gradients(native.policy.model)
    conditioning=entry.get('velocity_conditioning')
    if conditioning is not None:native.tokenizer=UntruncatedTokenizer(native.tokenizer)
    return NativeTrainingPolicy(native,fingerprint,reference_sampler,conditioning)


class NativeTrainingPolicy(nn.Module):
    def __init__(self,native,fingerprint,reference_sampler,velocity_conditioning=None):
        super().__init__();self.native=native;self.architecture_signature=fingerprint
        self.reference_sampler=reference_sampler
        self.velocity_conditioning=velocity_conditioning

    @property
    def method(self):return self.native.method

    def prepare(self,batch):
        b,correction=self.native.prepare(velocity_prompt_batch(batch,self.velocity_conditioning))
        if self.native.recipe['kr']:
            correction=correction*batch['velocity_required'][:,None,None].to(correction)
            b['action']=self.native.affine(batch['action']-correction,'action')
        return b,correction

    def forward(self,batch,weights=None,noise=None,time=None):
        b,_=self.prepare(batch);p=self.native.policy
        tokens,masks=b['observation.language.tokens'],b['observation.language.attention_mask']
        actions=p.prepare_action(b)
        if self.method=='pi05':
            images,image_masks=p._preprocess_images(b)
            losses=p.model.forward(images,image_masks,tokens,masks,actions,noise,time)
        else:
            images,image_masks=p.prepare_images(b)
            losses=p.model.forward(images,image_masks,tokens,masks,p.prepare_state(b),actions,noise,time)
        dims=7 if self.method=='pi05' else 32
        losses=losses[...,:dims].float();valid=(~b['action_is_pad']).to(losses)
        w=valid if weights is None else valid*weights.to(valid)
        return ((losses*w[...,None]).sum((1,2))/(w.sum(1).clamp_min(1)*dims)).mean()

    def sample_absolute(self,batch,noise=None,reference=False):
        from utils.real_robot_contract import absolute_pose
        b,correction=self.prepare(batch);p=self.native.policy
        tokens,masks=b['observation.language.tokens'],b['observation.language.attention_mask']
        noise=None if noise is None else noise.clone()
        sampler=self.reference_sampler if reference else p.model.sample_actions
        with sampling_cache_context(p.model) if self.method=='pi05' else nullcontext():
            if self.method=='pi05':
                images,image_masks=p._preprocess_images(b)
                prediction=sampler(images,image_masks,tokens,masks,noise=noise)
            else:
                images,image_masks=p.prepare_images(b)
                prediction=sampler(images,image_masks,tokens,masks,p.prepare_state(b),noise=noise)
        delta=self.native.affine(prediction[...,:7].float(),'action',inverse=True)+correction
        return absolute_pose(delta,batch['reference_state'][:,None,:],'wrapped_xyz_v1')

    def check_velocity_tokens(self,batch):
        """Counterfactual preprocessing check, without changing training samples."""
        if self.velocity_conditioning is None:return None
        probe=dict(batch)
        probe['velocity_required']=torch.ones_like(batch['velocity_required'],dtype=torch.bool)
        probe['velocity']=torch.zeros_like(batch['velocity'])
        first,_=self.prepare(probe)
        probe['velocity']=probe['velocity'].clone();probe['velocity'][:,0]=.1
        second,_=self.prepare(probe)
        changed=not torch.equal(first['observation.language.tokens'],second['observation.language.tokens'])
        if not changed:raise RuntimeError('Current velocity does not reach the neural language input')
        return dict(passed=True,velocity_changes_neural_tokens=True,architecture_changes=False)

    def assert_architecture(self):
        if signature(self.native)!=self.architecture_signature:raise RuntimeError('Native architecture fingerprint changed')
