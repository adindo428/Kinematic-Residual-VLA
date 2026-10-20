#!/usr/bin/env python3
"""Real AV1 training batch: velocity sensitivity, backward, optimizer step and reload."""
import argparse,copy,dataclasses,json,random,sys
from pathlib import Path
import numpy as np
import torch,yaml,draccus,safetensors.torch
from huggingface_hub import hf_hub_download
from tracking_checkpoint import load_checkpoint
from tracking_recorded import DEFAULT_DATASET,read_episode,make_batch,noise_for
from tracking_online_common import write_json
from policies.dynamicvla.modeling_dynamicvla import load_dynamicvla

def main(a):
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    cfg=yaml.safe_load(a.config.read_text());reference=cfg['DATASET']['NORMALIZATION_REFERENCE_CHECKPOINT']
    policy=load_checkpoint(reference)
    # Restore shared pretrained MODEL weights only; retain matched train-only normalizers.
    checkpoint=hf_hub_download('hzxie/dynamic-vla-DOM','model.safetensors')
    load_dynamicvla(policy,checkpoint,device='cuda',checkpoint_keys_mapping='model._orig_mod.//model.')
    for key,value in cfg['POLICY'].items():
        if key != 'TYPE' and hasattr(policy.config,key.lower()):setattr(policy.config,key.lower(),value)
    policy.model._set_requires_grad()
    assert not policy.config.enable_residual_flow and policy.config.condition_on_object_velocity
    trainable={n:p.requires_grad for n,p in policy.named_parameters() if not n.startswith(('normalize','unnormalize'))}
    assert all(trainable.values()),[n for n,v in trainable.items() if not v][:10]
    splits=json.loads((a.dataset/'meta/split_assignments.json').read_text());ep=splits['train'][0]
    batch=make_batch(read_episode(a.dataset,ep),[10,30]);noise=noise_for(ep,[10,30]);t=torch.tensor([.25,.75],device='cuda')
    policy.train();loss,_=policy(batch,noise=noise,time=t)
    assert torch.isfinite(loss);loss.backward()
    grad=policy.model.state_proj.weight.grad[:,6:9]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    assert not policy.model._last_residual_enabled
    opt=torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],lr=1e-4)
    opt.step();opt.zero_grad(set_to_none=True);del opt
    policy.eval()
    with torch.inference_mode():
        pred=policy.predict_action_chunk(batch,noise=noise)
        changed=dict(batch);changed['observation.environment_state']=batch['observation.environment_state'].clone()
        changed['observation.environment_state'][:,-1,6]+=.05
        altered=policy.predict_action_chunk(changed,noise=noise)
    assert pred.shape==(2,20,7) and torch.isfinite(pred).all()
    sensitivity=float((pred-altered).abs().max());assert sensitivity>1e-7
    a.output.mkdir(parents=True,exist_ok=False)
    payload=draccus.encode(policy.config);payload['delta_timestamps']={'observation':[-2,0]}
    write_json(a.output/'config.json',payload)
    safetensors.torch.save_file({k:v.detach().cpu().contiguous() for k,v in policy.state_dict().items()},str(a.output/'model.best_validation_action_mse.safetensors'))
    del policy;torch.cuda.empty_cache();reloaded=load_checkpoint(a.output)
    with torch.inference_mode():again=reloaded.predict_action_chunk(batch,noise=noise)
    torch.testing.assert_close(pred,again,atol=1e-5,rtol=1e-5)
    report={'finite_loss':float(loss.detach()),'state_velocity_gradient_nonzero':True,'all_model_parameters_trainable':True,
      'ordinary_action_target_no_analytic_drift':True,'velocity_prediction_max_change':sensitivity,'reload_matches':True,
      'shape':list(pred.shape),'source':'train split recorded current velocity','checkpoint':checkpoint}
    write_json(a.output/'smoke.json',report);print('VELOCITY TRAINING SMOKE PASSED',report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=DEFAULT_DATASET);main(p.parse_args())
