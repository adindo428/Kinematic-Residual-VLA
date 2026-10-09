#!/usr/bin/env python3
"""Validate Coordinate's neural velocity branch, residual restoration, and reload."""
import argparse,json,random,sys
from pathlib import Path
import numpy as np
import torch,yaml,draccus,safetensors.torch
from huggingface_hub import hf_hub_download
from tracking_checkpoint import load_checkpoint
from tracking_recorded import DEFAULT_DATASET,read_episode,make_batch,noise_for
from tracking_online_common import write_json
from tracking_variants import validate_variant
from policies.dynamicvla.modeling_dynamicvla import load_dynamicvla
from proactive_vla.residual_flow import centered_action_drift

def main(a):
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    cfg=yaml.safe_load(a.config.read_text())
    policy=load_checkpoint(cfg['DATASET']['NORMALIZATION_REFERENCE_CHECKPOINT'])
    checkpoint=hf_hub_download('hzxie/dynamic-vla-DOM','model.safetensors')
    load_dynamicvla(policy,checkpoint,device='cuda',checkpoint_keys_mapping='model._orig_mod.//model.')
    for key,value in cfg['POLICY'].items():
        if key!='TYPE' and hasattr(policy.config,key.lower()):setattr(policy.config,key.lower(),value)
    # Use the previously measured train-only centering for this one-batch check.
    # Full training recomputes its own train-only centering with the original routine.
    old=json.loads(a.previous_config.read_text())
    policy.config.residual_drift_mean=old['residual_drift_mean']
    policy.config.validate_residual_formulation()
    validate_variant(policy.config,'coordinate',require_neural_velocity=True)
    policy.model._set_requires_grad()
    trainable={n:p.requires_grad for n,p in policy.named_parameters() if not n.startswith(('normalize','unnormalize'))}
    assert all(trainable.values()),[n for n,v in trainable.items() if not v][:10]
    splits=json.loads((a.dataset/'meta/split_assignments.json').read_text());ep=splits['train'][0]
    anchors=[10,30,50,70,90,110,130,150]
    batch=make_batch(read_episode(a.dataset,ep),anchors);noise=noise_for(ep,anchors)
    t=torch.linspace(.1,.9,len(anchors),device='cuda')
    policy.train();loss,_=policy(batch,noise=noise,time=t);finite_loss=float(loss.detach())
    assert torch.isfinite(loss);loss.backward()
    grad=policy.model.state_proj.weight.grad[:,6:9]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    assert policy.model._last_residual_enabled and policy.model._last_residual_drift_l2>0
    opt=torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],lr=1e-4)
    opt.step();opt.zero_grad(set_to_none=True);del opt,loss
    policy.eval()
    with torch.inference_mode():
        pred=policy.predict_action_chunk(batch,noise=noise)
        # Isolate neural sensitivity by keeping analytic restoration disabled for both.
        policy.config.enable_residual_flow=False
        neural=policy.predict_action_chunk(batch,noise=noise)
        changed=dict(batch);changed['observation.environment_state']=batch['observation.environment_state'].clone()
        changed['observation.environment_state'][:,-1,6]+=.05
        altered=policy.predict_action_chunk(changed,noise=noise)
        policy.config.enable_residual_flow=True
    sensitivity=float((neural-altered).abs().max());assert sensitivity>1e-7
    assert pred.shape==(8,20,7) and torch.isfinite(pred).all()
    std=policy.get_action_normalization_std(pred.device,pred.dtype)
    drift,_=centered_action_drift(batch['observation.environment_state'][:,-1,6:9],(8,20,32),
        action_std=std,drift_mean=policy.config.residual_drift_mean,center_drift=True,
        action_dt=policy.config.residual_action_dt,device=pred.device,dtype=pred.dtype)
    torch.testing.assert_close(pred-neural,drift[...,:7]*std[:7],atol=2e-5,rtol=1e-4)
    a.output.mkdir(parents=True,exist_ok=False)
    payload=draccus.encode(policy.config);payload['delta_timestamps']={'observation':[-2,0]}
    write_json(a.output/'config.json',payload)
    safetensors.torch.save_file({k:v.detach().cpu().contiguous() for k,v in policy.state_dict().items()},str(a.output/'model.best_validation_action_mse.safetensors'))
    del policy;torch.cuda.empty_cache();reloaded=load_checkpoint(a.output)
    validate_variant(reloaded.config,'coordinate',require_neural_velocity=True)
    with torch.inference_mode():again=reloaded.predict_action_chunk(batch,noise=noise)
    torch.testing.assert_close(pred,again,atol=1e-5,rtol=1e-5)
    report={'status':'passed','finite_loss':finite_loss,'state_velocity_gradient_nonzero':True,
        'neural_velocity_prediction_max_change':sensitivity,'analytic_restoration_matches':True,
        'all_model_parameters_trainable':True,'reload_matches':True,'shape':list(pred.shape),
        'source':'train split recorded current velocity','initialization':checkpoint}
    write_json(a.output/'smoke.json',report);print('COORDINATE VELOCITY SMOKE PASSED',report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous-config',type=Path,required=True);p.add_argument('--dataset',type=Path,default=DEFAULT_DATASET)
    main(p.parse_args())
