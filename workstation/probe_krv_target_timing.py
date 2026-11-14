#!/usr/bin/env python3
"""Check target-lead units, centering, gating and the actual loaded checkpoint."""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ['DOM_GATE_DIAGNOSTIC_ROOT'])
SOURCE = REPO/'scratch/dom45/training/krv100-full-deltaai-20261106/source'
sys.path.insert(0, str(SOURCE))
import torch
from utils.helpers import get_policy_cfg
from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy
from safetensors.torch import load_model
from proactive_vla.residual_flow import centered_action_drift
from krv_target_timing import lead_correction


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = get_policy_cfg(cfg_file=ROOT/'checkpoint/config.json')
    cfg.enable_streaming = False
    policy = DynamicVLAPolicy.from_pretrained(ROOT/'checkpoint', config=cfg).cuda().eval()
    load_model(policy, str(ROOT/'checkpoint/model.safetensors'), strict=True)
    batch = {k: torch.zeros(1,cfg.n_obs_steps,*v.shape,device='cuda') for k,v in cfg.input_features.items()}
    batch['task'] = ['Place the moving avocado into the container.']
    velocity = torch.tensor([[.1,.2,-.05]],device='cuda')
    batch['dynamic_object.velocity_xyz'] = velocity
    batch['observation.environment_state'] = torch.zeros(1,cfg.n_obs_steps,9,device='cuda')
    batch['observation.environment_state'][...,6:9] = velocity[:,None,:]
    gate = torch.linspace(1,0,cfg.chunk_size,device='cuda')[None,:,None]
    batch[cfg.residual_phase_gate_key] = gate
    noise = torch.randn(1,cfg.chunk_size,cfg.max_action_dim,device='cuda')
    with torch.inference_mode():
        old = policy.predict_action_chunk(dict(batch),noise.clone())
        correction = lead_correction(policy,batch,old,.2,gate)
        aligned = old + correction
        std = policy.get_action_normalization_std(old.device,old.dtype)
        mean = torch.tensor(cfg.residual_drift_mean,device='cuda',dtype=old.dtype)
        training_velocity = (mean[1,:3]-mean[0,:3])*std[:3]/cfg.residual_action_dt
        expected = .2*(velocity-training_velocity)[:,None,:]*gate
        torch.testing.assert_close(correction[...,:3],expected,rtol=1e-5,atol=1e-7)
        torch.testing.assert_close(old[...,3:],aligned[...,3:],rtol=0,atol=0)
        torch.testing.assert_close(old,old+lead_correction(policy,batch,old,0,gate),rtol=0,atol=0)
        assert not lead_correction(policy,batch,old,.2,torch.zeros_like(gate)).any()
        # Compare against an independently constructed normalized horizon and
        # its shifted training center. No policy forward/noise changes involved.
        kwargs=dict(action_std=std,drift_mean=mean,center_drift=True,
                    action_dt=cfg.residual_action_dt,phase_gate=gate)
        previous, _ = centered_action_drift(velocity,(1,20,32),**kwargs)
        h = torch.arange(20,device='cuda')*.04+.2
        full_raw = velocity[:,None,:]*h[None,:,None]/std[:3]
        shifted_mean = mean[:,:3]+(.2/.04)*(mean[1,:3]-mean[0,:3])
        new_centered = (full_raw-shifted_mean)*gate
        torch.testing.assert_close((new_centered-previous[...,:3])*std[:3],correction[...,:3],rtol=1e-5,atol=1e-7)
        assert torch.isfinite(aligned).all()
        for bad in (-.1,float('nan'),float('inf')):
            try: lead_correction(policy,batch,old,bad,gate)
            except ValueError: pass
            else: raise AssertionError('Invalid lead accepted')
    result = {'passed':True,'target_lead_s':.2,'strict_checkpoint_loaded':True,
              'zero_lead_matches_original_exactly':True,'zero_gate_disables_extra_correction':True,
              'matches_shifted_centered_horizon':True,'xyz_only':True,'finite_actions':True,
              'raw_time_grid_s':[.2,.96],'training_velocity_mean_xyz':training_velocity.tolist(),
              'max_additional_correction_m':float(correction.abs().max()),
              'training_alignment_changed':False,'interpretation':'Inference-only target-lead ablation'}
    (ROOT/'timing_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
