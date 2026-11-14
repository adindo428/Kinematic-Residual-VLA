"""Diagnostic-only streaming warmup and evidence that the worker receives gates."""
import json
import os
import sys
import time
from pathlib import Path


def inference_loop(pretrained_model,cfg,q_in,q_out):
    sys.path.insert(0,os.environ['KRVLA_POLICY_SOURCE'])
    import torch
    from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy
    original_loop=DynamicVLAPolicy._inference_loop
    original_chunk=DynamicVLAPolicy._get_action_chunk
    def chunk(policy,batch,*args,**kwargs):
        dummy=batch.get('task')==['dummy text input'] and 'index' not in batch
        if dummy:
            batch={**batch,'observation.environment_state':torch.zeros(1,cfg.n_obs_steps,9,device=next(policy.parameters()).device)}
            batch.update(policy.extract_residual_batch_fields(batch))
            batch[cfg.residual_phase_gate_key]=torch.ones(1,cfg.chunk_size,1,device=next(policy.parameters()).device)
        gate=policy.prepare_phase_gate(batch,next(policy.parameters()).device)
        assert gate is not None and torch.isfinite(gate).all() and ((gate>=0)&(gate<=1)).all()
        actions = original_chunk(policy,batch,*args,**kwargs)
        lead_s = float(os.environ.get('DOM_RESIDUAL_TARGET_LEAD_S', '0'))
        tick = time.perf_counter()
        from krv_target_timing import lead_correction
        correction = lead_correction(policy,batch,actions,lead_s,gate)
        if lead_s:
            actions = actions + correction
        if not dummy:
            with Path(os.environ['DOM_GATE_WORKER_TRACE']).open('a') as f:
                f.write(json.dumps({'index':int(batch['index']), 'gate_first':float(gate[0,0,0]),'gate_min':float(gate.min()),'gate_max':float(gate.max()),
                    'target_lead_s':lead_s, 'additional_correction_max_m':float(correction.abs().max()),
                    'additional_correction_host_ms':1000*(time.perf_counter()-tick)})+'\n')
        return actions
    DynamicVLAPolicy._get_action_chunk=chunk
    return original_loop(pretrained_model,cfg,q_in,q_out)
