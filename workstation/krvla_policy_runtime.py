"""Supply auxiliary object state only for the streaming process's dummy warmup."""
import os
import sys


def inference_loop(pretrained_model, cfg, q_in, q_out):
    sys.path.insert(0,os.environ['KRVLA_POLICY_SOURCE'])
    import torch
    from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy
    original_loop=DynamicVLAPolicy._inference_loop
    original_chunk=DynamicVLAPolicy._get_action_chunk

    def chunk(policy,batch,*args,**kwargs):
        if (batch.get('task')==['dummy text input'] and 'index' not in batch
                and 'observation.environment_state' not in batch):
            # The frozen streaming warmup enumerates input_features only; its
            # analytic velocity input is intentionally outside those features.
            # Real observations never receive a missing-velocity fallback.
            batch={**batch,'observation.environment_state':torch.zeros(
                1,cfg.n_obs_steps,9,device=next(policy.parameters()).device)}
        return original_chunk(policy,batch,*args,**kwargs)

    DynamicVLAPolicy._get_action_chunk=chunk
    return original_loop(pretrained_model,cfg,q_in,q_out)
