"""Validate exact official checkpoint loading and a finite GPU action chunk."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import torch
import safetensors.torch

def probe(source,checkpoint,output):
    sys.path.insert(0,str(source))
    from utils.helpers import get_policy_cfg
    from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy
    assert torch.cuda.is_available(),'GPU allocation required'
    torch.cuda.reset_peak_memory_stats()
    cfg=get_policy_cfg(cfg_file=checkpoint/'config.json');cfg.enable_streaming=False
    policy=DynamicVLAPolicy.from_pretrained(checkpoint,config=cfg).cuda().eval()
    saved=safetensors.torch.load_file(str(checkpoint/'model.safetensors'))
    actual=policy.state_dict()
    assert set(saved)==set(actual),'Released weights must match all parameters and normalizers'
    for key,tensor in saved.items():
        if 'normalize' in key:
            assert torch.isfinite(actual[key]).all(),key
            torch.testing.assert_close(actual[key].cpu(),tensor,rtol=0,atol=0)
    batch={k:torch.zeros(1,cfg.n_obs_steps,*v.shape,device='cuda') for k,v in cfg.input_features.items()}
    batch['task']=['Pick up the moving object and place it in the container.']
    with torch.inference_mode():
        actions=policy.predict_action_chunk(batch)
        torch.cuda.synchronize();start=time.perf_counter()
        actions=policy.predict_action_chunk(batch)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
    assert tuple(actions.shape)==(1,20,7) and torch.isfinite(actions).all()
    report={'validated':True,'gpu':torch.cuda.get_device_name(0),'checkpoint':str(checkpoint),
            'normalizers_exactly_match_checkpoint':True,'all_state_keys_match':True,
            'finite_action_shape':list(actions.shape),'warmed_chunk_time_s':elapsed,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'device_total_bytes':torch.cuda.get_device_properties(0).total_memory,
            'torch':torch.__version__,'transformers':importlib.metadata.version('transformers')}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();probe(a.source,a.checkpoint,a.output)
