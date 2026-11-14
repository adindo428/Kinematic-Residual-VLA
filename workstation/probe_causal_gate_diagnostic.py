"""Verify inference-only gating preserves ungated actions at gate one."""
import json,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'scratch/dom45'
OUT=Path(os.environ.get('DOM_GATE_DIAGNOSTIC_ROOT',str(WORK/'evaluations/krv100-full-causal-gate-cr-20261106')))
SOURCE=WORK/'training/krv100-full-deltaai-20261106/source'
sys.path.insert(0,str(SOURCE))
import torch
from safetensors.torch import load_model
from utils.helpers import get_policy_cfg
from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy

def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    cfg=get_policy_cfg(cfg_file=OUT/'checkpoint/config.json');assert cfg.residual_phase_gate_mode=='causal_rules';cfg.enable_streaming=False
    policy=DynamicVLAPolicy.from_pretrained(OUT/'checkpoint',config=cfg).cuda().eval()
    load_model(policy,str(OUT/'checkpoint/model.safetensors'),strict=True)
    batch={k:torch.zeros(1,cfg.n_obs_steps,*v.shape,device='cuda') for k,v in cfg.input_features.items()}
    batch['task']=['Place the moving object into the container.'];batch['observation.environment_state']=torch.zeros(1,cfg.n_obs_steps,9,device='cuda');batch['observation.environment_state'][...,6]=.1
    batch[cfg.residual_phase_gate_key]=torch.ones(1,cfg.chunk_size,1,device='cuda')
    noise=torch.randn(1,cfg.chunk_size,cfg.max_action_dim,device='cuda')
    with torch.inference_mode():
        one=policy.predict_action_chunk(dict(batch),noise.clone())
        cfg.residual_phase_gate_mode='constant'
        original=policy.predict_action_chunk(dict(batch),noise.clone())
        torch.testing.assert_close(one,original,rtol=0,atol=0)
        cfg.residual_phase_gate_mode='causal_rules';batch[cfg.residual_phase_gate_key].zero_()
        zero=policy.predict_action_chunk(dict(batch),noise.clone())
        assert zero.shape==(1,20,7) and torch.isfinite(zero).all()
        torch.testing.assert_close(zero[...,3:],one[...,3:],rtol=0,atol=0)
        assert (zero[...,:3]-one[...,:3]).abs().max()>0
    result={'strict_checkpoint_loaded':True,'same_noise_gate_one_matches_constant_exactly':True,'gate_zero_changes_only_translation':True,'finite_actions':True,'max_translation_difference_m':float((zero[...,:3]-one[...,:3]).abs().max()),'gate_mode':'causal_rules'}
    (OUT/'policy_probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
