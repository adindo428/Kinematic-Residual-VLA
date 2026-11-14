"""Test physical gate units, causality, residual roundtrip and sampler recycling."""
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

stage = Path(__file__).resolve().parents[1]/'scratch/dom45/training/krv-distance-gate-old20k-20261112'
sys.path.insert(0,str(stage/'source'))
from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy
from proactive_vla.distance_gate import current_distance_gate
from proactive_vla.residual_flow import residualize_actions, restore_residual_actions
from core.dom_distributed import RankBatchSampler
from core.dom_distributed_training import LimitedUpdates
from utils.dom_experiment import atomic_json

ee = torch.zeros(5,2,3)
obj = ee.clone(); obj[:,-1,0] = torch.tensor([0.,.049,.05,.051,.2]); obj[:,0,0] = 999
gate = current_distance_gate(ee,obj,.05,20,2,'cpu')
assert gate[:,0,0].tolist() == [0.,0.,0.,1.,1.]
torch.testing.assert_close(gate,current_distance_gate(ee.flip(-1),obj.flip(-1),.05,20,2,'cpu'))
translation = torch.tensor([.2,-.3,.1])
torch.testing.assert_close(gate[[0,1,3,4]],current_distance_gate(ee[[0,1,3,4]]+translation,obj[[0,1,3,4]]+translation,.05,20,2,'cpu'))
cfg = SimpleNamespace(enable_residual_flow=True,condition_on_object_velocity=True,residual_phase_gate_mode='distance_threshold',
    residual_phase_gate_key='dynamic_object.phase_gate',residual_position_key='dynamic_object.position_xyz',
    residual_velocity_key='dynamic_object.velocity_xyz',residual_covariance_key='dynamic_object.covariance_xyz',
    residual_lookahead_position_key='dynamic_object.lookahead_position_xyz',residual_lookahead_horizon_key='dynamic_object.lookahead_horizon_s',
    residual_distance_threshold_m=.05,chunk_size=20,n_obs_steps=2)
fake = SimpleNamespace(config=cfg)
batch = {'observation.state':torch.cat([ee,ee],-1),'observation.environment_state':torch.cat([obj,ee,torch.ones_like(ee)],-1)}
fields = DynamicVLAPolicy.extract_residual_batch_fields(fake,batch)
batch['observation.state'].fill_(999.)
torch.testing.assert_close(gate,DynamicVLAPolicy.prepare_phase_gate(fake,{**batch,**fields},'cpu'))
poison = {**fields,'dynamic_object.future_position_xyz':torch.full((5,20,3),999.),'dynamic_object.future_velocity_xyz':torch.full((5,20,3),999.),'dynamic_object.phase_gate':1-gate,'action':torch.randn(5,20,7)}
torch.testing.assert_close(gate,DynamicVLAPolicy.prepare_phase_gate(fake,poison,'cpu'))
try: current_distance_gate(torch.zeros(5,20,3),obj,.05,20,2,'cpu')
except ValueError: pass
else: raise AssertionError('Future trajectory accepted')
actions = torch.randn(5,20,7); velocity = torch.ones(5,3)
kw = dict(phase_gate=gate,drift_mean=torch.full((20,7),.2),center_drift=True,drift_mode='trajectory',action_dt=.04)
residual,correction,_ = residualize_actions(actions,velocity,**kw)
restored,_,_ = restore_residual_actions(residual,velocity,**kw)
torch.testing.assert_close(restored,actions)
torch.testing.assert_close(residual[:3],actions[:3],rtol=0,atol=0)
assert torch.count_nonzero(correction[:3]) == 0 and torch.count_nonzero(correction[...,3:]) == 0
for rank in range(2):
    full = list(RankBatchSampler([1000,300],40,rank,2))
    first = list(LimitedUpdates(RankBatchSampler([1000,300],40,rank,2),2))
    resumed = list(RankBatchSampler([1000,300],40,rank,2,cursor=704))
    assert first+resumed == full
atomic_json(stage/'gate_tests.json',dict(passed=True,radius_m=.05,checks=['5cm_boundary','current_observation_only',
    'physical_positions_preserved_before_normalization','translation_rotation_invariance','future_action_phase_inputs_ignored',
    'residual_roundtrip','zero_centered_correction_near_object','orientation_gripper_unchanged','loader_recycle_exact_sample_sequence']))
print('Distance gate and sampler tests passed')
