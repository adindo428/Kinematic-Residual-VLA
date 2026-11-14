"""Contract tests for the frozen current-speed gate (no GPU required)."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace, MethodType
import torch

root = Path(__file__).resolve().parents[1]/'scratch/dom45/training/krv-speed-gate-old20k-20261112'
sys.path.insert(0, str(root/'source'))
from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy
from proactive_vla.residual_flow import residualize_actions, restore_residual_actions
from utils.dom_experiment import atomic_json

threshold = json.loads((root/'family/speed_gate.json').read_text())['threshold_m_s']
cfg = SimpleNamespace(enable_residual_flow=True, residual_phase_gate_mode='speed_threshold',
    residual_speed_threshold=threshold, residual_velocity_key='dynamic_object.velocity_xyz', chunk_size=20, n_obs_steps=2)
fake = SimpleNamespace(config=cfg)
fake.prepare_residual_velocity = MethodType(DynamicVLAPolicy.prepare_residual_velocity, fake)
velocity = torch.tensor([[0.,0.,0.],[threshold*.99,0.,0.],[threshold*1.01,0.,0.],[-threshold*2,0.,0.]])
batch = {'observation.state':torch.zeros(4,2,6), cfg.residual_velocity_key:torch.stack([velocity*100,velocity],1)}
gate = DynamicVLAPolicy.prepare_phase_gate(fake,batch,'cpu')
assert gate[:,0,0].tolist() == [0.,0.,1.,1.]
poisoned = {**batch,'dynamic_object.position_xyz':torch.full((4,3),float('nan')),
    'dynamic_object.future_velocity_xyz':torch.full((4,20,3),999.),'dynamic_object.phase_gate':torch.rand(4,20,1)}
torch.testing.assert_close(gate,DynamicVLAPolicy.prepare_phase_gate(fake,poisoned,'cpu'),rtol=0,atol=0)
actions = torch.randn(4,20,7)
kwargs = dict(phase_gate=gate,drift_mean=torch.full((20,7),.2),center_drift=True,drift_mode='trajectory',action_dt=.04)
residual, correction, _ = residualize_actions(actions,velocity,**kwargs)
restored, _, _ = restore_residual_actions(residual,velocity,**kwargs)
torch.testing.assert_close(restored,actions)
torch.testing.assert_close(residual[:2],actions[:2],rtol=0,atol=0)
assert torch.count_nonzero(correction[:2]) == 0 and torch.count_nonzero(correction[...,3:]) == 0
torch.testing.assert_close(residual[...,3:],actions[...,3:],rtol=0,atol=0)
atomic_json(root/'gate_tests.json',dict(passed=True,threshold_m_s=threshold,
    checks=['polarity','current_history_frame_only','no_position_or_future_or_external_phase_dependency',
            'target_reconstruction_roundtrip','zero_entire_centered_drift_below_threshold','orientation_gripper_unchanged']))
print('All speed gate contract tests passed')
