"""Verify actual preprocessing and flow targets without downloading a backbone."""
import ast,json
from pathlib import Path
from types import SimpleNamespace
import pytest,torch
from proactive_vla.residual_flow import extract_dom_dynamic_fields
source=Path(__file__).resolve().parents[1]/'policies/dynamicvla/modeling_dynamicvla.py'
tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DynamicVLAPolicy')
ns={'torch':torch,'OBS_STATE':'observation.state','extract_dom_dynamic_fields':extract_dom_dynamic_fields}
for node in [next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pad_vector')]+[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('prepare_state','extract_residual_batch_fields')]:
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)

def policy(enabled=True):
 c=SimpleNamespace(enable_residual_flow=False,condition_on_object_velocity=enabled,max_state_dim=32,n_obs_steps=2,residual_phase_gate_mode='constant',
   velocity_input_mean=[1,2,3],velocity_input_std=[2,4,8],residual_velocity_key='dynamic_object.velocity_xyz',
   residual_phase_gate_key='dynamic_object.phase_gate', residual_position_key='dynamic_object.position_xyz',residual_covariance_key='dynamic_object.covariance_xyz',
   residual_lookahead_position_key='dynamic_object.lookahead_position_xyz',residual_lookahead_horizon_key='dynamic_object.lookahead_horizon_s')
 return SimpleNamespace(config=c)

def test_velocity_uses_current_observation_without_position_or_future():
 p=policy();env=torch.zeros(2,2,9);env[:,:-1,6:9]=-999;env[:,-1,6:9]=torch.tensor([3.,6.,11.])
 batch={'observation.state':torch.ones(2,2,6),'observation.environment_state':env,'dynamic_object.future_velocity_xyz':torch.full((2,20,3),999.)}
 batch.update(ns['extract_residual_batch_fields'](p,batch));state=ns['prepare_state'](p,batch)
 torch.testing.assert_close(state[:,:6],torch.ones(2,6));torch.testing.assert_close(state[:,6:9],torch.ones(2,3));assert torch.count_nonzero(state[:,9:])==0
 assert not p.config.enable_residual_flow
 batch['dynamic_object.velocity_xyz'][:,-1,0]+=2
 changed=ns['prepare_state'](p,batch);assert torch.all(changed[:,6]==2)
 torch.testing.assert_close(changed[:,:6],state[:,:6])

def test_plain_baseline_keeps_zero_padding_and_ignores_velocity():
 p=policy(False);batch={'observation.state':torch.ones(2,2,6),'dynamic_object.velocity_xyz':torch.randn(2,2,3)}
 assert ns['extract_residual_batch_fields'](p,batch)=={}
 state=ns['prepare_state'](p,batch);assert torch.count_nonzero(state[:,6:])==0

@pytest.mark.parametrize('value',[None,torch.zeros(2,20,3),torch.full((2,2,3),float('nan'))])
def test_missing_future_or_nonfinite_velocity_is_rejected(value):
 batch={'observation.state':torch.ones(2,2,6)}
 if value is not None:batch['dynamic_object.velocity_xyz']=value
 with pytest.raises(ValueError):ns['prepare_state'](policy(),batch)

def test_state_projection_has_velocity_gradient():
 p=policy();batch={'observation.state':torch.ones(2,2,6),'dynamic_object.velocity_xyz':torch.ones(2,2,3)*5}
 projection=torch.nn.Linear(32,12);projection(ns['prepare_state'](p,batch)).square().sum().backward()
 assert projection.weight.grad[:,6:9].abs().sum()>0


def history_policy():
 p=policy();p.config.velocity_conditioning_mode='history';return p


def test_history_retains_current_slots_and_normalizes_every_sample():
 p=history_policy()
 velocity=torch.tensor([[[1.,2.,3.],[3.,6.,11.]]])
 batch={'observation.state':torch.ones(1,2,6),'dynamic_object.velocity_xyz':velocity}
 state=ns['prepare_state'](p,batch)
 torch.testing.assert_close(state[:,6:9],torch.ones(1,3))
 torch.testing.assert_close(state[:,9:12],torch.zeros(1,3))
 assert torch.count_nonzero(state[:,12:])==0
 batch['dynamic_object.velocity_xyz']=velocity.clone();batch['dynamic_object.velocity_xyz'][:,0,0]+=2
 changed=ns['prepare_state'](p,batch)
 torch.testing.assert_close(changed[:,:9],state[:,:9])
 assert changed[0,9]==1


def test_history_gets_gradients_for_both_observation_velocities():
 p=history_policy();v=torch.randn(2,2,3,requires_grad=True)
 state=ns['prepare_state'](p,{'observation.state':torch.ones(2,2,6),'dynamic_object.velocity_xyz':v})
 state[:,6:12].sum().backward()
 assert (v.grad!=0).all()


@pytest.mark.parametrize('value',[torch.zeros(2,3),torch.zeros(2,20,3),torch.full((2,2,3),float('nan'))])
def test_history_rejects_missing_clip_future_clip_and_nonfinite(value):
 with pytest.raises(ValueError):
  ns['prepare_state'](history_policy(),{'observation.state':torch.ones(2,2,6),'dynamic_object.velocity_xyz':value})


def test_history_rejects_misaligned_robot_clip_and_insufficient_slots():
 p=history_policy();batch={'observation.state':torch.ones(2,1,6),'dynamic_object.velocity_xyz':torch.zeros(2,2,3)}
 with pytest.raises(ValueError):ns['prepare_state'](p,batch)
 batch['observation.state']=torch.ones(2,2,6);p.config.max_state_dim=9
 with pytest.raises(ValueError):ns['prepare_state'](p,batch)


def test_history_ignores_future_velocity_fields():
 p=history_policy();batch={'observation.state':torch.ones(2,2,6),'observation.environment_state':torch.randn(2,2,9)}
 batch.update(ns['extract_residual_batch_fields'](p,batch))
 state=ns['prepare_state'](p,batch)
 batch['dynamic_object.future_velocity_xyz']=torch.full((2,20,3),float('nan'))
 torch.testing.assert_close(ns['prepare_state'](p,batch),state)

