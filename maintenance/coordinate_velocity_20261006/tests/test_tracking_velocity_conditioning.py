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
 c=SimpleNamespace(enable_residual_flow=False,condition_on_object_velocity=enabled,max_state_dim=32,n_obs_steps=2,
   velocity_input_mean=[1,2,3],velocity_input_std=[2,4,8],residual_velocity_key='dynamic_object.velocity_xyz',
   residual_position_key='dynamic_object.position_xyz',residual_covariance_key='dynamic_object.covariance_xyz',
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

def test_tracking_instruction_and_grasp_fallback():
 from utils.instruction_generator import InstructionGenerator
 assert InstructionGenerator.generate_instruction(json.dumps({'task':'track','instruction':' Track the apple. '}))=='Track the apple.'
 assert 'apple' in InstructionGenerator.generate_instruction({'task':'pick','objects':['apple'],'instruction':''})

def test_four_method_flags_are_distinct():
 import sys
 sys.path.insert(0,str(source.parents[2]/'scripts'))
 from tracking_variants import validate_variant
 for variant in ('baseline','velocity','residual','coordinate'):
  c=SimpleNamespace(enable_residual_flow=variant in ('residual','coordinate'),condition_on_object_velocity=variant=='velocity',residual_flow_formulation='coordinate' if variant=='coordinate' else 'field')
  validate_variant(c,variant)
  if variant=='velocity':
   c.enable_residual_flow=True
   with pytest.raises(ValueError):validate_variant(c,variant)
