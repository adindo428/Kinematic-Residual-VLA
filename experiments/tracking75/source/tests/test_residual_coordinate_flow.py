"""Exercise the actual flow methods with a tiny expert, without loading a VLM."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
import pytest
from proactive_vla.residual_flow import scheduled_residual_flow_target, remove_scheduled_residual_drift, restore_residual_actions

source=Path(__file__).resolve().parents[1]/'policies/dynamicvla/modeling_dynamicvla.py'
cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='VLAFlowMatching')
namespace=dict(torch=torch,F=F,scheduled_residual_flow_target=scheduled_residual_flow_target,
               remove_scheduled_residual_drift=remove_scheduled_residual_drift,restore_residual_actions=restore_residual_actions,
               make_att_2d_masks=lambda a,b:a)
for name in ('forward','sample_actions'):
 node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)

class Expert:
 forward=namespace['forward'];sample_actions=namespace['sample_actions']
 def __init__(self,formulation,enabled=True):
  self.config=SimpleNamespace(enable_residual_flow=enabled,residual_flow_formulation=formulation,
   residual_lambda_type='constant',residual_drift_mean=None,residual_center_drift=False,
   residual_drift_scale=1.,residual_drift_mode='trajectory',residual_action_dt=.04,
   residual_lookahead_horizon_s=1.,residual_axis_scale_xyz=(1,1,1),num_steps=4,chunk_size=3,max_action_dim=7)
  self.seen=[]
 def sample_vlm_embedding(self,*a):return None,None
 def denoise_step(self,prefix,cache,x,t):self.seen.append(x.clone());return x*.3
 def _embed_prefix(self,*a,**kw):return torch.zeros(1,1,7),torch.ones(1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.bool)
 def _embed_suffix(self,x,t):self.seen.append(x.clone());return x,torch.ones(1,3,dtype=torch.bool),torch.ones(1,3,dtype=torch.bool)
 def _get_position_ids(self,*a):return None
 def vlm_with_expert(self,**kw):return (None,kw['inputs_embeds'][1]),None
 def action_out_proj(self,x):return self.oracle

@pytest.mark.parametrize('formulation',['field','coordinate'])
def test_training_coordinates_and_derivative(formulation):
 m=Expert(formulation);a=torch.ones(1,3,7);noise=torch.zeros_like(a);t=torch.tensor([.25]);v=torch.tensor([[2.,0,0]])
 drift=torch.zeros_like(a);drift[0,:,0]=torch.tensor([0,.08,.16])
 m.oracle=noise-a+drift
 loss=m.forward(None,None,None,None,None,a,noise=noise,time=t,residual_velocity=v)
 assert torch.count_nonzero(loss)==0
 expected=.75*(a-drift if formulation=='coordinate' else a)
 torch.testing.assert_close(m.seen[0],expected)
 torch.testing.assert_close(a,torch.ones_like(a))

def sample(m,v):
 return m.sample_actions(None,None,None,None,torch.zeros(1,6),noise=torch.ones(1,3,7),residual_velocity=v)

def test_coordinate_velocity_changes_only_final_xyz_restoration():
 a,b=Expert('coordinate'),Expert('coordinate')
 out_a=sample(a,torch.zeros(1,3));out_b=sample(b,torch.tensor([[2.,0,0]]))
 for x,y in zip(a.seen,b.seen):torch.testing.assert_close(x,y)
 expected=torch.zeros_like(out_a);expected[0,:,0]=torch.tensor([0,.08,.16])
 torch.testing.assert_close(out_b-out_a,expected,atol=1e-6,rtol=1e-5)

def test_field_retains_original_per_step_drift_integration():
 m=Expert('field');actual=sample(m,torch.tensor([[2.,0,0]]))
 x=torch.ones_like(actual);d=torch.zeros_like(x);d[0,:,0]=torch.tensor([0,.08,.16])
 for _ in range(4):x=x-.25*(.3*x-d)
 torch.testing.assert_close(actual,x)

@pytest.mark.parametrize('formulation',['field','coordinate'])
def test_zero_drift_equals_baseline(formulation):
 torch.testing.assert_close(sample(Expert(formulation),torch.zeros(1,3)),sample(Expert('field',False),torch.zeros(1,3)))

@pytest.mark.parametrize('formulation,enabled',[('field',False),('field',True),('coordinate',True)])
def test_sampling_preserves_reusable_noise(formulation,enabled):
 m=Expert(formulation,enabled);noise=torch.ones(1,3,7);original=noise.clone()
 args=(None,None,None,None,torch.zeros(1,6))
 first=m.sample_actions(*args,noise=noise,residual_velocity=torch.tensor([[2.,0,0]]))
 torch.testing.assert_close(noise,original)
 second=m.sample_actions(*args,noise=noise,residual_velocity=torch.tensor([[2.,0,0]]))
 torch.testing.assert_close(first,second)
