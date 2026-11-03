import unittest,types
import torch
from client import batch_from_observation,Adapter
class Fake:
 def __init__(self):
  self.policy=types.SimpleNamespace(config=types.SimpleNamespace(input_features={},output_features={}),reset=lambda:None);self.calls=0
 def predict(self,b):
  self.calls+=1;return torch.arange(140).reshape(1,20,7).float()
class Contracts(unittest.TestCase):
 def obs(self):return {'observation.state':torch.arange(7).reshape(1,1,7).float(),'observation.images.opst_cam':torch.zeros(1,1,3,4,4),'task':['Pick up the object.']}
 def test_state_and_image(self):
  b=batch_from_observation(self.obs(),torch);self.assertEqual(tuple(b['observation.state'].shape),(1,6));self.assertEqual(tuple(b['observation.images.opst_cam'].shape),(1,3,4,4));self.assertEqual(b['observation.state'][0,-1],5)
 def test_dummy_targets(self):
  b=batch_from_observation(self.obs(),torch);self.assertEqual(b['action'].sum(),0);self.assertFalse(b['action_is_pad'].any());self.assertEqual(b['velocity'].sum(),0)
 def test_queue(self):
  f=Fake();a=Adapter(f,torch)
  for i in range(20):self.assertEqual(a.select_action(self.obs())[0,0],i*7)
  self.assertEqual(f.calls,1);a.select_action(self.obs());self.assertEqual(f.calls,2);self.assertFalse(hasattr(a.config,'use_delta_action'))
 def test_reset(self):
  f=Fake();a=Adapter(f,torch);a.select_action(self.obs());a.reset();a.select_action(self.obs());self.assertEqual(f.calls,2)
if __name__=='__main__':unittest.main()
