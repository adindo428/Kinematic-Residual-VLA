import json,tempfile,unittest,sys,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from unittest.mock import patch
import handoff
import torch
class HandoffContracts(unittest.TestCase):
 def checkpoint(self,folder):
  r=json.loads((Path(__file__).parents[1]/'recipes/smolvla_baseline.json').read_text())
  p=torch.nn.Parameter(torch.ones(2));o=torch.optim.AdamW([p],lr=r['lr'],betas=(.9,.95),eps=1e-8,weight_decay=r['weight_decay']);p.grad=torch.ones_like(p);o.step()
  for state in o.state.values():state['step'].fill_(20000)
  o.param_groups[0]['lr']=r['decay_lr'];o.param_groups[0]['initial_lr']=r['lr']
  c={'model':{'p':p.detach()},'optimizer':o.state_dict(),'scheduler':{'last_epoch':20000,'_last_lr':[r['decay_lr']]},'progress':dict(updates=20000,samples=1280000,epoch=0,cursor=1280000),'rng':[dict(python=1,numpy=1,torch=1,cuda=1)]*2,'identity':'trusted-production'}
  meta={k:c[k] for k in ['identity','progress']};path=Path(folder)/'latest.pt';torch.save(c,path);return path,c,meta,r
 def test_completed_old_budget_is_valid_continuation_not_completion(self):
  with tempfile.TemporaryDirectory() as d:
   path,c,m,r=self.checkpoint(d);handoff.inspect_checkpoint(path,m,r)
   self.assertLess(m['progress']['updates'],r['steps'])
 def test_missing_corrupt_incomplete_and_wrong_counters_fail_closed(self):
  with tempfile.TemporaryDirectory() as d:
   p,c,m,r=self.checkpoint(d)
   for field in ['optimizer','rng','model']:
    bad=copy.deepcopy(c);bad.pop(field);torch.save(bad,p)
    with self.assertRaises((ValueError,KeyError)):handoff.inspect_checkpoint(p,m,r)
   bad=copy.deepcopy(c);bad['scheduler']['last_epoch']=0;torch.save(bad,p)
   with self.assertRaises(ValueError):handoff.inspect_checkpoint(p,m,r)
   p.unlink()
   with self.assertRaises(ValueError):handoff.inspect_checkpoint(p,m,r)
 def test_stale_signal_or_restarted_allocation_rejected(self):
  m={'identity':'x','progress':{'updates':10}};s={'reason':'allocation_signal','progress':m['progress']}
  for times,old,new in [((10,9,11),('old',),('old',)),((10,11,11),('old',),('new',))]:
   with self.assertRaises(ValueError):handoff.validate_saved_stop(m,s,m,*times,old,new)
 def test_allocate_source_failure_never_signals_parent(self):
  with tempfile.TemporaryDirectory() as d,patch.object(handoff,'HERE',Path(d)),patch.object(handoff,'verify_frozen',side_effect=ValueError('bad source')),patch.object(handoff,'stop_parent') as stop:
   c={'root':d,'runs':{'smolvla_baseline':{}}}
   with self.assertRaises(ValueError):handoff.allocate(c,'smolvla_baseline')
   stop.assert_not_called()
if __name__=='__main__':unittest.main()
