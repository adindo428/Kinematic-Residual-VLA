import copy,json,unittest
from pathlib import Path
import torch
from budget import validate_budget,validate_extension,validate_progress,advance,lr_factor,validate_migration
from data import Batches
from common import fingerprint
ROOT=Path(__file__).parent
class BudgetTests(unittest.TestCase):
 def recipes(self,name='smolvla_baseline'):
  return json.loads((ROOT/'parent-recipes'/(name+'.json')).read_text()),json.loads((ROOT/'recipes'/(name+'.json')).read_text())
 def test_exact_update_counts_and_same_math(self):
  for file in (ROOT/'recipes').glob('*.json'):
   old,r=self.recipes(file.stem);validate_budget(r,22370201);validate_extension(old,r)
   self.assertEqual(r['steps'],1747670 if r['global_batch']==64 else 3495340)
   self.assertEqual(r['steps']*r['global_batch'],111850880)
 def test_reject_objective_or_schedule_changes(self):
  old,r=self.recipes()
  for key,value in [('global_batch',128),('lr',.01),('weight_decay',.2),('horizon',10),('steps',20000),('schedule_steps',1747670)]:
   bad={**r,key:value}
   with self.assertRaises(ValueError):validate_extension(old,bad)
 def test_real_sampler_resume_across_five_epoch_boundaries(self):
  # The same native Batches class, two ranks and drop-last semantics.
  r={'training_anchors':101,'global_batch':32,'steps':15}
  all_steps=[]
  for e in range(5):
   per_rank=[list(Batches(101,e,0,32,16,k,2)) for k in range(2)]
   for j in range(3):all_steps.append(sorted(i for rank in per_rank for _,i in rank[j]))
  prefix={'updates':4,'samples':128,'epoch':1,'cursor':32};validate_progress(prefix,r)
  resumed=[]
  while prefix['updates']<15:
   ranks=[list(Batches(101,prefix['epoch'],prefix['cursor'],32,16,k,2)) for k in range(2)]
   for j in range(len(ranks[0])):
    resumed.append(sorted(i for rank in ranks for _,i in rank[j]));advance(prefix,r)
  self.assertEqual(resumed,all_steps[4:]);self.assertEqual(prefix,dict(updates=15,samples=480,epoch=5,cursor=0))
 def test_progress_rejects_logged_or_wrong_cursor(self):
  _,r=self.recipes()
  with self.assertRaises(ValueError):validate_progress(dict(updates=20000,samples=1280000,epoch=1,cursor=1280000),r)
 def test_final_boundary_normalized_before_save(self):
  _,r=self.recipes();end=r['training_anchors']//64*64
  p=dict(updates=r['steps']-1,samples=r['steps']*64-64,epoch=4,cursor=end-64)
  advance(p,r);self.assertEqual((p['epoch'],p['cursor']),(5,0))
 def test_original_lr_exact_and_no_rewarm_after_resume(self):
  import math
  for name in ['pi05_baseline','smolvla_kr','vlash_baseline']:
   old,r=self.recipes(name)
   for u in [0,500,1000,12447,20000,30000,500000,r['steps']]:
    floor=old['decay_lr']/old['lr'];t=min(1,(u-1000)/max(1,old['steps']-1000))
    expected=(u+1)/1000 if u<1000 else floor+(1-floor)*.5*(1+math.cos(math.pi*t))
    self.assertEqual(lr_factor(u,r),expected)
 def test_actual_optimizer_scheduler_save_continue_at_floor(self):
  _,r=self.recipes();p=torch.nn.Parameter(torch.ones(2));opt=torch.optim.AdamW([p],lr=r['lr'],betas=(.9,.95),eps=1e-8,weight_decay=r['weight_decay'])
  sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda u:lr_factor(u,r))
  # Test actual state restoration at both pre-cap and post-cap counters.
  for step in [12000,20000,500000]:
   sched.last_epoch=step;sched._step_count=step+1;opt.param_groups[0]['lr']=r['lr']*lr_factor(step,r);sched._last_lr=[opt.param_groups[0]['lr']]
   p.grad=torch.ones_like(p);state=copy.deepcopy(opt.state_dict());ss=copy.deepcopy(sched.state_dict())
   q=torch.nn.Parameter(p.detach().clone());o=torch.optim.AdamW([q],lr=r['lr'],betas=(.9,.95),eps=1e-8,weight_decay=r['weight_decay']);s=torch.optim.lr_scheduler.LambdaLR(o,lambda u:lr_factor(u,r));o.load_state_dict(state);s.load_state_dict(ss)
   q.grad=torch.ones_like(q);o.step();s.step()
   self.assertEqual(s.last_epoch,step+1);self.assertEqual(o.param_groups[0]['lr'],r['lr']*lr_factor(step+1,r))
   if step>=20000:self.assertEqual(o.param_groups[0]['lr'],r['decay_lr'])
if __name__=='__main__':unittest.main()
