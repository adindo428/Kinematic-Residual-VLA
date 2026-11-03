import unittest,copy
from checkpoint_contract import validate
class Contracts(unittest.TestCase):
 def setUp(self):
  self.r={'steps':12,'parent_updates':2,'parent_samples':100,'global_batch':32,'training_anchors':35}
  self.c={'identity':'a','progress':{'updates':3,'samples':132,'epoch':1,'cursor':0},'scheduler':{'last_epoch':3},'rng':[{k:1 for k in ('python','numpy','torch','cuda')}]*2,'model':{'a':1},'optimizer':{'state':{1:1},'param_groups':[1]}}
 def runcheck(self,c=None):
  c=c or self.c;return validate(c,{'identity':'a','progress':copy.deepcopy(c['progress'])},self.r,{'identity':'a'})
 def test_resume(self):self.assertFalse(self.runcheck())
 def test_done(self):
  self.c['progress'].update(updates=12,samples=420,epoch=10);self.c['scheduler']['last_epoch']=12;self.assertTrue(self.runcheck())
 def test_identity(self):
  self.c['identity']='b'
  with self.assertRaises(AssertionError):self.runcheck()
 def test_scheduler(self):
  self.c['scheduler']['last_epoch']=0
  with self.assertRaises(AssertionError):self.runcheck()
 def test_rng(self):
  self.c['rng']=[]
  with self.assertRaises(AssertionError):self.runcheck()
 def test_cursor(self):
  self.c['progress']['cursor']=1
  with self.assertRaises(AssertionError):self.runcheck()
 def test_samples(self):
  self.c['progress']['samples']=0
  with self.assertRaises(AssertionError):self.runcheck()
 def test_optimizer(self):
  self.c['optimizer']['state']={}
  with self.assertRaises(AssertionError):self.runcheck()
if __name__=='__main__':unittest.main()
