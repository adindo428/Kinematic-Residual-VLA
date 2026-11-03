import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import handoff
from test_handoff_guards import GuardContracts

class ControllerTests(unittest.TestCase):
    def test_purged_queue_is_absent_but_access_errors_fail_closed(self):
        with patch.object(handoff,'command',return_value=Mock(returncode=1,stderr='Invalid job id specified',stdout='')):self.assertFalse(handoff.visible('1'))
        with patch.object(handoff,'command',return_value=Mock(returncode=1,stderr='Unable to contact slurm controller',stdout='')):
            with self.assertRaises(ValueError):handoff.visible('1')
    def test_accounting_lag_waits_after_queue_disappearance(self):
        with patch.object(handoff,'accounting',side_effect=[{'state':'RUNNING','exit_code':'0:0'},{'state':'CANCELLED by 432600','exit_code':'0:0'}]),patch.object(handoff.time,'sleep') as sleep:
            result=handoff.terminal_accounting('17719993',handoff.time.monotonic()+60)
            self.assertTrue(result['state'].startswith('CANCELLED'));sleep.assert_called_once_with(2)
    def test_allocated_parent_restart_is_detectable(self):
        self.assertNotEqual(handoff.allocation_key(dict(StartTime='a',Restarts='0',BatchHost='one')),handoff.allocation_key(dict(StartTime='b',Restarts='1',BatchHost='two')))
    def test_trusted_synthetic_checkpoint_contains_exact_resume_state(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            p=torch.nn.Parameter(torch.ones(3));o=torch.optim.AdamW([p],lr=2.5e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01);s=torch.optim.lr_scheduler.LambdaLR(o,lambda step:(step+1)/1000)
            for _ in range(2):o.zero_grad();p.sum().backward();o.step();s.step()
            progress={'updates':2,'samples':64,'epoch':0,'cursor':64};rng=[{'python':1,'numpy':2,'torch':torch.get_rng_state(),'cuda':torch.get_rng_state()} for _ in range(2)];c={'model':{'p':p.detach()},'optimizer':o.state_dict(),'scheduler':s.state_dict(),'rng':rng,'progress':progress,'identity':'trusted'};path=Path(tmp)/'latest.pt';torch.save(c,path);meta={'identity':'trusted','progress':progress}
            handoff.inspect_checkpoint(path,meta,dict(steps=30000,global_batch=32))
            c['scheduler']['last_epoch']=0;torch.save(c,path)
            with self.assertRaises(ValueError):handoff.inspect_checkpoint(path,meta,dict(steps=30000,global_batch=32))
    def test_pending_autorequeue_cancel_requires_valid_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);out=base/'runs/pi05_baseline';out.mkdir(parents=True);root=base/'target';(root/'recipes').mkdir(parents=True);(root/'recipes/pi05_baseline.json').write_text(json.dumps(dict(steps=30000,global_batch=32)));latest={'identity':'parent','progress':{'updates':7000,'samples':224000,'epoch':0,'cursor':224000}};(out/'latest.json').write_text(json.dumps(latest));(out/'stopped.json').write_text(json.dumps({'reason':'allocation_signal','progress':latest['progress']}));folder=base/'control';folder.mkdir();config={'root':str(root),'parent_root':str(base),'runs':{'pi05_baseline':{'parent_script':'exact'}}};job=dict(JobState='PENDING',Command='exact')
            with patch.object(handoff,'live_job',return_value=job),patch.object(handoff,'inspect_checkpoint',side_effect=ValueError('missing state')),patch.object(handoff,'command') as cmd:
                with self.assertRaises(ValueError):handoff.stop_parent(config,'pi05_baseline',folder)
                cmd.assert_not_called()
            with patch.object(handoff,'live_job',return_value=job),patch.object(handoff,'inspect_checkpoint'),patch.object(handoff,'command') as cmd,patch.object(handoff,'visible',return_value=False),patch.object(handoff,'accounting',return_value={'state':'CANCELLED','exit_code':'0:0'}):
                handoff.stop_parent(config,'pi05_baseline',folder);cmd.assert_called_once_with(['scancel','17719993']);self.assertTrue((folder/'parent-terminal.json').exists())

if __name__=='__main__':unittest.main()
