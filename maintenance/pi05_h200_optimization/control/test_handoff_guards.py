import copy,unittest
from handoff_guards import *

class GuardContracts(unittest.TestCase):
    def setUp(self):
        self.progress={'updates':8000,'samples':256000,'epoch':0,'cursor':256000}
        self.latest={'identity':'parent','progress':self.progress}
        self.prov={'job_id':'17719993','wandb_id':'ifalqwwf','hardware':'NVIDIA A100 80GB PCIe','world_size':2,'identity':'parent'}
        self.recipe={'steps':30000,'global_batch':32}
        self.c={'model':{'trusted':'weights'},'optimizer':{'state':{0:{'step':8000}},'param_groups':[{'betas':(.9,.95),'eps':1e-8,'weight_decay':.01,'initial_lr':2.5e-5}]},'scheduler':{'last_epoch':8000},'rng':[{k:k for k in ('python','numpy','torch','cuda')} for _ in range(2)],'progress':self.progress,'identity':'parent'}
    def test_no_parent_action_while_queued(self):
        with self.assertRaises(ValueError):parent_decision('pi05_baseline','RUNNING',self.prov,self.latest,None,False)
    def test_wrong_parent_or_wandb_refused(self):
        for k,v in [('job_id','17668718'),('wandb_id','other'),('hardware','NVIDIA H200')]:
            p={**self.prov,k:v}
            with self.assertRaises(ValueError):parent_decision('pi05_baseline','RUNNING',p,self.latest,None,True)
    def test_running_parent_requests_checkpoint(self):
        self.assertEqual(parent_decision('pi05_baseline','RUNNING',self.prov,self.latest,None,True),'checkpoint_running_parent')
    def test_completed_budget_never_repeated(self):
        p={'updates':30000,'samples':960000,'epoch':0,'cursor':960000};latest={'identity':'parent','progress':p};done={'progress':p,'stop_reason':'official_step_budget'}
        self.assertEqual(parent_decision('pi05_baseline','RUNNING',self.prov,latest,done,True),'release_target_without_training')
    def test_hardware_and_identity_before_signal(self):
        good=[dict(name='NVIDIA H200',total_memory=150109880320,cuda_usable=True)]*2
        self.assertTrue(validate_allocation(good,'new','new',True,True))
        for args in [(good,'other','new',True,True),(good[:1],'new','new',True,True),(good,'new','new',False,True)]:
            with self.assertRaises(ValueError):validate_allocation(*args)
    def test_stale_stop_and_autorequeue_fail_closed(self):
        stopped={'reason':'allocation_signal','progress':self.progress}
        self.assertTrue(validate_saved_stop(self.latest,stopped,self.c,1,2,2,('start',0),('start',0)))
        for args in [(self.latest,stopped,self.c,3,2,2,('start',0),('start',0)),(self.latest,stopped,self.c,1,2,2,('start',0),('newstart',1))]:
            with self.assertRaises(ValueError):validate_saved_stop(*args)
    def test_checkpoint_requires_optimizer_scheduler_rng_cursor(self):
        self.assertTrue(validate_checkpoint(self.c,self.latest,self.recipe))
        for key in ('optimizer','scheduler','rng'):
            c=copy.deepcopy(self.c);c.pop(key)
            with self.assertRaises(ValueError):validate_checkpoint(c,self.latest,self.recipe)
        c=copy.deepcopy(self.c);c['scheduler']['last_epoch']=0
        with self.assertRaises(ValueError):validate_checkpoint(c,self.latest,self.recipe)
    def test_exact_budget_manifest_and_parent_terminal(self):
        source={n:n for n in CORE};args=('pi05_baseline','/immutable/latest.pt','a'*64,self.latest,self.recipe,{'complete':True},source,source,'2.7.1+cu126','NVIDIA H200',{'id':'ifalqwwf'},True)
        m=migration_manifest(*args);self.assertEqual(m['progress'],self.progress);self.assertEqual(m['recipe_fingerprint'],fingerprint(self.recipe))
        bad=list(args);bad[-1]=False
        with self.assertRaises(ValueError):migration_manifest(*bad)
    def test_pair_requires_both_sustained_receipts_and_accounting(self):
        observations=[{'Train/optimizer_updates':10*(i+1),'Train/samples_per_second':10.0} for i in range(20)];source={'native':'unchanged'}
        approval=dict(approved=True,source=source,hardware='NVIDIA H200',world_size=2,decoded_optimizer_updates=200,global_batch=32,strict_checkpoint_rng_reload=True,strict_inference_reload=True,coordinated_checkpoint_test=True,includes_intermediate_checkpoint_in_timed_training=True,peak_gpu_reserved_bytes=60,total_gpu_bytes=150,node_process_peak_rss_bytes=117,memory_limit_bytes=274,identity='approved',samples_per_second=25.0)
        comparison=dict(identity='approved',h200_sustained_samples_per_second=25.0,a100_observations=observations,a100_production_median_samples_per_second=10.0,ratio_to_a100=2.5,gpu_profile_sha256='a'*64)
        a={n:copy.deepcopy(approval) for n in NAMES};c={n:copy.deepcopy(comparison) for n in NAMES};complete={'complete':True,'job_id':'17740320','profiles':a};accounting={'state':'COMPLETED','exit_code':'0:0'}
        self.assertTrue(validate_pair(a,c,accounting,complete,source))
        with self.assertRaises(ValueError):validate_pair(a,c,{'state':'RUNNING','exit_code':'0:0'},complete,source)
        with self.assertRaises(ValueError):validate_pair({'pi05_baseline':approval},c,accounting,complete,source)
        c['pi05_kr']['ratio_to_a100']=1.1
        with self.assertRaises(ValueError):validate_pair(a,c,accounting,complete,source)

if __name__=='__main__':unittest.main()
