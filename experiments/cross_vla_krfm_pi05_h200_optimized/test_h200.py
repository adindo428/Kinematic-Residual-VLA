import copy
import unittest
from runtime import validate_h200,validate_runtime
from data import Batches
from common import fingerprint
from migration import validate_migration,validate_release,PARENT_SOURCE

class H200Contracts(unittest.TestCase):
    def test_hardware_and_global32_anchor_sets(self):
        validate_h200('NVIDIA H200',140*1024**3,2)
        for name,mem,world in [('NVIDIA A100 80GB PCIe',80*1024**3,2),('NVIDIA H200',80*1024**3,2),('NVIDIA H200',140*1024**3,4)]:
            with self.assertRaises(ValueError):validate_h200(name,mem,world)
        def updates(micro,cursor=64):
            rank=[list(Batches(224,0,cursor,32,micro,r,2)) for r in range(2)]
            accum=32//(micro*2)
            return [sorted(i for batches in rank for batch in batches[s*accum:(s+1)*accum] for _,i in batch) for s in range(5)]
        for micro in [2,4,8,16]:
            validate_runtime(dict(microbatch=micro,workers=12,main_threads=2),32,2)
            self.assertEqual(updates(1),updates(micro))

    def test_cross_hardware_migration_identity_is_explicit(self):
        recipe=dict(steps=30000,global_batch=32);prepared={'complete':True};source={n:n for n in ['common.py','data.py','policy.py','instruction_generator.py']}
        m=dict(latest_complete_production_checkpoint=True,user_authorized_cross_hardware=True,recipe_fingerprint=fingerprint(recipe),preparation=prepared,parent_source_sha256=PARENT_SOURCE,parent_hardware='NVIDIA A100 80GB PCIe',target_hardware='NVIDIA H200',world_size=2,torch_version='2.7.1',parent_source=source,progress=dict(updates=5000,samples=160000))
        validate_migration(m,recipe,prepared,'NVIDIA H200',2,source,'2.7.1')
        for key,value in [('latest_complete_production_checkpoint',False),('user_authorized_cross_hardware',False),('parent_source_sha256','different'),('world_size',4),('target_hardware','NVIDIA A100'),('progress',dict(updates=0,samples=0))]:
            bad=copy.deepcopy(m);bad[key]=value
            with self.assertRaises(ValueError):validate_migration(bad,recipe,prepared,'NVIDIA H200',2,source,'2.7.1')

    def test_production_release_requires_sustained_speed_and_headroom(self):
        approval=dict(identity='identity',decoded_optimizer_updates=200,strict_checkpoint_rng_reload=True,strict_inference_reload=True,coordinated_checkpoint_test=True,peak_gpu_reserved_bytes=80,total_gpu_bytes=100,node_process_peak_rss_bytes=80,memory_limit_bytes=100)
        release=dict(approved=True,paired_migration_authorized=True,gpu_profile_sha256='sha',identity='identity',throughput_ratio_to_a100=1.3)
        validate_release(release,approval,'sha')
        for key,value in [('gpu_profile_sha256','bad'),('throughput_ratio_to_a100',1.24),('paired_migration_authorized',False)]:
            bad=copy.deepcopy(release);bad[key]=value
            with self.assertRaises(ValueError):validate_release(bad,approval,'sha')
        for key,value in [('decoded_optimizer_updates',20),('peak_gpu_reserved_bytes',86),('node_process_peak_rss_bytes',86)]:
            bad=copy.deepcopy(approval);bad[key]=value
            with self.assertRaises(ValueError):validate_release(release,bad,'sha')

if __name__=='__main__':unittest.main()
