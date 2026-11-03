import os
import unittest
from unittest.mock import patch
from placements import layout, memory_by_node, NAMES
from budget import validate_budget,initial_progress,advance,lr_factor
from runtime import validate_runtime

class PlacementContracts(unittest.TestCase):
    def test_four_a100_are_two_nodes_with_two_80gb_each(self):
        s=layout('a100');self.assertEqual((s['nodes'],s['local_world'],s['world']),(2,2,4));self.assertEqual(s['constraint'],'gpu_a100_80gb')
    def test_two_h200_single_node(self):
        s=layout('h200');self.assertEqual((s['nodes'],s['world']),(1,2))
    def test_per_node_cpu_budget(self):
        with patch.dict(os.environ,{'LOCAL_WORLD_SIZE':'2','SLURM_CPUS_PER_TASK':'32'}):
            validate_runtime({'microbatch':4,'workers':8,'main_threads':2},32,4)
            with self.assertRaises(ValueError):validate_runtime({'microbatch':4,'workers':12,'main_threads':8},32,4)
    def test_wrong_global_batch_rejected(self):
        with self.assertRaises(ValueError):validate_runtime({'microbatch':3,'workers':4,'main_threads':2},32,4)
    def test_memory_is_summed_within_each_node(self):
        rows=[{'hostname':h,'rss':30,'limit':100} for h in ['a','a','b','b']]
        nodes,ok=memory_by_node(rows);self.assertTrue(ok);self.assertEqual(nodes['a']['rss'],60)
    def test_one_overloaded_node_fails(self):
        _,ok=memory_by_node([{'hostname':'a','rss':86,'limit':100},{'hostname':'b','rss':1,'limit':100}]);self.assertFalse(ok)
    def test_node_allocation_disagreement_fails(self):
        with self.assertRaises(ValueError):memory_by_node([{'hostname':'a','rss':1,'limit':100},{'hostname':'a','rss':1,'limit':200}])
    def test_only_requested_four_models(self):self.assertEqual(len(NAMES),4);self.assertNotIn('smolvla_kr',NAMES)
    def test_four_rank_global_anchor_partition_is_exact(self):
        for world,micro in [(4,4),(2,16),(2,8)]:
            accum=32//(world*micro)
            anchors=[j*world*micro+rank*micro+k for j in range(accum) for rank in range(world) for k in range(micro)]
            self.assertEqual(sorted(anchors),list(range(32)))

if __name__=='__main__':unittest.main()
