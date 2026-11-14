"""A queued one-checkpoint evaluation still receives a real latency probe."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from resident_table1_worker import ResidentPool


def test_one_policy_qualification_has_nonempty_reference():
    class Front:
        def command(self,kind,**kw):
            assert kind=='probe' and kw['samples']==100
            return {'passed':True,'p50_ms':200.,'p95_ms':250.,'samples':100}
    pool=ResidentPool.__new__(ResidentPool)
    pool.fronts={'latest':Front()};pool.profile={}
    pool.qualify()
    assert pool.profile['gpu_residency_capacity']==1
    assert pool.profile['single_policy_only']
    assert pool.profile['single_resident']['samples']==100
    assert pool.profile['all_resident']['p95_ms']==250.
