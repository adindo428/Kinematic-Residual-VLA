"""Live physical-reference handling remains independent of normalized inputs."""
import sys
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workstation'))


def test_live_delta_decode_anchors_to_original_physical_state_even_if_preprocessor_mutates(monkeypatch):
    import real_policy_worker as worker
    state=torch.tensor([[[.3,.1,.12,0.,1.,0.],[.35,.02,.09,0.,1.,0.]]])
    batch={'observation.state':state.clone()}
    delta=torch.tensor([.01,-.02,-.03,0.,0.,0.,.5]).repeat(1,20,1)
    noise=torch.ones(1,20,32)
    class Policy:
        def _prepare_batch(self,b):
            b['observation.state'].fill_(7.)
            return b
        def _get_action_chunk(self,b,noise=None):
            assert noise is not None
            return delta.clone()
    monkeypatch.setattr(worker,'prepare_live_batch',lambda *a,**k:batch)
    monkeypatch.setattr(worker.torch,'autocast',lambda *a,**k:nullcontext())
    cfg=SimpleNamespace(use_delta_action=True,euler_delta_convention='wrapped_xyz_v1')
    prediction=worker.infer(Policy(),cfg,{}, {},{},noise=noise)
    np.testing.assert_allclose(prediction[:,:3],np.tile([.36,0.,.06],(20,1)),atol=1e-7)
    np.testing.assert_allclose(prediction[:,6],.5)
    torch.testing.assert_close(noise,torch.ones_like(noise))
