import unittest
import torch
from common import coordinates, drift, masked_loss
from data import Batches, reference_state


class Contract(unittest.TestCase):
    def test_global_optimizer_anchor_sets_survive_microbatch_change(self):
        def steps(micro):
            batches=[list(Batches(192,0,0,64,micro,rank,2)) for rank in range(2)]
            accumulation=64//(micro*2)
            return [sorted(i for rank in batches for batch in rank[step*accumulation:(step+1)*accumulation] for _,i in batch) for step in range(3)]
        for micro in [4,8,16,32]:self.assertEqual(steps(1),steps(micro))

    def test_microbatch_objective_and_gradient_invariance_with_padding(self):
        losses=torch.randn(5,20,32,requires_grad=True)
        padding=torch.arange(20)[None,:]>=torch.tensor([1,3,8,14,20])[:,None]
        expected=torch.stack([masked_loss(losses[i:i+1],padding[i:i+1],32) for i in range(5)]).mean()
        actual=masked_loss(losses,padding,32)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-7))
        a=torch.autograd.grad(actual,losses,retain_graph=True)[0]
        b=torch.autograd.grad(expected,losses)[0]
        self.assertTrue(torch.allclose(a,b,atol=1e-7))

    def test_runtime_cannot_change_effective_batch_or_oversubscribe_cpus(self):
        from runtime import validate_runtime
        validate_runtime(dict(microbatch=16,workers=4,main_threads=4),64,2)
        for runtime in [dict(microbatch=3,workers=4,main_threads=4),
                        dict(microbatch=16,workers=12,main_threads=8)]:
            with self.assertRaises(ValueError):validate_runtime(runtime,64,2)

    def test_loader_does_not_consume_training_rng(self):
        from train import loader
        class Tiny(torch.utils.data.Dataset):
            def __len__(self):return 100
            def __getitem__(self,x):return x[1]
        state=torch.get_rng_state().clone()
        batch=next(iter(loader(Tiny(),2,32,0,2,workers=0)))
        self.assertEqual(len(batch),2)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_native_padded_channel_loss_option(self):
        losses=torch.zeros(1,20,32);losses[...,7:]=1
        padding=torch.zeros(1,20,dtype=torch.bool)
        self.assertEqual(float(masked_loss(losses,padding,7)),0.)
        self.assertAlmostEqual(float(masked_loss(losses,padding,32)),25/32)

    def test_dom_rollforward_preserves_pose6(self):
        import numpy as np
        state=np.ones((5,6));action=np.zeros((5,7));action[:,6]=1234
        for delay in range(4):
            x=reference_state(state,action,3,delay)
            self.assertEqual(x.shape,(6,))
            self.assertFalse((x==1234).any())
    def test_physical_roundtrip_and_channels(self):
        raw = torch.randn(2,20,7)
        velocity = torch.tensor([[1.,2.,3.],[0.,0.,0.]])
        d = drift(velocity,torch.zeros(3),20,torch.tensor([0,3]))
        self.assertTrue(torch.allclose(d[0,1,:3],velocity[0]*.04))
        self.assertTrue(torch.equal(d[...,3:],torch.zeros_like(d[...,3:])))
        self.assertTrue(torch.allclose(coordinates(coordinates(raw,d),d,True),raw,atol=1e-6))

    def test_vlash_offset_uses_observation_time(self):
        d=drift(torch.tensor([[1.,0.,0.]]),torch.zeros(3),20,torch.tensor([3]))
        self.assertAlmostEqual(float(d[0,0,0]),.12,places=6)
        self.assertAlmostEqual(float(d[0,19,0]),.88,places=6)

    def test_centering_zero_is_identity(self):
        self.assertTrue(torch.equal(drift(torch.ones(2,3),torch.ones(3),20),torch.zeros(2,20,7)))

    def test_pad_mask_ignores_invalid_targets_and_padding_dims(self):
        losses=torch.ones(2,20,32,requires_grad=True)
        padding=torch.zeros(2,20,dtype=torch.bool);padding[:,-5:]=True
        value=masked_loss(losses,padding);value.backward()
        self.assertAlmostEqual(value.item(),1.)
        self.assertTrue(torch.equal(losses.grad[:,-5:],torch.zeros_like(losses.grad[:,-5:])))
        self.assertTrue(torch.equal(losses.grad[...,7:],torch.zeros_like(losses.grad[...,7:])))

    def test_ddp_shards_and_resume(self):
        a=list(Batches(100,0,0,32,2,0,2));b=list(Batches(100,0,0,32,2,1,2))
        flat_a=[i for batch in a for _,i in batch];flat_b=[i for batch in b for _,i in batch]
        self.assertFalse(set(flat_a)&set(flat_b))
        self.assertEqual(len(set(flat_a+flat_b)),96)
        self.assertEqual(list(Batches(100,0,32,32,2,0,2)),a[8:])


if __name__=='__main__':unittest.main()
