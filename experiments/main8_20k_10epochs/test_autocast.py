"""Verify cache selection preserves BF16 checkpointed computation and gradients."""
import unittest
import torch
from torch.utils.checkpoint import checkpoint

def verify_autocast(device='cpu'):
    if str(device)=='cpu':torch.set_num_threads(2)
    # Deterministic tensors avoid perturbing the training RNG stream.
    w=torch.nn.Parameter(torch.linspace(-.1,.1,512,device=device).reshape(32,16))
    x=torch.linspace(-1,1,64,device=device).reshape(4,16).requires_grad_()
    outputs=[];grads=[]
    for enabled in [True,False]:
        w.grad=None;x.grad=None
        with torch.autocast(torch.device(device).type,dtype=torch.bfloat16,cache_enabled=enabled):
            y=checkpoint(lambda value:torch.nn.functional.linear(value,w).tanh(),x,use_reentrant=False)
            loss=y.float().square().mean()
        loss.backward();outputs.append(y.detach().float());grads.append((w.grad.detach().clone(),x.grad.detach().clone()))
    torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)
    for a,b in zip(grads[0],grads[1]):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)
    return {'cache_on_off_bf16_forward_equivalent':True,'checkpointed_gradient_equivalent':True}

class AutocastContracts(unittest.TestCase):
    def test_cache_preserves_checkpointed_computation(self):
        self.assertTrue(verify_autocast()['checkpointed_gradient_equivalent'])

if __name__=='__main__':unittest.main()
