"""Compare native fused AdamW with the unfused equations and strict continuation."""
import copy
import io
import unittest
import torch

def verify_fused(device='cpu'):
    if str(device)=='cpu':torch.set_num_threads(2)
    initial=torch.linspace(-1,1,257,device=device)
    p=torch.nn.Parameter(initial.clone());q=torch.nn.Parameter(initial.clone())
    options=dict(lr=2.5e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    ref=torch.optim.AdamW([p],foreach=False,**options)
    fused=torch.optim.AdamW([q],fused=True,**options)
    sched=torch.optim.lr_scheduler.LambdaLR(fused,lambda step:(step+1)/1000)
    # Match the scheduler-controlled learning rate for the reference path.
    ref_sched=torch.optim.lr_scheduler.LambdaLR(ref,lambda step:(step+1)/1000)
    for i in range(12):
        gradient=torch.sin(initial*(i+1))+.1
        p.grad=gradient.clone();q.grad=gradient.clone()
        ref.step();fused.step();ref_sched.step();sched.step()
    torch.testing.assert_close(p,q,rtol=1e-6,atol=1e-7)
    for k in ['exp_avg','exp_avg_sq']:
        torch.testing.assert_close(ref.state[p][k],fused.state[q][k],rtol=1e-5,atol=1e-7)
    buffer=io.BytesIO();torch.save({'model':q.detach(),'optimizer':fused.state_dict(),'scheduler':sched.state_dict()},buffer)
    buffer.seek(0);state=torch.load(buffer,map_location=device,weights_only=False)
    restored=torch.nn.Parameter(state['model'].clone());opt=torch.optim.AdamW([restored],fused=True,**options)
    schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:(step+1)/1000)
    opt.load_state_dict(state['optimizer']);schedule.load_state_dict(state['scheduler'])
    assert opt.param_groups[0]['fused'] is True
    gradient=torch.cos(initial);q.grad=gradient.clone();restored.grad=gradient.clone()
    fused.step();opt.step();sched.step();schedule.step()
    torch.testing.assert_close(q,restored,rtol=0,atol=0)
    assert sched.state_dict()==schedule.state_dict()
    assert int(opt.state[restored]['step'])==13
    return {'device':str(device),'fused_vs_reference':True,'optimizer_scheduler_checkpoint_continuation':True}

class OptimizerContracts(unittest.TestCase):
    def test_cpu_fused_and_resume(self):
        self.assertTrue(verify_fused()['fused_vs_reference'])

if __name__=='__main__':unittest.main()
