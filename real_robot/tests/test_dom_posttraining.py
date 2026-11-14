from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'third_party/DynamicVLA'))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workstation'))
from dom_posttraining import EpochWeightedSampler,lr_factor,accumulation_size
from utils.dom_data import DOMData


def test_accumulation_matches_a_full_batch_including_partial_final_window():
    inputs=torch.arange(1.,12.);targets=inputs*.3
    a=torch.tensor(.7,requires_grad=True);b=a.detach().clone().requires_grad_(True)
    for start in (0,8):
        a.grad=None;b.grad=None
        ((a*inputs[start:start+8]-targets[start:start+8])**2).mean().backward()
        # Two microbatches of four, including a final microbatch of three.
        for index in range(start//4,min(start//4+2,3)):
            x=inputs[index*4:index*4+4];y=targets[index*4:index*4+4]
            loss=(b*x-y).square().mean()
            (loss*len(x)/accumulation_size(index,4,2,len(inputs))).backward()
        torch.testing.assert_close(a.grad,b.grad)


def test_epoch_sampling_and_cosine_schedule_are_reproducible():
    sampler=EpochWeightedSampler(np.ones(50));sampler.epoch=3
    first=list(sampler);assert list(sampler)==first and all(epoch==3 for epoch,index in first)
    sampler.epoch=4;assert list(sampler)!=first
    assert lr_factor(0,100,10)==pytest.approx(.1)
    assert lr_factor(9,100,10)==1 and lr_factor(10,100,10)==1
    assert lr_factor(100,100,10)==pytest.approx(.1)


class Base:
    episodes=[0];meta=SimpleNamespace(fps=25)
    episode_data_index={'length':torch.tensor([20])}
    quality_indices=None
    def __init__(self):
        self.states=torch.zeros(20,6);self.states[:,2]=torch.arange(20)*.004
        self.env=torch.zeros(20,9);self.env[:,2]=self.states[:,2]+.1;self.env[:,8]=.1
        self.actions=torch.zeros(20,7);self.actions[:,-1]=-1
    def __getitem__(self,index):
        return {'dynamic_object.velocity_required':torch.tensor([True,True]),'episode_index':torch.tensor(0),'frame_index':torch.tensor(index)}
    def _get_episode(self,ep):
        return {'observation.state':self.states,'observation.environment_state':self.env,'action':self.actions,'timestamp':torch.arange(20)*.04}
    def __len__(self):return 20


def test_real_quality_flags_cannot_bypass_requested_causal_replay():
    constant=DOMData(Base(),phase_gate_mode='constant')[8]['dynamic_object.phase_gate']
    base=Base();causal=DOMData(base,phase_gate_mode='causal_rules')[8]['dynamic_object.phase_gate']
    assert constant.min()==1 and causal[-1]==0 and causal[0]<1
    # Commands and observations after the current anchor cannot alter its gate.
    base=Base();base.actions[8:,-1]=1;base.env[9:,:3]+=1
    alternate=DOMData(base,phase_gate_mode='causal_rules')[8]['dynamic_object.phase_gate']
    torch.testing.assert_close(alternate,causal)
