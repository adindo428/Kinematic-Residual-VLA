"""EMA over existing trainable tensors; adds no learned parameters or inputs."""
from contextlib import contextmanager
import torch


class TrainingEMA:
    def __init__(self,model,decay=.999):
        if not 0<=decay<1:raise ValueError('EMA decay must be in [0,1)')
        self.decay=decay;self.updates=0
        self.shadow={name:p.detach().float().clone() for name,p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self,model):
        self.updates+=1;decay=min(self.decay,(1+self.updates)/(10+self.updates))
        for name,p in model.named_parameters():
            if name in self.shadow:self.shadow[name].lerp_(p.detach().float(),1-decay)

    @contextmanager
    def apply(self,model):
        previous={}
        try:
            for name,p in model.named_parameters():
                if name in self.shadow:
                    previous[name]=p.data;p.data=self.shadow[name].to(dtype=p.dtype)
            yield
        finally:
            for name,p in model.named_parameters():
                if name in previous:p.data=previous[name]

    def state_dict(self):
        return dict(decay=self.decay,updates=self.updates,shadow=self.shadow)

    def load_state_dict(self,state):
        if set(state['shadow'])!=set(self.shadow):raise ValueError('EMA trainable parameter set differs')
        self.decay=state['decay'];self.updates=state['updates']
        for name,value in state['shadow'].items():self.shadow[name].copy_(value)
