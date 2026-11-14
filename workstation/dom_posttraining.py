"""Shared deterministic data and optimization contracts for fresh DOM post-training."""
import json
import math
from pathlib import Path
import torch
from utils.dom_data import DOMData


class PostTrainingData(DOMData):
    def __init__(self,base,**kwargs):
        super().__init__(base,**kwargs)
        self.instructions={}
        for row in map(json.loads,(Path(base.root)/'meta/episodes.jsonl').read_text().splitlines()):
            if len(row['tasks'])!=1:raise ValueError('Expected one recorded task per episode')
            task=row['tasks'][0];task=json.loads(task) if isinstance(task,str) else task
            self.instructions[row['episode_index']]=task['instruction']
    def __getitem__(self,index):
        result=super().__getitem__(index)
        result['task']=self.instructions[int(result['episode_index'])]
        return result


class EpochWeightedSampler(torch.utils.data.Sampler):
    def __init__(self,weights,seed=42):self.weights=torch.as_tensor(weights,dtype=torch.double);self.seed=seed;self.epoch=0
    def __len__(self):return len(self.weights)
    def __iter__(self):
        indices=torch.multinomial(self.weights,len(self),True,generator=torch.Generator().manual_seed(self.seed+self.epoch))
        return iter((self.epoch,int(index)) for index in indices)


def optimizer_groups(model,lr,backbone_lr,weight_decay):
    groups={}
    for name,param in model.named_parameters():
        if not param.requires_grad:continue
        backbone='.vlm_with_expert.vlm.' in name
        decay=0. if param.ndim<2 or 'norm' in name.lower() else weight_decay
        key=(backbone,decay)
        groups.setdefault(key,dict(params=[],lr=backbone_lr if backbone else lr,weight_decay=decay,group_name=('backbone' if backbone else 'action')+('-decay' if decay else '-no-decay')))['params'].append(param)
    if not any(key[0] for key in groups):raise ValueError('Backbone parameter group did not match native architecture')
    return list(groups.values())


def lr_factor(step,total,warmup,min_ratio=.1):
    if step<warmup:return (step+1)/max(1,warmup)
    progress=min(1.,max(0.,(step-warmup)/max(1,total-warmup)))
    return min_ratio+(1-min_ratio)*.5*(1+math.cos(math.pi*progress))


def accumulation_size(batch_index,microbatch,accumulation,total_samples):
    start=(batch_index//accumulation)*accumulation*microbatch
    return min(accumulation*microbatch,total_samples-start)
