"""Run the production loop on small data, including interruption at validation."""
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import safetensors.torch
import torch

from core import dom_training, dom_profiling
from core.dom_checkpoint import resolve_generation
from utils.dom_experiment import atomic_json, file_digest


class Data(torch.utils.data.Dataset):
    def __init__(self, training):
        self.lengths=np.array([301,424] if training else [100])
    def __len__(self):return int(self.lengths.sum())
    def __getitem__(self, item):
        epoch,idx=item if isinstance(item,tuple) else (0,item)
        state=torch.tensor([idx/725,idx%3,1.])
        actions=torch.ones(20,7)*idx/1000
        mask=torch.zeros(20,dtype=torch.bool)
        if idx%5==0:mask[17:]=True
        return {'observation.state':state,'action':actions,'action_is_pad':mask,
            'sample_index':torch.tensor(idx),'dynamic_object.phase_id':torch.tensor(idx%3)}


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.projection=torch.nn.Linear(3,7)
        self.config=SimpleNamespace(chunk_size=20,n_action_steps=20,max_action_dim=7)
    def reset(self):pass
    def forward(self,batch):
        prediction=self.projection(batch['observation.state']+torch.randn_like(batch['observation.state'])*.02)
        error=(prediction[:,None,:]-batch['action']).square()
        valid=~batch['action_is_pad']
        return ((error*valid[:,:,None]).sum((1,2))/(valid.sum(1)*7)).mean(),{}
    def predict_action_chunk(self,batch,noise=None):
        return self.projection(batch['observation.state'])[:,None,:].expand(-1,20,-1)+noise*.001
    def save_pretrained(self,path):
        Path(path).mkdir(parents=True,exist_ok=True)
        safetensors.torch.save_model(self,str(Path(path)/'model.safetensors'))
        atomic_json(Path(path)/'config.json',vars(self.config))


@pytest.mark.parametrize('interruption',['update','validation'])
@pytest.mark.parametrize('deferred',[False,True])
def test_actual_100_epoch_loop_resumes_updates_and_pending_validation(tmp_path,monkeypatch,interruption,deferred):
    monkeypatch.setattr(dom_training,'load_family',lambda *a,**kw:{'fingerprint':'frozen','train_frames':725})
    monkeypatch.setattr(dom_training,'build_dataset',lambda root,episodes,training=False:Data(training))
    monkeypatch.setattr(dom_training,'build_policy',lambda *a,**kw:Policy())
    monkeypatch.setattr(dom_training,'to_device',lambda batch,device='cuda':batch)
    fixed_noise=dom_training.fixed_noise
    monkeypatch.setattr(dom_training,'fixed_noise',lambda batch,policy,device:fixed_noise(batch,policy,'cpu'))
    evaluate=dom_training.evaluate
    phase={'interrupt':False,'fired':False}
    def evaluate_cpu(policy,loader,precision,device='cuda',should_stop=None):
        if phase['interrupt'] and interruption=='validation' and not phase['fired']:
            phase['fired']=True
            os.kill(os.getpid(),signal.SIGUSR1)
        return evaluate(policy,loader,precision,device='cpu',should_stop=should_stop)
    monkeypatch.setattr(dom_training,'evaluate',evaluate_cpu)
    check_gradients=dom_profiling.check_gradients
    calls={'count':0}
    def checked(policy):
        calls['count']+=1
        if phase['interrupt'] and interruption=='update' and calls['count']==2:
            os.kill(os.getpid(),signal.SIGUSR1);phase['fired']=True
        return check_gradients(policy)
    monkeypatch.setattr(dom_profiling,'check_gradients',checked)
    from scripts import dom_closed_loop
    submitted=[]
    monkeypatch.setattr(dom_closed_loop,'dispatch',lambda root,request:submitted.append(str(request)))
    writer=SimpleNamespace(log=lambda *a,**kw:None,finish=lambda **kw:None)
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(util=SimpleNamespace(generate_id=lambda:'short-id'),init=lambda **kw:writer))
    runtime={'precision':'fp32','microbatch':44,'workers':0,'pin_memory':False,'prefetch':2,'fused':False}
    recipe={'precision':'fp32','lr':.001,'sampling':'blocks'}
    def family(name):
        root=tmp_path/name;root.mkdir()
        atomic_json(root/'split.json',{'splits':{'train':[0,1],'validation':[2],'test':[3]}})
        atomic_json(root/'validation_anchors.json',list(range(100)))
        if deferred:atomic_json(root/'evaluation_plan.json',{'mode':'after_training'})
        return root
    baseline=family('all');resumed=family('resumed')
    if deferred:recipe.update(simulator_evaluation_mode='after_training',
        evaluation_plan_sha256=file_digest(baseline/'evaluation_plan.json'))
    complete=dom_training.train(baseline,'krvp100',runtime,recipe)
    assert complete['epoch']==100 and complete['updates']==300 and complete['samples']==72500
    phase['interrupt']=True;calls['count']=0
    partial=dom_training.train(resumed,'krvp100',runtime,recipe)
    assert partial['interrupted'] and phase['fired']
    if interruption=='validation':
        assert partial['epoch']==1 and partial['cursor']==0 and partial['pending_validation']
    else:
        assert partial['updates']==2 and partial['cursor']==704
    phase['interrupt']=False
    done=dom_training.train(resumed,'krvp100',runtime,recipe)
    assert done['epoch']==100 and done['updates']==300 and done['samples']==72500
    assert not done['pending_validation']
    a=resolve_generation(baseline/'runs/krvp100');b=resolve_generation(resumed/'runs/krvp100')
    weights_a=safetensors.torch.load_file(str(a/'model.safetensors'))
    weights_b=safetensors.torch.load_file(str(b/'model.safetensors'))
    for name in weights_a:
        torch.testing.assert_close(weights_a[name],weights_b[name],rtol=0,atol=0)
    state=torch.load(b/'training_state.pt',map_location='cpu',weights_only=False)
    assert state['wandb_id']=='short-id' and state['optimizer']['state']
    assert len(list((resumed/'runs/krvp100').glob('validation-epoch-*.json')))==100
    assert len(list((resumed/'runs/krvp100').glob('evaluation-epoch-*.json')))==12
    assert (resumed/'runs/krvp100/evaluation-test-best.json').exists()
    assert (resumed/'runs/krvp100/evaluation-test-final.json').exists()
    assert not submitted if deferred else submitted
