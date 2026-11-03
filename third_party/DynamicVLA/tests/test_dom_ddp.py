"""Real Gloo collectives verify exact sharding, weighted gradients and resume."""
import json
from pathlib import Path
import os
import signal
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import safetensors.torch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from core.dom_distributed import RankBatchSampler
from utils.dom_experiment import atomic_json, file_digest


@pytest.mark.parametrize('world',[4,8])
@pytest.mark.parametrize('sampling',['frames','blocks'])
@pytest.mark.parametrize('lengths',[[3],[351],[301,424]])
def test_global_batches_have_no_real_duplicates_and_resume_exactly(world,sampling,lengths):
    samplers=[RankBatchSampler(lengths,22,rank,world,sampling=sampling) for rank in range(world)]
    streams=[list(s) for s in samplers]
    assert len({len(s) for s in streams})==1
    assert all(len(s)==len(stream) for s,stream in zip(samplers,streams))
    real=[index for stream in streams for batch in stream for _,index,valid,_,_ in batch if valid]
    assert sorted(real)==list(range(sum(lengths)))
    if sum(lengths)>352:
        for rank in range(world):
            resumed=list(RankBatchSampler(lengths,22,rank,world,cursor=352,sampling=sampling))
            steps=next(i+1 for i,batch in enumerate(streams[rank]) if batch[0][-1])
            assert resumed==streams[rank][steps:]


class Data(torch.utils.data.Dataset):
    def __init__(self,n):self.lengths=np.array([n]);self.n=n
    def __len__(self):return self.n
    def __getitem__(self,item):
        epoch,index=item if isinstance(item,tuple) else (0,item)
        return {'observation.state':torch.tensor([index/700,1.,index%3]),
            'action':torch.full((20,7),index/1000), 'action_is_pad':torch.tensor([False]*17+[True]*3),
            'sample_index':torch.tensor(index),'dynamic_object.phase_id':torch.tensor(index%3)}


class Policy(torch.nn.Module):
    def __init__(self,stochastic=False):
        super().__init__();self.linear=torch.nn.Linear(3,7);self.stochastic=stochastic
        self.config=SimpleNamespace(chunk_size=20,n_action_steps=20,max_action_dim=7)
    def reset(self):pass
    def forward(self,batch):
        state=batch['observation.state']
        if self.stochastic:state=state+torch.randn_like(state)*.02
        prediction=self.linear(state)[:,None,:]
        valid=~batch['action_is_pad']
        loss=(((prediction-batch['action']).square()*valid[:,:,None]).sum((1,2))/(valid.sum(1)*7)).mean()
        return loss,{}
    def predict_action_chunk(self,batch,noise=None):
        return self.linear(batch['observation.state'])[:,None,:].expand(-1,20,-1)
    def save_pretrained(self,path):
        safetensors.torch.save_model(self,str(Path(path)/'model.safetensors'))
        atomic_json(Path(path)/'config.json',vars(self.config))


def gradient_worker(rank,world,init,output):
    from core.dom_distributed_training import distributed_loader,wrap_policy,updates
    dist.init_process_group('gloo',init_method=init,rank=rank,world_size=world)
    torch.set_num_threads(1)
    try:
        data=Data(355);runtime={'microbatch':44,'workers':0,'pin_memory':False,'prefetch':2}
        torch.manual_seed(12);policy=Policy();baseline=Policy();baseline.load_state_dict(policy.state_dict())
        model=wrap_policy(policy,torch.device('cpu'))
        optimizer=torch.optim.SGD(policy.parameters(),lr=.01)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
        progress={'cursor':0,'updates':0,'samples':0}
        list(updates(model,optimizer,scheduler,distributed_loader(data,runtime),progress,'fp32',torch.device('cpu')))
        # Rebuild the identical global order and compare both optimizer updates,
        # including the uneven three-sample tail with empty ranks.
        from utils.dom_data import EpochBatchSampler
        reference=torch.optim.SGD(baseline.parameters(),lr=.01)
        for group in EpochBatchSampler(data.lengths,352):
            batch=torch.utils.data.default_collate([data[index] for index in group])
            loss,_=baseline(batch);loss.backward();reference.step();reference.zero_grad()
        for a,b in zip(policy.parameters(),baseline.parameters()):torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-7)
        assert progress['updates']==2 and progress['samples']==355 and progress['cursor']==355
        if rank==0:atomic_json(Path(output)/'gradient-result.json',progress)
    finally:dist.destroy_process_group()


def resume_worker(rank,world,init,output):
    from core import dom_distributed_training as trainer
    from core.dom_checkpoint import resolve_generation
    dist.init_process_group('gloo',init_method=init,rank=rank,world_size=world)
    torch.set_num_threads(1)
    try:
        output=Path(output)
        trainer.load_family=lambda *a,**kw:{'fingerprint':'frozen','train_frames':5}
        trainer.build_dataset=lambda root,episodes,training=False:Data(5 if training else 3)
        trainer.build_policy=lambda *a,**kw:Policy(stochastic=True)
        writer=SimpleNamespace(log=lambda *a,**kw:None,finish=lambda **kw:None)
        sys.modules['wandb']=SimpleNamespace(util=SimpleNamespace(generate_id=lambda:'krv-id'),init=lambda **kw:writer)
        trainer.enqueue_evaluations=lambda *a,**kw:None
        recipe={'precision':'fp32','lr':.001,'sampling':'frames','effective_batch':352,'epochs':100,
            'simulator_evaluation_mode':'after_training','evaluation_plan_sha256':file_digest(output/'baseline/evaluation_plan.json')}
        runtime={'precision':'fp32','microbatch':2,'workers':0,'pin_memory':False,'prefetch':2,'fused':False}
        complete=trainer.train_distributed(output/'baseline','krv100',runtime,recipe,torch.device('cpu'))
        original=trainer.updates
        def interrupted(*args,**kwargs):
            for record in original(*args,**kwargs):
                if rank==0 and args[4]['updates']==2:os.kill(os.getpid(),signal.SIGUSR1)
                yield record
        trainer.updates=interrupted
        partial=trainer.train_distributed(output/'resumed','krv100',runtime,recipe,torch.device('cpu'))
        assert partial['updates']==2 and partial['interrupted']
        trainer.updates=original
        done=trainer.train_distributed(output/'resumed','krv100',runtime,recipe,torch.device('cpu'))
        assert done['epoch']==100 and done['updates']==100 and done['samples']==500 and not done['pending_validation']
        a=resolve_generation(output/'baseline/runs/krv100');b=resolve_generation(output/'resumed/runs/krv100')
        wa=safetensors.torch.load_file(str(a/'model.safetensors'));wb=safetensors.torch.load_file(str(b/'model.safetensors'))
        for name in wa:torch.testing.assert_close(wa[name],wb[name],rtol=0,atol=0)
        state=torch.load(b/'training_state.pt',map_location='cpu',weights_only=False)
        assert state['distributed']['world_size']==world and len(state['distributed']['rng_by_rank'])==world
        assert state['wandb_id']=='krv-id'
        assert len(list((output/'resumed/runs/krv100').glob('validation-epoch-*.json')))==100
        if rank==0:atomic_json(output/'resume-result.json',{'updates':done['updates'],'samples':done['samples'],'epochs':done['epoch']})
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('world',[4,8])
def test_real_rank_weighted_gradients_match_global_batch(tmp_path,world):
    mp.spawn(gradient_worker,args=(world,'file://'+str(tmp_path/'gradient-init'),str(tmp_path)),nprocs=world,join=True)
    assert json.loads((tmp_path/'gradient-result.json').read_text())['samples']==355


def test_real_collective_checkpoint_resume_completes_100_epochs(tmp_path):
    for name in ['baseline','resumed']:
        root=tmp_path/name;root.mkdir()
        atomic_json(root/'split.json',{'splits':{'train':[0],'validation':[1],'test':[2]}})
        atomic_json(root/'validation_anchors.json',[0,1,2])
        atomic_json(root/'evaluation_plan.json',{'mode':'after_training'})
    mp.spawn(resume_worker,args=(2,'file://'+str(tmp_path/'resume-init'),str(tmp_path)),nprocs=2,join=True)
    assert json.loads((tmp_path/'resume-result.json').read_text())['epochs']==100


def stopping_worker(rank,world,init,output):
    from core import dom_distributed_training as trainer
    from core.dom_checkpoint import resolve_generation
    from tests.test_dom_stopping import RULE
    dist.init_process_group('gloo',init_method=init,rank=rank,world_size=world)
    torch.set_num_threads(1)
    try:
        root=Path(output)
        trainer.load_family=lambda *a,**kw:{'fingerprint':'frozen','train_frames':5}
        trainer.build_dataset=lambda root,episodes,training=False:Data(5 if training else 3)
        trainer.build_policy=lambda *a,**kw:Policy(stochastic=True)
        trainer.evaluate=lambda *a,**kw:{'h20/action_mse':1.}
        finishes=[]
        writer=SimpleNamespace(log=lambda *a,**kw:None,finish=lambda **kw:finishes.append(kw))
        sys.modules['wandb']=SimpleNamespace(util=SimpleNamespace(generate_id=lambda:'krv-id'),init=lambda **kw:writer)
        trainer.enqueue_evaluations=lambda *a,**kw:None
        recipe={'precision':'fp32','lr':.001,'sampling':'frames','effective_batch':352,'epochs':100,
            'early_stopping':RULE,'simulator_evaluation_mode':'after_training',
            'evaluation_plan_sha256':file_digest(root/'evaluation_plan.json')}
        runtime={'precision':'fp32','microbatch':2,'workers':0,'pin_memory':False,'prefetch':2,'fused':False}
        original=trainer.updates
        def interrupted(*args,**kwargs):
            for record in original(*args,**kwargs):
                if rank==0 and args[4]['updates']==3:os.kill(os.getpid(),signal.SIGUSR1)
                yield record
        trainer.updates=interrupted
        partial=trainer.train_distributed(root,'krv100',runtime,recipe,torch.device('cpu'))
        assert partial['interrupted'] and partial['updates']==3
        trainer.updates=original
        done=trainer.train_distributed(root,'krv100',runtime,recipe,torch.device('cpu'))
        assert done['epoch']==5 and done['updates']==5 and done['samples']==25 and done['cursor']==0
        assert done['early_stopped'] and done['completion_reason']=='early_stopping'
        assert not done['pending_validation']
        if rank==0:assert finishes[-1]['exit_code']==0
        destination=root/'runs/krv100'
        final=resolve_generation(destination,'final.json')
        state=torch.load(final/'training_state.pt',map_location='cpu',weights_only=False)
        assert state['scheduler']['last_epoch']==5 and all(int(s['step'])==5 for s in state['optimizer']['state'].values())
        assert json.loads((destination/'best.json').read_text())['progress']['epoch']==1
        assert len(list(destination.glob('validation-epoch-*.json')))==5
        assert trainer.train_distributed(root,'krv100',runtime,recipe,torch.device('cpu'))['epoch']==5
        continued=trainer.train_distributed(root,'krv100',runtime,recipe,torch.device('cpu'),resume_stopped=True)
        assert continued['epoch']==8 and continued['updates']==8 and continued['resume_stopped_epochs']==[5]
        # First stopped generation is still available after explicitly continuing.
        assert (final/'training_state.pt').exists()
        if rank==0:atomic_json(root/'stop-result.json',{'stopped_epoch':5,'continued_epoch':8,'exit_code':finishes[-1]['exit_code']})
    finally:dist.destroy_process_group()


def test_real_collective_early_stop_after_interruption_and_explicit_resume(tmp_path):
    atomic_json(tmp_path/'split.json',{'splits':{'train':[0],'validation':[1],'test':[2]}})
    atomic_json(tmp_path/'validation_anchors.json',[0,1,2])
    atomic_json(tmp_path/'evaluation_plan.json',{'mode':'after_training'})
    mp.spawn(stopping_worker,args=(2,'file://'+str(tmp_path/'stop-init'),str(tmp_path)),nprocs=2,join=True)
    assert json.loads((tmp_path/'stop-result.json').read_text())=={'stopped_epoch':5,'continued_epoch':8,'exit_code':0}


class BatchNormPolicy(Policy):
    def __init__(self):
        super().__init__()
        self.normalizer = torch.nn.BatchNorm1d(3)
    def forward(self,batch):
        return self.linear(self.normalizer(batch['observation.state'])).square().mean(),{}
    def predict_action_chunk(self,batch,noise=None):
        return self.linear(self.normalizer(batch['observation.state']))[:,None,:].expand(-1,20,-1)


def buffer_reload_worker(rank,world,init,output):
    from core.dom_distributed import rank_check, save_distributed, synchronize_checkpoint_buffers
    from core.dom_distributed_training import wrap_policy, restore_distributed
    from core.dom_checkpoint import load_generation
    dist.init_process_group('gloo',init_method=init,rank=rank,world_size=world)
    torch.set_num_threads(1)
    try:
        torch.manual_seed(42)
        policy=BatchNormPolicy()
        model=wrap_policy(policy,torch.device('cpu'))
        optimizer=torch.optim.SGD(policy.parameters(),lr=.01,momentum=.9)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
        batch={'observation.state':torch.arange(12,dtype=torch.float32).reshape(4,3)+rank*20}
        for _ in range(2):
            model(batch)[0].backward();optimizer.step();scheduler.step();optimizer.zero_grad()
        # DDP weights agree while BatchNorm statistics contain local batches.
        means=[torch.zeros(3) for _ in range(world)]
        dist.all_gather(means,policy.normalizer.running_mean)
        assert not torch.equal(means[0],means[1])
        synchronize_checkpoint_buffers(policy)
        torch.testing.assert_close(policy.normalizer.running_mean,means[0],rtol=0,atol=0)
        policy.eval()
        with torch.no_grad():prediction=policy.predict_action_chunk(batch).clone()
        torch.manual_seed(200+rank)
        progress={'epoch':0,'cursor':8,'updates':2,'samples':8}
        save_distributed(output,policy,optimizer,scheduler,progress,'buffers','test-id',device='cpu')
        expected=torch.rand(4)
        with torch.no_grad():
            policy.linear.weight.add_(1.)
            policy.normalizer.running_mean.add_(100.)
        state=load_generation(output,policy,optimizer,scheduler,'buffers',restore_random=False)
        restore_distributed(state)
        torch.testing.assert_close(expected,torch.rand(4),rtol=0,atol=0)
        with torch.no_grad():torch.testing.assert_close(prediction,policy.predict_action_chunk(batch),rtol=0,atol=0)
        assert scheduler.last_epoch==2
        # A verification failure must reach every peer without GPU collectives.
        def fail_one_rank():
            if rank==1:raise ValueError('deliberate reload failure')
        with pytest.raises(RuntimeError,match='rank 1: ValueError: deliberate reload failure'):
            rank_check(fail_one_rank)
        if rank==0:atomic_json(Path(output)/'buffer-result.json',{'strict_reload':True,'rank_rng':True,'failure_propagated':True})
    finally:dist.destroy_process_group()


def test_rank_local_batchnorm_buffers_checkpoint_and_reload_exactly(tmp_path):
    mp.spawn(buffer_reload_worker,args=(2,'file://'+str(tmp_path/'buffers-init'),str(tmp_path)),nprocs=2,join=True)
    assert json.loads((tmp_path/'buffer-result.json').read_text())=={
        'strict_reload':True,'rank_rng':True,'failure_propagated':True}
