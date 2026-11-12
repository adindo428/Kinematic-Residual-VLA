import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'workstation'))
from velocity_conditioning import VELOCITY_PROMPT,velocity_prompt_batch,UntruncatedTokenizer
from family_policy import NativeTrainingPolicy,signature
from real_robot.action_objective import physical_errors


def test_velocity_prompt_is_current_quality_gated_and_never_mutates_targets():
    batch=dict(task=['grasp the ball.','grasp the ball.'],velocity=torch.tensor([[.1254,-.0106,0.],[2.,3.,4.]]),
        velocity_required=torch.tensor([True,False]),action=torch.zeros(2,20,7),
        future_velocity=torch.full((2,20,3),999.))
    actual=velocity_prompt_batch(batch,VELOCITY_PROMPT)
    assert actual['task']==['Object base velocity mm/s: 125,-11,0; grasp the ball.',
                            'Object base velocity mm/s: 0,0,0; grasp the ball.']
    assert actual['action'] is batch['action'] and batch['task'][0]=='grasp the ball.'
    assert velocity_prompt_batch(batch,None) is batch
    with pytest.raises(ValueError):velocity_prompt_batch(dict(batch,velocity=torch.zeros(2,20,3)),VELOCITY_PROMPT)
    with pytest.raises(ValueError):velocity_prompt_batch(dict(batch,velocity=torch.full((2,3),float('nan'))),VELOCITY_PROMPT)


def test_tokenizer_rejects_lost_instruction_or_velocity():
    class Tokenizer:
        def __call__(self,texts,**kwargs):
            return {'input_ids':[list(range(len(t))) for t in texts]}
    tok=UntruncatedTokenizer(Tokenizer())
    assert tok(['abc'],max_length=3)['input_ids']==[[0,1,2]]
    with pytest.raises(ValueError):tok(['abcd'],max_length=3)


@pytest.mark.parametrize('method',['smolvla','vlash','pi05'])
@pytest.mark.parametrize('kr',[False,True])
def test_every_family_pair_passes_velocity_to_native_neural_preprocessing_without_new_parameters(method,kr):
    class Native(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.ones(2));self.method=method;self.recipe={'kr':kr}
        def prepare(self,batch):
            self.seen=batch
            return {'action':batch['action']},torch.zeros_like(batch['action'])
        def affine(self,actions,*args,**kwargs):return actions
    native=Native();before=signature(native)
    model=NativeTrainingPolicy(native,before,None,VELOCITY_PROMPT)
    batch=dict(task=['pick up the ball.'],velocity=torch.tensor([[.1,0.,0.]]),
               velocity_required=torch.tensor([True]),action=torch.zeros(1,20,7))
    model.prepare(batch)
    assert native.seen['task'][0].startswith('Object base velocity mm/s: 100,0,0;')
    assert signature(native)==before
    batch['velocity'][0,0]=-.1;model.prepare(batch)
    assert native.seen['task'][0].startswith('Object base velocity mm/s: -100,0,0;')


def test_physical_metrics_match_frozen_v6_fixture():
    fixture=json.loads((PROJECT/'post_training/metric-fixture.json').read_text())
    actual=physical_errors(torch.tensor(fixture['prediction']),torch.tensor(fixture['target']))
    assert set(actual)==set(fixture['expected'])
    for key,expected in fixture['expected'].items():
        torch.testing.assert_close(actual[key],torch.tensor(expected),atol=1e-7,rtol=1e-6)
