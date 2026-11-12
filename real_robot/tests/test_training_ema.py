import torch
import pytest
from real_robot.training_ema import TrainingEMA


def test_ema_preserves_architecture_frozen_weights_and_restores_after_exception():
    model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Linear(2,1))
    model[0].requires_grad_(False)
    names=[(n,tuple(p.shape),id(p)) for n,p in model.named_parameters()]
    original={n:p.detach().clone() for n,p in model.named_parameters()}
    ema=TrainingEMA(model)
    with torch.no_grad():model[1].weight.add_(1)
    raw=model[1].weight.detach().clone();ema.update(model)
    with pytest.raises(RuntimeError):
        with ema.apply(model):
            assert not torch.equal(model[1].weight,raw)
            torch.testing.assert_close(model[0].weight,original['0.weight'])
            raise RuntimeError('fixture')
    torch.testing.assert_close(model[1].weight,raw)
    assert names==[(n,tuple(p.shape),id(p)) for n,p in model.named_parameters()]
    other=TrainingEMA(model);other.load_state_dict(ema.state_dict())
    assert other.updates==ema.updates
    for n,v in ema.shadow.items():torch.testing.assert_close(other.shadow[n],v)
