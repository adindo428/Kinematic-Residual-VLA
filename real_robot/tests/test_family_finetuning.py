from types import SimpleNamespace
from unittest.mock import patch
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workstation'))
from family_policy import enable_sampler_gradients,signature
from family_data import RealFamilyDataset


class TinySampler(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.3))
    def sample_actions(self,noise):
        x_t=noise
        dt=-.25
        for _ in range(4):
            v_t=self.weight*x_t
            x_t += dt * v_t
        return x_t
    @torch.no_grad()
    def denoise_step(self,x):return self.weight*x


def test_gradient_sampler_is_numerically_identical_without_new_parameters_or_noise_mutation():
    m=TinySampler();before=signature(m);noise=torch.ones(1,3,7)
    with torch.no_grad():expected=m.sample_actions(noise.clone())
    enable_sampler_gradients(m)
    actual=m.sample_actions(noise);actual.square().sum().backward()
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(noise,torch.ones_like(noise))
    assert m.weight.grad.abs()>0 and signature(m)==before
    assert m.denoise_step(noise).requires_grad


def test_vlash_delay_uses_previous_command_reference_and_restores_absolute_targets():
    from utils.real_robot_contract import absolute_pose
    d=RealFamilyDataset.__new__(RealFamilyDataset);d.anchors=[(5,0)];d.max_delay=3
    d.metadata={5:{'tasks':['{"task":"pick","objects":"ball"}']}}
    actions=torch.zeros(25,7);actions[:,0]=torch.arange(25)*.01;actions[:,3]=6.27;actions[:,6]=.5
    arrays={'action':actions,'observation.state':torch.zeros(25,6),'observation.environment_state':torch.zeros(25,9)}
    quality=[dict(velocity_required=False,velocity_estimated=False,velocity_zero_filled=True)]*25
    decoder=SimpleNamespace(get_frame_at=lambda i:SimpleNamespace(data=torch.zeros(3,8,8,dtype=torch.uint8)))
    d.episode=lambda ep:(arrays,quality,{'observation.images.opst_cam':decoder,'observation.images.wrist_cam':decoder})
    with patch('family_data.random.randrange',return_value=2):item=d[0]
    torch.testing.assert_close(item['reference_state'],actions[1,:6])
    restored=absolute_pose(item['action'],item['reference_state'][None,:],'wrapped_xyz_v1')
    torch.testing.assert_close(restored,actions[2:22],atol=1e-6,rtol=0)
    assert item['offset']==2 and not item['velocity_required'] and not item['velocity'].any()


def test_shared_checkpoint_mapping_does_not_alias_trainable_model_or_change_source(tmp_path):
    import hashlib,mmap
    path=tmp_path/'checkpoint.pt';original=torch.nn.Linear(3,2)
    torch.save({'model':original.state_dict(),'unused_optimizer':torch.zeros(16)},path)
    before=hashlib.sha256(path.read_bytes()).hexdigest();previous=torch.serialization.get_default_mmap_options()
    model=torch.nn.Linear(3,2)
    with torch.serialization.set_default_mmap_options(mmap.MAP_SHARED):
        snapshot=torch.load(path,mmap=True,weights_only=True)
        model.load_state_dict(snapshot['model'],strict=True)
    assert torch.serialization.get_default_mmap_options()==previous
    with torch.no_grad():model.weight.add_(2)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    torch.testing.assert_close(snapshot['model']['weight'],original.weight)


def test_sampling_temporarily_disables_checkpointing_without_disabling_gradients_or_training_mode():
    import pytest
    from family_policy import sampling_cache_context
    model=torch.nn.Sequential(torch.nn.Linear(3,2));model.train()
    model.gradient_checkpointing_enabled=True;model[0].gradient_checkpointing=True
    original_ids=[id(p) for p in model.parameters()]
    with pytest.raises(RuntimeError):
        with sampling_cache_context(model):
            assert not model.gradient_checkpointing_enabled and not model[0].gradient_checkpointing
            assert model.training and model[0].training
            assert model(torch.ones(1,3)).requires_grad
            raise RuntimeError('fixture')
    assert model.gradient_checkpointing_enabled and model[0].gradient_checkpointing
    assert original_ids==[id(p) for p in model.parameters()]
