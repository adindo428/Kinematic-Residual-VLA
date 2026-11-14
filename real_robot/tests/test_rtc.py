from types import SimpleNamespace
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'third_party/DynamicVLA'))
from real_robot.rtc import prefix_weights,guided_velocity,normalize_prefix,guide_smolvla
from real_robot.policy_stream import ActionStream


def test_guidance_vjp_matches_analytic_flow_and_leaves_padding_unconstrained():
    x=torch.ones(1,4,9);target=torch.full_like(x,3.)
    weights=torch.zeros_like(x);weights[:,:,:7]=prefix_weights(1,3,4)[None,:,None]
    # v=.2*x; endpoint=.9*x, Jacobian=.9 at t=.5; gain=2.
    result=guided_velocity(x,.5,lambda y:.2*y,target,weights)
    expected=.2*x-2*.9*(target-.9*x)*weights
    torch.testing.assert_close(result,expected)
    torch.testing.assert_close(result[:,:,7:],.2*x[:,:,7:])
    assert not result.requires_grad and torch.equal(x,torch.ones_like(x))
    torch.testing.assert_close(prefix_weights(2,4,5),torch.tensor([1.,1.,2/3,1/3,0.]))
    assert torch.isfinite(guided_velocity(x,1.,lambda y:.2*y,target,weights)).all()


def mock_model():
    native=SimpleNamespace(policy=SimpleNamespace(config=SimpleNamespace(max_action_dim=32),model=SimpleNamespace(denoise_step=lambda *a:a[-2]*.2)),
        affine=lambda x,kind:x*2)
    return SimpleNamespace(method='smolvla',native=native,prepare=lambda b:(None,torch.full((1,20,7),.01)))


def test_prefix_reanchors_wraps_and_removes_kr_before_normalization():
    model=mock_model();state=torch.tensor([[.1,.2,.3,0.,0.,6.27]])
    absolute=np.array([[.2,.3,.4,0.,0.,.01,.8]],np.float32)
    original=absolute.copy();result=normalize_prefix(model,dict(reference_state=state),absolute)
    torch.testing.assert_close(result[0,0,:3],torch.tensor([.18,.18,.18]))
    assert result[0,0,5].item()==pytest.approx(2*(.01-6.27+2*np.pi-.01),abs=1e-6)
    assert result[0,0,6].item()==pytest.approx(1.58)
    assert result.shape==(1,1,32) and not result[:,:,7:].any()
    np.testing.assert_array_equal(absolute,original)
    original_method=model.native.policy.model.denoise_step
    with guide_smolvla(model,dict(reference_state=state),absolute) as diagnostics:
        model.native.policy.model.denoise_step(None,None,torch.ones(1,20,32),torch.tensor([.5]))
        assert diagnostics['guided_denoise_steps']==1
    assert model.native.policy.model.denoise_step is original_method
    with guide_smolvla(model,dict(reference_state=state),np.empty((0,7))) as diagnostics:
        assert diagnostics['prefix_actions']==0
        assert model.native.policy.model.denoise_step is original_method


def test_prefix_only_copies_contiguous_unconsumed_current_trial_actions():
    stream=ActionStream('a');actions=np.arange(140).reshape(20,7)
    stream.merge('a',0,actions,4);stream.pop(4)
    prefix=stream.rtc_prefix(5);np.testing.assert_array_equal(prefix,actions[5:])
    prefix[:]=0;np.testing.assert_array_equal(stream.rtc_prefix(5),actions[5:])
    assert stream.rtc_prefix(4).shape==(0,7)
    del stream.pending[7]
    np.testing.assert_array_equal(stream.rtc_prefix(5),actions[5:7])
    assert ActionStream('b').rtc_prefix(5).shape==(0,7)
    aged=ActionStream('a',max_source_age_steps=5)
    aged.merge('a',0,actions,4)
    np.testing.assert_array_equal(aged.rtc_prefix(4),actions[4:6])


def test_worker_waits_for_the_controller_to_accept_the_previous_result():
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workstation'))
    from real_family_policy_worker import rtc_observation_ready
    assert rtc_observation_ready(dict(token='a'),None)
    assert not rtc_observation_ready(dict(token='a',rtc_plan_source_index=3),('a',8))
    assert rtc_observation_ready(dict(token='a',rtc_plan_source_index=8),('a',8))
    assert rtc_observation_ready(dict(token='b',rtc_plan_source_index=-1),('a',8))
