import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from real_robot.action_objective import physical_loss,physical_errors,xyz_rotation_matrix,weighted_mean


def recipe():
    return dict(early_steps=2,early_weight=4.,position_scale_m=.02,rotation_scale_deg=10.,gripper_scale=.2)


def test_physical_rotation_matches_scipy_and_wrap_has_zero_error():
    x=torch.tensor([[.3,-.5,2.],[6.2,1.1,-3.]],dtype=torch.float64)
    np.testing.assert_allclose(xyz_rotation_matrix(x).numpy(),Rotation.from_euler('xyz',x).as_matrix(),atol=1e-12)
    a=torch.zeros(1,1,7);b=a.clone();b[...,3]=2*math.pi
    assert physical_errors(a,b)['rotation_chordal'].item()<1e-12


def test_padding_and_early_weighting_have_correct_normalization():
    values=torch.tensor([[1.,3.,100.],[2.,4.,6.]])
    valid=torch.tensor([[True,True,False],[True,True,True]])
    w=torch.tensor([4.,1.,1.])
    torch.testing.assert_close(weighted_mean(values,valid,w),torch.tensor(((4+3)/5+(8+4+6)/6)/2))


def test_sampled_physical_objective_has_finite_nonzero_gradients_and_masks_padding():
    prediction=torch.zeros(2,3,7,requires_grad=True)
    target=torch.ones_like(prediction)*.01
    valid=torch.tensor([[True,True,False],[True,True,False]])
    loss,_=physical_loss(prediction,target,valid,recipe())
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[...,:2,:].abs().sum()>0
    assert prediction.grad[:,2].abs().sum()==0
    assert prediction.grad[:,0].abs().sum()==prediction.grad[:,1].abs().sum()


def test_matching_pose_has_zero_loss_and_finite_gradients():
    a=torch.zeros(1,4,7,requires_grad=True)
    loss,_=physical_loss(a,torch.zeros_like(a),torch.ones(1,4,dtype=torch.bool),recipe())
    loss.backward()
    assert loss.item()==0
    assert torch.isfinite(a.grad).all()
