from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from real_robot.training_augmentation import augment_training_batch,model_inputs
from third_party.DynamicVLA.utils.real_robot_contract import absolute_pose,delta_pose


def batch():
    state=torch.tensor([[[.3,0,.2,.01,1.,6.27],[.31,0,.19,6.27,1.,.01]]]*4)
    absolute=torch.tensor([[[.32,.01,.18,6.27,1.,.02,.5]]*20]*4)
    image=torch.rand(4,2,3,16,16)
    return {'observation.state':state,'action':delta_pose(absolute,state[:,-1:,:],'wrapped_xyz_v1'),
        'observation.images.opst_cam':image.clone(),'observation.images.wrist_cam':image.clone(),
        'observation.images.opst_cam_is_pad':torch.zeros(4,2,dtype=torch.bool),
        'observation.environment_state':torch.ones(4,2,9),'dynamic_object.velocity_valid':torch.ones(4,2,dtype=torch.bool)}


def test_proprioception_noise_preserves_absolute_labels_gripper_and_clean_batch():
    original=batch();old_state=original['observation.state'].clone();old_actions=original['action'].clone()
    torch.manual_seed(42)
    result,metrics=augment_training_batch(original,{'proprioception':{'enabled':True,'sample_probability':1.}})
    assert metrics['proprioception_samples']==4
    np.testing.assert_array_equal(original['observation.state'],old_state);np.testing.assert_array_equal(original['action'],old_actions)
    torch.testing.assert_close(result['action'][...,6],old_actions[...,6])
    a=absolute_pose(old_actions,old_state[:,-1:,:],'wrapped_xyz_v1')
    b=absolute_pose(result['action'],result['observation.state'][:,-1:,:],'wrapped_xyz_v1')
    torch.testing.assert_close(a[...,:3],b[...,:3],atol=1e-6,rtol=0)
    np.testing.assert_allclose(Rotation.from_euler('xyz',a[...,3:6].numpy().reshape(-1,3)).as_matrix(),
        Rotation.from_euler('xyz',b[...,3:6].numpy().reshape(-1,3)).as_matrix(),atol=2e-6)
    shift=result['observation.state'][...,:3]-old_state[...,:3]
    torch.testing.assert_close(shift[:,0],shift[:,1])
    assert result['observation.environment_state'] is original['observation.environment_state']


def test_photometric_parameters_are_shared_across_history_and_cameras():
    original=batch();original['observation.images.opst_cam'][:,1]=original['observation.images.opst_cam'][:,0]
    original['observation.images.wrist_cam']=original['observation.images.opst_cam'].clone()
    result,_=augment_training_batch(original,{'photometric':{'enabled':True}})
    torch.testing.assert_close(result['observation.images.opst_cam'],result['observation.images.wrist_cam'])
    torch.testing.assert_close(result['observation.images.opst_cam'][:,0],result['observation.images.opst_cam'][:,1])
    torch.testing.assert_close(result['action'],original['action'])
    torch.testing.assert_close(result['observation.images.opst_cam_is_pad'],original['observation.images.opst_cam_is_pad'])


def test_validation_is_clean_and_plain_dom_has_no_object_conditioning_fields():
    original=batch();result,metrics=augment_training_batch(original,{'proprioception':{'enabled':True}},training=False)
    assert result is original and metrics=={}
    plain=model_inputs(original,SimpleNamespace(condition_on_object_velocity=False,enable_residual_flow=False))
    assert 'observation.environment_state' not in plain and not any(k.startswith('dynamic_object.') for k in plain)
    conditioned=model_inputs(original,SimpleNamespace(condition_on_object_velocity=True,enable_residual_flow=False))
    assert conditioned['observation.environment_state'] is original['observation.environment_state']


def test_future_object_information_is_never_a_model_input():
    original=batch();original['dynamic_object.future_velocity']=torch.ones(4,20,3)
    original['dynamic_object.future_valid']=torch.ones(4,20,dtype=torch.bool)
    for velocity,residual in [(False,False),(True,False),(True,True)]:
        inputs=model_inputs(original,SimpleNamespace(condition_on_object_velocity=velocity,enable_residual_flow=residual))
        assert not any(k.startswith('dynamic_object.future_') for k in inputs)
