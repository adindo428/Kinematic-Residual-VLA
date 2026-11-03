import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('finger_contact_friction',Path(__file__).resolve().parents[1]/'diagnostics/friction_20261009/friction.py')
friction=importlib.util.module_from_spec(spec);spec.loader.exec_module(friction)

def test_only_finger_shapes_are_selected():
    assert friction.finger_shape_indices(['arm','panda_leftfinger','hand','panda_rightfinger'],[2,3,1,2])==[2,3,4,6,7]
    with pytest.raises(ValueError,match='Both'):
        friction.finger_shape_indices(['hand'],[2])

def test_contact_update_preserves_rest_of_robot_and_restitution():
    torch=pytest.importorskip('torch')
    materials=torch.tensor([[[.5,.5,.2],[.5,.4,.1],[.3,.2,0.]]])
    updated=friction.update_materials(materials,[1],2.)
    assert torch.equal(updated[:,[0,2]],materials[:,[0,2]])
    assert torch.equal(updated[...,2],materials[...,2])
    assert updated[0,1,:2].tolist()==[2.,2.]
    assert materials[0,1,:2].tolist()==pytest.approx([.5,.4])
    assert torch.equal(friction.update_materials(materials,[1],None),materials)
    with pytest.raises(ValueError,match='reduce'):
        friction.update_materials(materials,[1],.1)
