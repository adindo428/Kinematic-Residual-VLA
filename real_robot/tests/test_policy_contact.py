"""Grasp contact is limited to named fingertips, not other robot geometry."""
from types import SimpleNamespace
import numpy as np
import pytest
from real_robot.collision import GeometryGuard


def guard(profile,heights,contacts=()):
    names=['table','vx300s_right/10_left_gripper_finger','vx300s_right/10_right_gripper_finger',
        'wrist_camera_bound','vx300s_right/3_upper_arm']
    g=GeometryGuard.__new__(GeometryGuard);g.profile=profile;g.baseline={};g._pose=lambda q:None
    g.model=SimpleNamespace(ngeom=5,geom_type=np.full(5,2),geom_size=np.full((5,3),.001))
    positions=np.zeros((5,3));positions[:,2]=np.asarray(heights)+.001
    g.data=SimpleNamespace(contact=[SimpleNamespace(geom1=a,geom2=b,dist=d)for a,b,d in contacts],
        geom_xmat=np.tile(np.eye(3).ravel(),(5,1)),geom_xpos=positions)
    g.mj=SimpleNamespace(mjtObj=SimpleNamespace(mjOBJ_GEOM=0),mjtGeom=SimpleNamespace(mjGEOM_MESH=1,mjGEOM_BOX=2),
        mj_id2name=lambda m,kind,i:names[i])
    return g


def test_grasp_allows_table_contact_at_fingers_and_camera_below_old_hover_margin():
    g=guard('grasp',[0,-.02,-.021,.006,.10],[(0,1,-.02),(0,2,-.021)])
    g.check(np.zeros(6))


def test_hover_keeps_the_previous_20mm_margin():
    with pytest.raises(RuntimeError,match='clearance'):
        guard('hover',[0,.005,.006,.03,.10]).check(np.zeros(6))


def test_grasp_does_not_allow_camera_or_arm_to_enter_table():
    for heights in [[0,0,0,-.002,.10],[0,0,0,.01,-.002]]:
        with pytest.raises(RuntimeError,match='clearance'):
            guard('grasp',heights).check(np.zeros(6))
    with pytest.raises(RuntimeError,match='contact'):
        guard('grasp',[0,0,0,.01,.10],[(0,3,-.01)]).check(np.zeros(6))


def test_allowed_table_contact_does_not_allow_finger_to_arm_self_collision():
    with pytest.raises(RuntimeError,match='contact'):
        guard('grasp',[0,-.02,-.02,.01,.10],[(1,4,-.01)]).check(np.zeros(6))
