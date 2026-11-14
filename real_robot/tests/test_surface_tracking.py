import unittest
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from real_robot.surface_tracking import SCHEMA,SurfaceModel,plane_from_poses
from real_robot.tracking import VelocityTracker


def spec():
    return dict(schema=SCHEMA,ball_diameter_m=.029,camera_serial='front',surfaces=[
        dict(name='table',normal_camera=[0,0,-1],offset_camera_m=-1.,region_xy=[[5,5],[620,5],[620,470],[5,470]]),
        dict(name='ramp',normal_camera=[0,.5,-np.sqrt(.75)],offset_camera_m=-.9,region_xy=[[10,10],[250,10],[250,450],[10,450]])])


def mask(x=400,y=240):
    result=np.zeros((480,640),np.uint8);cv2.circle(result,(x,y),5,255,-1);return result


def ray(pixel,z):return [(pixel[0]-320)/400*z,(pixel[1]-240)/400*z,z]


class SurfaceTrackingTests(unittest.TestCase):
    def test_table_center_height_includes_measured_radius(self):
        model=SurfaceModel(spec());point,q=model.measure(mask(),ray)
        np.testing.assert_allclose(point,[.2*.9855,0,.9855],atol=1e-10)
        self.assertEqual(q['surface'],'table');self.assertFalse(q['depth_used'])

    def test_ramp_has_priority_and_uses_its_own_normal(self):
        model=SurfaceModel(spec());point,q=model.measure(mask(120,200),ray)
        n=np.asarray(spec()['surfaces'][1]['normal_camera'])
        self.assertEqual(q['surface'],'ramp')
        self.assertAlmostEqual(n@point+.9,.0145)

    def test_boundary_and_clipped_masks_are_unavailable(self):
        model=SurfaceModel(spec())
        point,q=model.measure(mask(248),ray)
        self.assertIsNone(point);self.assertEqual(q['reason'],'surface_boundary_or_partial_region')
        point,q=model.measure(mask(0),ray)
        self.assertIsNone(point);self.assertEqual(q['reason'],'object_mask_clipped')

    def test_board_plane_normal_and_board_thickness(self):
        pose=np.eye(4);pose[2,3]=1.
        rows=[dict(T_camera_board=pose.tolist(),ray_rmse_equivalent_px=.2) for _ in range(20)]
        plane=plane_from_poses(rows,.003)
        np.testing.assert_allclose(plane['normal_camera'],[0,0,-1])
        self.assertAlmostEqual(plane['offset_camera_m'],-1.003)
        rows[-1]['T_camera_board'][2][3]=1.05
        with self.assertRaises(ValueError):plane_from_poses(rows)

    def test_constant_metric_velocity_from_pixels_without_depth(self):
        model=SurfaceModel(spec());tracker=VelocityTracker(measurement_std_m=.003,stationary_updates=False,max_speed_mps=1.2)
        speeds=[]
        for i in range(65):
            t=i/30;x=.02+.2*t
            pixel=int(round(320+400*x/.9855))
            point,q=model.measure(mask(pixel),ray)
            self.assertIsNotNone(point)
            track=tracker.update(t,point)
            if track.ready and track.accepted and i>30:speeds.append(np.linalg.norm(track.velocity_mps))
        self.assertAlmostEqual(float(np.mean(speeds)),.2,delta=.015)


if __name__=='__main__':unittest.main()
