"""Read-only diagnostic distinguishes color-ray geometry from depth rays."""
from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import cv2
import numpy as np

from real_robot.calibration_board import Chessboard,ChessboardDetector
from real_robot.camera_check import compare_board
from real_robot.camera_texture_check import check_perimeter, main as texture_main, perimeter_pixels, plane_sample, summarize
from real_robot.multiview import estimate_board


class CameraCheckTests(unittest.TestCase):
    def test_native_color_comparison_does_not_change_input_depth_or_use_robot_state(self):
        image=np.full((480,640,3),200,np.uint8)
        pattern=np.uint8((np.indices((10,10)).sum(0)%2)*255)
        image[110:360,200:450]=cv2.resize(pattern,(250,250),interpolation=cv2.INTER_NEAREST)[...,None]
        color_intr=SimpleNamespace(width=640,height=480,fx=420.,fy=420.,ppx=320.,ppy=240.,coeffs=[0.]*5,model='none')
        aligned_intr=SimpleNamespace(**{**vars(color_intr),'fx':400.,'fy':400.})
        def deproject(intr,pixel,z):return [(pixel[0]-intr.ppx)*z/intr.fx,(pixel[1]-intr.ppy)*z/intr.fy,z]
        depth=np.full((480,640),.5208)
        frame=dict(bgr=image,depth=depth,
                   timing=dict(intrinsics=dict(fx=400.,fy=400.)),
                   deproject=lambda p,z:deproject(aligned_intr,p,z))
        board=Chessboard([10,10],.031)
        report=compare_board(frame,board,ChessboardDetector(board),color_intr,
                             SimpleNamespace(rs2_deproject_pixel_to_point=deproject))
        self.assertGreater(report['aligned_depth_rays']['agreement']['max_gap_m'],.020)
        self.assertLess(report['native_color_rays']['agreement']['max_gap_m'],.001)
        np.testing.assert_array_equal(depth,np.full((480,640),.5208))

    def test_texture_check_allows_central_cover_but_rejects_outer_occlusion_and_movement(self):
        image=np.full((480,640,3),200,np.uint8)
        pattern=np.uint8((np.indices((10,10)).sum(0)%2)*255)
        image[110:360,200:450]=cv2.resize(pattern,(250,250),interpolation=cv2.INTER_NEAREST)[...,None]
        board=Chessboard([10,10],.031)
        frame=dict(bgr=image,timing=dict(intrinsics=dict(fx=400.,fy=400.)),
                   deproject=lambda p,z:[(p[0]-320)*z/400,(p[1]-240)*z/400,z])
        detection=estimate_board(frame,board,ChessboardDetector(board))
        pixels=perimeter_pixels(detection,board)
        covered=image.copy();covered[200:270,290:360]=180
        check=check_perimeter(image,covered,pixels)
        self.assertEqual(check['checked_corners'],32)
        self.assertLess(check['max_displacement_px'],.01)
        moved=cv2.warpAffine(image,np.float32([[1,0,3],[0,1,0]]),(640,480))
        with self.assertRaises(ValueError):check_perimeter(image,moved,pixels)
        obscured=covered.copy();obscured[120:150,210:240]=128
        with self.assertRaises(ValueError):check_perimeter(image,obscured,pixels)

    def test_texture_sample_accounts_for_tilt_and_rejects_missing_depth(self):
        pose=np.eye(4);pose[:3,:3]=cv2.Rodrigues(np.array([.4,0.,0.]))[0];pose[2,3]=.9
        normal=pose[:3,2];plane_distance=normal@pose[:3,3]
        def ray(p,z):return np.array([(p[0]-20)/50,(p[1]-20)/50,1.])*z
        depth=np.array([[plane_distance/(normal@ray([u,v],1.))+.02 for u in range(40)] for v in range(40)])
        frame=dict(depth=depth,deproject=ray)
        result=plane_sample(frame,pose,[20.,20.])
        self.assertAlmostEqual(result['median_signed_z_error_mm'],20.,places=8)
        self.assertAlmostEqual(result['p90_abs_z_error_mm'],20.,places=8)
        before=depth.copy()
        summary=summarize(dict(exposed=[dict(sample=result)],covered=[dict(sample={**result,'median_signed_z_error_mm':2.})]))
        self.assertAlmostEqual(summary['covered_minus_exposed_z_error_mm'],-18.)
        np.testing.assert_array_equal(before,depth)
        depth[18:23,18:23]=0
        with self.assertRaises(ValueError):plane_sample(frame,pose,[20.,20.])

    def test_texture_command_saves_two_stages_without_robot_or_camera_calibration_access(self):
        image=np.full((480,640,3),200,np.uint8)
        pattern=np.uint8((np.indices((10,10)).sum(0)%2)*255)
        image[110:360,200:450]=cv2.resize(pattern,(250,250),interpolation=cv2.INTER_NEAREST)[...,None]
        covered=image.copy();covered[200:270,290:360]=180
        class Camera:
            def __init__(self,*args,**kwargs):self.n=0;self.closed=False
            def read(self):
                is_covered=self.n>=12;self.n+=1
                return ((covered if is_covered else image).copy(),
                        np.full((480,640),.50 if is_covered else .56),
                        dict(host_arrival_monotonic_s=time.monotonic(),intrinsics=dict(fx=400.,fy=400.)),
                        lambda p,z:[(p[0]-320)*z/400,(p[1]-240)*z/400,z])
            def close(self):self.closed=True
        camera=Camera()
        with tempfile.TemporaryDirectory() as folder:
            session=Path(folder)/'session';session.mkdir()
            (session/'manifest.json').write_text(json.dumps(dict(board_type='chessboard',
                board_squares=[10,10],square_mm=31.,rig=dict(front_serial='fake'))))
            output=Path(folder)/'output'
            keys=[-1]*8+[ord('b')]+[-1]*4+[ord('s')]+[-1]*2
            with patch('real_robot.camera.RGBDCamera',return_value=camera), \
                 patch('real_robot.recording_lease.LeaseOwner'), \
                 patch('cv2.imshow'),patch('cv2.destroyAllWindows'),patch('cv2.waitKey',side_effect=keys):
                texture_main(['--session',str(session),'--output',str(output),'--frames','3'])
            report=json.loads((output/'texture_check.json').read_text())
            self.assertTrue(report['complete'])
            self.assertFalse(report['validated'])
            self.assertFalse(report['robot_ports_opened'])
            self.assertFalse(report['calibration_parameters_changed'])
            self.assertTrue(camera.closed)
            self.assertEqual(len(report['records']['exposed']),3)
            self.assertEqual(len(report['records']['covered']),3)
            self.assertAlmostEqual(report['summary']['covered_minus_exposed_z_error_mm'],-60.,places=6)


if __name__=='__main__':unittest.main()
