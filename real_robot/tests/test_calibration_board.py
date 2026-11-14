"""Plain-board detection, label symmetry, occluded contacts and metric fitting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
import io
import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock,patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from real_robot.calibration_board import Chessboard,ChessboardDetector,contact_layout
from real_robot.base_calibration import pending_contact_verified,fit_contacts,write,capture
from real_robot.multiview import estimate_board
from real_robot.tests.test_base_calibration import fixture


class ChessboardTests(unittest.TestCase):
    def test_capture_commits_frozen_contact_only_after_the_occluded_board_returns(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rig,_,_=fixture(root);write(root/'rig.json',rig)
            config=root/'config.yaml';config.write_text('schema: right_arm_tracking_v1\nperception:\n  rig: '+str(root/'rig.json')+'\n')
            args=SimpleNamespace(config=config,output=root/'capture',board_type='chessboard',board_squares=[10,10],
                square_mm=31.,marker_mm=37.5,tcp_offset_m=[.16,0,0],execute=True,calibration_dir='unused',seconds=10.)
            clock=[0.];frame_number=[-1]
            pixels=[[100+c*20,80+r*20] for r in range(9) for c in range(9)]
            pose=np.eye(4);pose[2,3]=.5
            detection=dict(valid=True,corner_count=81,ids=list(range(81)),pixels=pixels,T_camera_board=pose.tolist())
            def frame():
                return dict(bgr=np.zeros((480,640,3),np.uint8),depth=np.full((480,640),.5),
                    deproject=lambda p,z:[((p[0]-100)/20+1)*.031*z/.5,((p[1]-80)/20+1)*.031*z/.5,z],
                    timing=dict(host_arrival_monotonic_s=clock[0],camera_serial=rig['front_serial'],
                                pyrealsense2_version='test',intrinsics={},distortion_model='test'))
            def poll():
                frame_number[0]+=1
                return frame(),None,{}
            camera=Mock();camera.start.return_value=camera;camera.poll.side_effect=poll
            def detect(*_):
                return {'valid':False,'corner_count':0} if frame_number[0]==11 else copy.deepcopy(detection)
            keys=iter([255]*10+[ord('b'),ord('s'),255,255,ord('q')])
            def key(*_):
                clock[0]+=.1
                return next(keys)
            def robot_row():
                tcp=np.eye(4);tcp[:3,3]=[.3,.1,.03 if clock[0]<1.25 else .13]
                tcp[:3,:3]=Rotation.from_euler('y',np.deg2rad(55)).as_matrix()
                return dict(T_base_tcp_candidate=tcp.tolist(),host_read_end_monotonic_s=clock[0])
            history=[]
            for i in range(8):
                r=robot_row();r['host_read_end_monotonic_s']=1.2-i*.03;history.append(r)
            control=Mock(pending=queue.Queue(),enabled=False,feedback_timing=None,lock=threading.Lock(),history=history)
            control.stationary.return_value=True;control.snapshot.side_effect=robot_row
            bus=Mock(is_connected=True)
            timing=SimpleNamespace(monotonic=lambda:clock[0],sleep=lambda delay:clock.__setitem__(0,clock[0]+delay))
            with patch('real_robot.base_calibration.time',timing), \
                 patch('real_robot.grasp_capture.connect_buses',return_value=({'leader':bus,'follower':bus},{})), \
                 patch('real_robot.grasp_capture.RobotIO'), \
                 patch('real_robot.collection_control.CollectionControl',return_value=control), \
                 patch('real_robot.recording_lease.LeaseOwner'), \
                 patch('real_robot.base_calibration.CameraPair',return_value=camera), \
                 patch('real_robot.base_calibration.warm_camera',side_effect=lambda *a:(frame(),copy.deepcopy(detection))), \
                 patch('real_robot.base_calibration.warm_feedback',return_value={'stable_reads':8}), \
                 patch('real_robot.base_calibration.estimate_board',side_effect=detect), \
                 patch('real_robot.base_calibration.cv2.imshow'), \
                 patch('real_robot.base_calibration.cv2.waitKey',side_effect=key), \
                 patch('real_robot.base_calibration.cv2.destroyAllWindows'),redirect_stdout(io.StringIO()):
                capture(args)
            contacts=json.loads((args.output/'contacts.json').read_text())
            self.assertEqual(len(contacts),1)
            np.testing.assert_allclose(contacts[0]['base_tcp_m'],[.3,.1,.03])
            self.assertGreater(contacts[0]['board_verified_after_contact_monotonic_s'],1.2)
            self.assertFalse((args.output/'pending_contact.json').exists())

    def test_squares_are_not_inner_corners_and_metric_spacing_is_preserved(self):
        board=Chessboard([10,10],.031)
        self.assertEqual(board.pattern_size,(9,9))
        self.assertEqual(board.getChessboardCorners().shape,(81,3))
        np.testing.assert_allclose(board.getChessboardCorners()[0],[.031,.031,0])
        np.testing.assert_allclose(board.getChessboardCorners()[-1],[.279,.279,0])
        layout=contact_layout([10,10])
        self.assertEqual(layout['fit_ids'],(0,8,72,80))
        self.assertEqual(layout['check_ids'],(20,60))
        self.assertEqual(len(layout['targets']),12)
        self.assertEqual(contact_layout([6,6])['fit_ids'],(0,4,20,24))

    def test_real_opencv_detection_finds_81_intersections_and_recovers_metric_pose(self):
        board=Chessboard([10,10],.031);detector=ChessboardDetector(board)
        image=np.full((480,640,3),200,np.uint8)
        pattern=np.uint8((np.indices((10,10)).sum(0)%2)*255)
        image[110:360,200:450]=cv2.resize(pattern,(250,250),interpolation=cv2.INTER_NEAREST)[...,None]
        frame=dict(bgr=image,timing={'intrinsics':dict(fx=400,fy=400)},
                   deproject=lambda pixel,z:[(pixel[0]-320)*z/400,(pixel[1]-240)*z/400,z])
        result=estimate_board(frame,board,detector)
        self.assertTrue(result['valid'])
        self.assertEqual(result['corner_count'],81)
        self.assertEqual(result['marker_count'],0)
        # 31 mm per 25 pixels at f=400 => depth 0.496 m.
        self.assertAlmostEqual(result['T_camera_board'][2][3],.496,places=3)
        np.testing.assert_allclose(result['object_points_m'][-1],[.279,.279,0],atol=1e-7)

    def test_detector_ordering_flips_cannot_relabel_a_locked_square_grid(self):
        board=Chessboard([10,10],.031);detector=ChessboardDetector(board)
        grid=np.array([[[100+col*20,80+row*20] for col in range(9)] for row in range(9)],np.float32)
        detector.lock_reference(grid.reshape(-1,2))
        for permuted in [grid,grid[::-1,::-1],grid.transpose(1,0,2),grid[:,::-1]]:
            np.testing.assert_array_equal(detector.order_corners(permuted).reshape(9,9,2),grid)

    def test_contact_tcp_is_frozen_until_a_fresh_unoccluded_board_is_verified(self):
        pose=np.eye(4).tolist()
        pending=dict(recorded_monotonic_s=10.,sample={'base_tcp_m':[.4,.1,.03]})
        reference=dict(detection={'T_camera_board':pose})
        frame=dict(timing={'host_arrival_monotonic_s':9.99})
        detection=dict(valid=True,T_camera_board=pose)
        self.assertFalse(pending_contact_verified(pending,frame,detection,reference,10.05))
        frame['timing']['host_arrival_monotonic_s']=10.1
        self.assertFalse(pending_contact_verified(pending,frame,{'valid':False},reference,10.12))
        self.assertTrue(pending_contact_verified(pending,frame,detection,reference,10.12))
        self.assertEqual(pending['sample']['base_tcp_m'],[.4,.1,.03])
        self.assertEqual(pending['sample']['board_verified_after_contact_monotonic_s'],10.1)
        shifted=copy.deepcopy(detection);shifted['T_camera_board'][0][3]=.01
        with self.assertRaisesRegex(RuntimeError,'Board moved'):
            pending_contact_verified(pending,frame,shifted,reference,10.12)

    def test_31mm_board_fit_uses_its_new_contact_ids_and_reserved_checks(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,truth,_=fixture(root)
            manifest=json.loads((root/'manifest.json').read_text())
            layout=contact_layout([10,10])
            manifest.update(board_type='chessboard',board_squares=[10,10],square_mm=31.,targets=layout['targets'])
            samples=[]
            for placement,corner,split in layout['targets']:
                row,col=divmod(corner,9)
                base=np.array([.25+(col+1)*.031,-.16+(row+1)*.031,.03+placement*.05])
                rotation=Rotation.from_euler('y',np.deg2rad(55 if corner in layout['low_pitch_ids'] else 90)).as_matrix()
                pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=base-rotation[:,0]*(.152-.160)
                camera=truth[:3,:3].T@(base-truth[:3,3])
                samples.append(dict(placement=placement,corner_id=corner,split=split,
                    camera_pnp_m=camera.tolist(),camera_depth_m=camera.tolist(),base_tcp_m=pose[:3,3].tolist(),
                    robot={'T_base_tcp_candidate':pose.tolist()},contact_operator_confirmed=True,
                    board_verified_after_contact_monotonic_s=100.))
            write(root/'manifest.json',manifest);write(root/'contacts.json',samples)
            result=fit_contacts(root)
            self.assertTrue(result['passed'])
            self.assertEqual(result['board_squares'],[10,10])
            self.assertAlmostEqual(result['tcp_offset_m'][0],.152,places=6)
            np.testing.assert_allclose(result['T_base_front'],truth,atol=1e-6)
            del samples[0]['board_verified_after_contact_monotonic_s'];write(root/'contacts.json',samples)
            with self.assertRaisesRegex(ValueError,'visibility check'):fit_contacts(root)


if __name__=='__main__':unittest.main()
