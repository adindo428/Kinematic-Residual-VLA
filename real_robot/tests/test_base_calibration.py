"""Physical contact calibration math and evidence checks; no hardware access."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import queue
import time

import numpy as np
from scipy.spatial.transform import Rotation

from real_robot.base_calibration import (TARGETS, fit_contacts, validated_rig,
                                         check_evidence, write, digest, main, warm_feedback, warm_camera, capture,
                                         MultiScaleCharucoDetector, board_readiness,
                                         check_contact_pitch, tool_pitch_deg, board_depth_agreement)
from real_robot.multiview import FRONT_SERIAL, SIDE_SERIAL
from real_robot.multiview_track import load_rig


def fixture(root, true_tcp=.151):
    transform=np.eye(4)
    transform[:3,:3]=Rotation.from_euler('xyz',[.2,-.1,.15]).as_matrix()
    transform[:3,3]=[.3,-.05,-.6]
    old=transform.copy();old[0,3]+=.012
    relative=np.eye(4);relative[0,3]=.4
    rig=dict(schema='front_side_rig_v1',front_serial=FRONT_SERIAL,side_serial=SIDE_SERIAL,
             relative_calibration_validated=True,fixed_cameras=True,output_frame='right_base_estimate',
             operator_confirmed_front_and_base_unchanged=True,base_calibration_validated=False,
             T_front_side=relative.tolist(),T_base_front_candidate=old.tolist(),
             T_base_side_candidate=(old@relative).tolist())
    samples=[]
    for placement,corner,split in TARGETS:
        row,col=divmod(corner,5)
        point=np.array([.30+col*.05,-.12+row*.05,.03+placement*.05])
        rotation=Rotation.from_euler('y',np.deg2rad(55 if corner in (0,20,6) else 90)).as_matrix()
        base_guess=point-rotation[:,0]*(true_tcp-.16)
        pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=base_guess
        camera=transform[:3,:3].T@(point-transform[:3,3])
        samples.append(dict(placement=placement,corner_id=corner,split=split,
                            camera_pnp_m=camera.tolist(),camera_depth_m=camera.tolist(),
                            base_tcp_m=base_guess.tolist(),robot={'T_base_tcp_candidate':pose.tolist()},
                            contact_operator_confirmed=True))
    write(root/'manifest.json',dict(schema='krvla_base_contacts_v1',rig=rig,rig_sha256='source-hash',
                                   tcp_offset_m=[.16,0,0],mounts_unchanged=True))
    write(root/'contacts.json',samples)
    return rig,transform,samples


class BaseCalibrationTests(unittest.TestCase):
    def test_pitch_feedback_rejects_wrong_angle_before_a_contact_is_recorded(self):
        pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('y',np.deg2rad(80)).as_matrix()
        row={'T_base_tcp_candidate':pose.tolist()}
        self.assertAlmostEqual(tool_pitch_deg(row),80.)
        with self.assertRaisesRegex(ValueError,'Actual pitch 80.0 deg; need 55'):
            check_contact_pitch(row,55)
        pose[:3,:3]=Rotation.from_euler('y',np.deg2rad(89)).as_matrix()
        self.assertAlmostEqual(check_contact_pitch({'T_base_tcp_candidate':pose.tolist()},90),89.)

    def test_large_board_depth_disagreement_fails_before_collecting_contacts(self):
        reference={'points':{'0':dict(camera_pnp_m=[.1,.2,1.],camera_depth_m=[.1,.2,1.06])}}
        result=board_depth_agreement(reference)
        self.assertFalse(result['passed']);self.assertAlmostEqual(result['max_gap_m'],.06)
        reference['points']['0']['camera_depth_m']=[.1,.2,1.003]
        self.assertTrue(board_depth_agreement(reference)['passed'])

    def test_small_marker_fallback_preserves_native_pixel_coordinates(self):
        native_pixels=np.arange(50,dtype=np.float32).reshape(25,1,2)+.25
        enlarged_pixels=3*(native_pixels+.5)-.5
        native_markers=np.array([[[[1.,2.],[3.,4.],[5.,6.],[7.,8.]]]],np.float32)
        inner=Mock()
        inner.detectBoard.side_effect=[(None,None,[],None),
            (enlarged_pixels,np.arange(25,dtype=np.int32).reshape(-1,1),
             [3*(native_markers[0]+.5)-.5],np.array([[0]],np.int32))]
        detector=MultiScaleCharucoDetector(inner)
        corners,ids,markers,_=detector.detectBoard(np.zeros((20,20,3),np.uint8))
        np.testing.assert_allclose(corners,native_pixels,atol=1e-6)
        np.testing.assert_allclose(markers[0],native_markers[0],atol=1e-6)
        self.assertEqual(detector.last_scale,3)
        self.assertEqual(ids.size,25)

    def test_sufficient_native_detection_avoids_enlargement(self):
        inner=Mock();result=(np.zeros((25,1,2),np.float32),np.arange(25).reshape(-1,1),[],None)
        inner.detectBoard.return_value=result
        detector=MultiScaleCharucoDetector(inner)
        self.assertIs(detector.detectBoard(np.zeros((20,20,3),np.uint8)),result)
        inner.detectBoard.assert_called_once()
        self.assertEqual(detector.last_scale,1)

    def test_missing_detection_reports_detection_instead_of_blame_for_motion(self):
        result=board_readiness([],dict(valid=False,corner_count=0,marker_count=0,reason='not_enough_corners'),.05)
        self.assertFalse(result['ready'])
        self.assertIn('0/25 corners, 0 markers',result['message'])
        self.assertNotIn('still',result['message'])

    def test_readiness_distinguishes_missing_ids_short_history_and_pose_jitter(self):
        detection=dict(valid=True,corner_count=25,marker_count=18,ids=list(range(25)),T_camera_board=np.eye(4).tolist())
        history=[(i*.1,copy.deepcopy(detection)) for i in range(10)]
        self.assertTrue(board_readiness(history,detection,.05)['ready'])
        self.assertIn('building steady history',board_readiness(history[:2],detection,.05)['message'])
        partial={**detection,'ids':list(range(1,25))}
        self.assertIn('missing IDs [0]',board_readiness(history,partial,.05)['message'])
        history[0][1]['T_camera_board'][0][3]=.003
        self.assertIn('pose estimate varies',board_readiness(history,detection,.05)['message'])
        self.assertFalse(board_readiness(history,detection,.05)['ready'])
        self.assertIn('fresh camera',board_readiness(history,detection,.3)['message'])

    def test_read_only_warmup_requires_consecutive_fast_reads_and_never_commands(self):
        clock=[0.]
        durations=iter([.20,.01,.01,.19,.01,.01,.01])
        robot=Mock()
        robot.health.return_value=dict(hardware_errors=np.zeros(9))
        def read():
            clock[0]+=next(durations)
            return dict(temperatures_c=[25]*9)
        robot.read.side_effect=read
        timing=SimpleNamespace(monotonic=lambda:clock[0],sleep=lambda delay:clock.__setitem__(0,clock[0]+delay))
        with patch('real_robot.base_calibration.time',timing):
            result=warm_feedback(robot,stable_reads=3)
        self.assertEqual(result['stable_reads'],3)
        self.assertEqual(len(result['samples']),7)
        self.assertAlmostEqual(result['samples'][0]['total_ms'],200.)
        robot.goal.assert_not_called()
        self.assertEqual({c[0] for c in robot.method_calls},{'health','read'})

    def test_permanently_slow_warmup_fails_without_relaxing_150ms_limit(self):
        clock=[0.];robot=Mock()
        robot.health.return_value=dict(hardware_errors=np.zeros(9))
        def read():
            clock[0]+=.16
            return dict(temperatures_c=[25]*9)
        robot.read.side_effect=read
        timing=SimpleNamespace(monotonic=lambda:clock[0],sleep=lambda delay:clock.__setitem__(0,clock[0]+delay))
        with patch('real_robot.base_calibration.time',timing):
            with self.assertRaisesRegex(RuntimeError,'below 150 ms'):
                warm_feedback(robot,timeout_s=.5)
        robot.goal.assert_not_called()

    def test_first_camera_frame_primes_detector_and_window_before_returning(self):
        frame=dict(bgr=np.zeros((4,4,3),np.uint8),timing={'host_arrival_monotonic_s':0.})
        camera=Mock();camera.poll.return_value=(frame,None,{})
        with patch('real_robot.base_calibration.time.monotonic',return_value=0.), \
             patch('real_robot.base_calibration.estimate_board',return_value={'valid':False}) as detector, \
             patch('real_robot.base_calibration.cv2.imshow') as show, \
             patch('real_robot.base_calibration.cv2.waitKey',return_value=255) as keys:
            latest=warm_camera(camera,Mock(),Mock())
        self.assertIs(latest[0],frame)
        detector.assert_called_once();show.assert_called_once();keys.assert_called_once()

    def test_camera_and_feedback_initialization_precede_controller_start(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rig,_,_=fixture(root);write(root/'rig.json',rig)
            config=root/'config.yaml'
            config.write_text('schema: right_arm_tracking_v1\nperception:\n  rig: '+str(root/'rig.json')+'\n')
            args=SimpleNamespace(config=config,output=root/'capture',square_mm=50,marker_mm=37.5,
                                 tcp_offset_m=[.16,0,0],execute=True,calibration_dir='unused',seconds=1.)
            events=[]
            camera=Mock();camera.start.return_value=camera;camera.poll.return_value=None
            control=Mock(pending=queue.Queue(),enabled=False,feedback_timing=None)
            pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('y',np.deg2rad(55)).as_matrix()
            control.snapshot.return_value={'T_base_tcp_candidate':pose.tolist()}
            control.start.side_effect=lambda:events.append('controller_start')
            control.close.side_effect=lambda:events.append('controller_close')
            bus=Mock(is_connected=True);bus.disconnect.side_effect=lambda:events.append('disconnect')
            frame=dict(bgr=np.zeros((480,640,3),np.uint8),timing={'host_arrival_monotonic_s':time.monotonic()})
            with patch('real_robot.grasp_capture.connect_buses',return_value=({'leader':bus,'follower':bus},{})), \
                 patch('real_robot.grasp_capture.RobotIO'), \
                 patch('real_robot.collection_control.CollectionControl',return_value=control), \
                 patch('real_robot.recording_lease.LeaseOwner'), \
                 patch('real_robot.base_calibration.CameraPair',return_value=camera), \
                 patch('real_robot.base_calibration.warm_camera',side_effect=lambda *a: (events.append('camera_ready') or (frame,{}))), \
                 patch('real_robot.base_calibration.warm_feedback',side_effect=lambda *a:(events.append('feedback_ready') or {'stable_reads':8})), \
                 patch('real_robot.base_calibration.cv2.imshow'), \
                 patch('real_robot.base_calibration.cv2.waitKey',return_value=ord('q')), \
                 patch('real_robot.base_calibration.cv2.destroyAllWindows'),redirect_stdout(io.StringIO()):
                capture(args)
            self.assertEqual(events[:3],['camera_ready','feedback_ready','controller_start'])
            self.assertLess(events.index('controller_close'),events.index('disconnect'))
            control.toggle.assert_not_called()
            summary=json.loads((args.output/'capture_summary.json').read_text())
            self.assertIsNone(summary['error']);self.assertEqual(summary['contacts'],0)

    def test_joint_fit_recovers_tcp_and_camera_without_using_check_points(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rig,truth,samples=fixture(root)
            result=fit_contacts(root)
            self.assertTrue(result['passed'])
            np.testing.assert_allclose(result['T_base_front'],truth,atol=1e-6)
            self.assertAlmostEqual(result['tcp_offset_m'][0],.151,places=6)
            self.assertGreater(result['old_transform_check']['max_m'],.005)
            self.assertEqual(result['check']['count'],4)
            # Corrupt only the reserved locations: the fitted transform/TCP
            # must remain unchanged while validation fails.
            for s in samples:
                if s['split']=='check':s['base_tcp_m'][1]+=.020
            write(root/'contacts.json',samples)
            failed=fit_contacts(root)
            self.assertFalse(failed['passed'])
            np.testing.assert_allclose(failed['T_base_front'],result['T_base_front'],atol=1e-8)
            np.testing.assert_allclose(failed['tcp_offset_m'],result['tcp_offset_m'],atol=1e-8)
            with self.assertRaises(ValueError):validated_rig(rig,failed)

    def test_depth_bias_cannot_be_hidden_by_rgb_pose_fit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,_,samples=fixture(root)
            for s in samples:s['camera_depth_m'][2]+=.020
            write(root/'contacts.json',samples)
            report=fit_contacts(root)
            self.assertFalse(report['passed'])
            self.assertLess(report['check']['max_m'],1e-6)
            self.assertGreater(report['depth_all']['max_m'],.019)

    def test_validate_rig_preserves_relative_geometry_and_requires_matching_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rig,truth,_=fixture(root)
            result=fit_contacts(root);updated=validated_rig(rig,result)
            write(root/'rig.json',updated)
            loaded,relative,base,_=load_rig(root/'rig.json')
            self.assertTrue(loaded['base_calibration_validated'])
            np.testing.assert_allclose(base,truth,atol=1e-6)
            np.testing.assert_allclose(base@relative,updated['T_base_side_candidate'])
            updated['T_base_front_candidate'][0][3]+=.010
            with self.assertRaisesRegex(ValueError,'differs'):check_evidence(updated)

    def test_rank_and_coverage_fail_without_two_heights_or_pitch_variation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,_,samples=fixture(root)
            for s in samples:s['robot']['T_base_tcp_candidate']=np.eye(4).tolist()
            write(root/'contacts.json',samples)
            with self.assertRaisesRegex(ValueError,'pitch variation'):fit_contacts(root)
            _,_,samples=fixture(root)
            for s in samples:s['base_tcp_m'][2]=.03
            write(root/'contacts.json',samples)
            with self.assertRaisesRegex(ValueError,'height'):fit_contacts(root)

    def test_flipping_validation_flag_without_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rig,_,_=fixture(root)
            rig['base_calibration_validated']=True;write(root/'rig.json',rig)
            with self.assertRaisesRegex(ValueError,'evidence'):load_rig(root/'rig.json')

    def test_invalid_error_threshold_or_incomplete_contacts_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,_,samples=fixture(root)
            with self.assertRaisesRegex(ValueError,'10 mm'):fit_contacts(root,max_error_m=.02)
            write(root/'contacts.json',samples[:-1])
            with self.assertRaisesRegex(ValueError,'12 prescribed'):fit_contacts(root)

    def test_cli_writes_new_config_and_refuses_reusing_a_check_set(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root)
            config=root/'original.yaml';original='schema: right_arm_tracking_v1\nperception:\n  rig: original-rig.json\n'
            config.write_text(original)
            manifest=json.loads((root/'manifest.json').read_text())
            manifest['config_source']=str(config);manifest['config_yaml']=original
            write(root/'manifest.json',manifest)
            with redirect_stdout(io.StringIO()):main(['fit','--session',str(root)])
            self.assertTrue((root/'base_validation.json').is_file())
            self.assertTrue((root/'rig-validated.json').is_file())
            self.assertIn('rig-validated.json',(root/'config-validated.yaml').read_text())
            self.assertEqual(config.read_text(),original)
            with self.assertRaisesRegex(ValueError,'already evaluated'):
                main(['fit','--session',str(root)])


if __name__=='__main__':unittest.main()
