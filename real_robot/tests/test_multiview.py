import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np

from real_robot.geometry import transform_point
from real_robot.multiview import FramePairer, fit_relative, fuse_proxy, pose_difference, ProxyAssociation
from real_robot.perception import detect_mask
from real_robot.multiview_calibrate import steady, build_rig
from real_robot.multiview_track import load_rig, live_observation, check_camera_profile
from real_robot import multiview_track


def frame(stamp, color=None, arrival=10., domain='timestamp_domain.global_time'):
    return {'timing': {'sensor_timestamp_s': stamp, 'color_timestamp_s': stamp if color is None else color,
                       'host_arrival_monotonic_s': arrival, 'timestamp_domain': domain}}


class PairTests(unittest.TestCase):
    def test_complete_rgb_depth_span_and_no_reuse(self):
        p = FramePairer(.025)
        p.add('front', [frame(1.)]); p.add('side', [frame(1.01, color=1.02)])
        a, b, t = p.pop(10.1)
        self.assertAlmostEqual(t['span_s'], .02); self.assertAlmostEqual(t['timestamp_s'], 1.02)
        self.assertIsNone(p.pop(10.1))
        p.add('front', [frame(1.066)]); p.add('side', [frame(1.076, color=1.11)])
        self.assertIsNone(p.pop(10.1))  # Depth alone is close; RGB is outside the budget.

    def test_old_frames_drop_and_independent_hardware_clocks_reject(self):
        p = FramePairer(.025)
        p.add('front', [frame(.9), frame(1.)]); p.add('side', [frame(1.01)])
        self.assertIsNotNone(p.pop(10.1)); self.assertEqual(p.dropped['front'], 1)
        q = FramePairer(); q.add('front', [frame(1., domain='hardware_clock')])
        q.add('side', [frame(1., domain='hardware_clock')])
        with self.assertRaises(ValueError): q.pop(10.1)

    def test_stale_and_nonmonotonic_frames_are_not_used(self):
        p = FramePairer(); p.add('front', [frame(1.)]); p.add('side', [frame(1.)])
        self.assertIsNone(p.pop(11.))
        with self.assertRaises(ValueError): p.add('front', [frame(1.)])


class FusionTests(unittest.TestCase):
    def test_identity_association_rejects_distant_blue_regions_and_expires(self):
        association = ProxyAssociation(np.eye(4))
        target = [0, 0, .3]; distractor = [1, 0, .3]
        initial = association.update(1., {'front': [distractor, target], 'side': [target]})
        self.assertEqual(initial['status'], 'fused')
        self.assertEqual(initial['selected_candidate_indices']['front'], 1)
        fallback = association.update(1.1, {'front': [distractor], 'side': [[.02, 0, .3]]})
        self.assertEqual(fallback['status'], 'single_view'); self.assertEqual(fallback['sources'], ['side'])
        rejected = association.update(1.2, {'front': [distractor], 'side': []})
        self.assertIsNone(rejected['position_front_m'])
        expired = association.update(1.8, {'front': [target], 'side': []})
        self.assertEqual(expired['status'], 'awaiting_two_view_reacquisition')
        recovered = association.update(1.9, {'front': [target], 'side': [target]})
        self.assertEqual(recovered['status'], 'fused')

    def test_two_matching_blue_objects_remain_ambiguous(self):
        association = ProxyAssociation(np.eye(4))
        points = [[0, 0, .3], [.2, 0, .3]]
        result = association.update(1., {'front': points, 'side': points})
        self.assertEqual(result['status'], 'association_ambiguous'); self.assertIsNone(result['position_front_m'])

    def test_candidate_masks_preserve_single_view_ambiguity_and_holes(self):
        im = np.zeros((200, 300, 3), np.uint8)
        for x in (70, 220):
            cv2.circle(im, (x, 100), 30, (255, 0, 0), -1)
            cv2.circle(im, (x, 100), 15, (0, 0, 0), -1)
        self.assertIsNone(detect_mask(im, 'blue_tape'))
        candidates = detect_mask(im, 'blue_tape', all_candidates=True)
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(c[100, 70] == 0 and c[100, 220] == 0 for c in candidates))

    def test_candidate_listing_keeps_target_in_blue_background_clutter(self):
        im = np.zeros((300, 500, 3), np.uint8)
        for x in range(40, 480, 40): cv2.circle(im, (x, 50), 12, (255, 0, 0), -1)
        cv2.circle(im, (250, 200), 30, (255, 0, 0), -1)
        cv2.circle(im, (250, 200), 15, (0, 0, 0), -1)
        candidates = detect_mask(im, 'blue_tape', all_candidates=True)
        self.assertEqual(len(candidates), 12)
        self.assertTrue(any(c[200, 273] != 0 for c in candidates))

    def test_camera_profile_change_requires_recalibration(self):
        profile = {'camera_serial': 'side', 'pyrealsense2_version': '2.59',
                   'intrinsics': {'fx': 393., 'width': 640}, 'distortion_model': 'inverse_brown_conrady'}
        rig = {'camera_profiles': {'side': profile}}
        check_camera_profile(rig, 'side', {**profile, 'sensor_timestamp_s': 100.})
        check_camera_profile(rig, 'side', {**profile, 'intrinsics': {'fx': 393.21, 'width': 640}})
        for changed in ({**profile, 'intrinsics': {'fx': 435., 'width': 640}},
                        {**profile, 'intrinsics': {'fx': 396., 'width': 640}},
                        {**profile, 'pyrealsense2_version': '2.55'}, {**profile, 'camera_serial': 'other'}):
            with self.assertRaises(ValueError): check_camera_profile(rig, 'side', changed)

    def test_camera_only_cli_logs_fusion_and_single_view_without_actions(self):
        fs = np.eye(4); fs[0, 3] = .4
        rig = {'schema': 'front_side_rig_v1', 'front_serial': '234222300865',
               'side_serial': '218722271409', 'fixed_cameras': True,
               'relative_calibration_validated': True, 'T_front_side': fs.tolist(),
               'output_frame': 'front_camera_optical'}
        class Cameras:
            def __init__(self, **kwargs):
                self.i = 0; self.closed = False; self.latest = {}
                self.pairer = SimpleNamespace(dropped={'front': 0, 'side': 0})
                self.overruns = {'front': 0, 'side': 0}
            def start(self): return self
            def poll(self):
                if self.i >= 2: return None
                self.i += 1
                f = frame(100+self.i*.067, arrival=time.monotonic())
                f.update(bgr=np.zeros((480, 640, 3), np.uint8), depth=np.ones((480, 640)), deproject=None)
                self.latest = {'front': f, 'side': f}
                return f, f, {'timestamp_s': 100+self.i*.067, 'span_s': .01}
            def views(self): return self.latest
            def close(self): self.closed = True
        camera = Cameras()
        points = [(np.array([.4, 0, .3]), {}), (np.array([0, 0, .3]), {}),
                  (None, {}), (np.array([0, 0, .3]), {})]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'rig.json'; path.write_text(json.dumps(rig)); output = Path(d)/'capture'
            argv = ['fusion', '--rig', str(path), '--output', str(output), '--seconds', '.08', '--save-every', '0']
            with patch('sys.argv', argv), patch.object(multiview_track, 'CameraPair', return_value=camera), \
                    patch.object(multiview_track, 'detect_mask', return_value=[np.zeros((480, 640), np.uint8)]), \
                    patch.object(multiview_track, 'depth_measurement', side_effect=points), patch('builtins.print'):
                multiview_track.main()
            rows = [json.loads(line) for line in (output/'fusion.jsonl').read_text().splitlines()]
            self.assertEqual([r['fusion']['status'] for r in rows], ['fused', 'single_view'])
            self.assertTrue(all(r['action'] is None and r['motion_sent'] is False for r in rows))
            np.testing.assert_allclose(rows[1]['fusion']['position_output_m'], [.4, 0, .3])
            self.assertEqual(rows[1]['track']['status'], 'warming_up')
            self.assertTrue(camera.closed)
            status = json.loads((output/'status.json').read_text())
            self.assertFalse(status['running']); self.assertIsNone(status['latest_fusion']['position_output_m'])

    def test_published_coordinates_expire_without_new_pairs(self):
        views = {r: frame(1.) for r in ('front', 'side')}
        fusion = {'status': 'fused', 'position_output_m': [1, 2, 3], 'position_front_m': [1, 2, 3]}
        self.assertFalse(live_observation(fusion, views, 10., 10.1)['stale'])
        for missing_views, pair_time, now in [(views, 10., 10.6), ({'front': frame(1.)}, 10., 10.1),
                                               (views, None, 10.1), (views, 11., 11.1)]:
            result = live_observation(fusion, missing_views, pair_time, now)
            self.assertTrue(result['stale']); self.assertIsNone(result['position_output_m'])
            self.assertIsNone(result['position_front_m'])
        self.assertEqual(fusion['position_output_m'], [1, 2, 3])

    def test_rig_requires_heldout_validation_and_correct_base_composition(self):
        fs = np.eye(4); fs[0, 3] = .4
        base = np.eye(4); base[:3, :3] = cv2.Rodrigues(np.array([0., 0., .8]))[0]
        rig = {'schema': 'front_side_rig_v1', 'front_serial': '234222300865',
               'side_serial': '218722271409', 'fixed_cameras': True,
               'relative_calibration_validated': True, 'T_front_side': fs.tolist(),
               'output_frame': 'right_base_estimate', 'base_calibration_validated': False,
               'operator_confirmed_front_and_base_unchanged': True,
               'T_base_front_candidate': base.tolist(), 'T_base_side_candidate': (base@fs).tolist()}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'rig.json'; path.write_text(json.dumps(rig))
            self.assertEqual(load_rig(path)[0]['output_frame'], 'right_base_estimate')
            for key, wrong in [('relative_calibration_validated', False), ('side_serial', 'wrong'),
                               ('base_calibration_validated', True), ('operator_confirmed_front_and_base_unchanged', False),
                               ('T_base_side_candidate', (fs@base).tolist())]:
                bad = {**rig, key: wrong}; path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): load_rig(path)

    def test_rotate_translate_before_fusing_and_keep_uncertainty_floor(self):
        t = np.eye(4); t[:3, :3] = cv2.Rodrigues(np.array([0., .4, 0.]))[0]; t[:3, 3] = [.3, .02, .1]
        side = [.05, .01, .3]; front = transform_point(t, side)
        result = fuse_proxy(front, side, t)
        self.assertEqual(result['status'], 'fused')
        np.testing.assert_allclose(result['position_front_m'], front)
        self.assertGreaterEqual(np.linalg.eigvalsh(result['covariance_front_m2']).min(), .015**2)
        self.assertFalse(result['motion_sent'])

    def test_disagreement_loss_and_single_view_remain_explicit(self):
        t = np.eye(4)
        self.assertEqual(fuse_proxy([0, 0, .3], [.2, 0, .3], t)['status'], 'view_disagreement')
        self.assertIsNone(fuse_proxy([0, 0, .3], [.2, 0, .3], t)['position_front_m'])
        self.assertEqual(fuse_proxy(None, [.1, 0, .3], t)['status'], 'single_view')
        self.assertEqual(fuse_proxy(None, None, t)['status'], 'lost')
        with self.assertRaises(ValueError): fuse_proxy([np.nan, 0, .3], None, t)


class CalibrationTests(unittest.TestCase):
    def test_documented_exclusions_preserve_original_pose_indices_and_holdout(self):
        _, rows = self.observations()
        for row in rows:
            for role in ('front', 'side'):
                row[role]['timing'].update(camera_serial=role, pyrealsense2_version='test', distortion_model='test')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(ValueError): build_rig(root, rows, excluded_indices=[0, 1])
            rig = build_rig(root, rows, excluded_indices=[0, 1], exclusion_reason='Independent image evidence of camera movement')
            self.assertEqual(rig['fit_indices'], [2, 3, 4, 5, 6])
            self.assertEqual(rig['heldout_indices'], [7, 8])
            self.assertEqual(rig['excluded_indices'], [0, 1])
            self.assertTrue(rig['relative_calibration_validated'])
            self.assertFalse(rig['base_calibration_validated'])
            self.assertFalse(rig['motion_enabled'])

    def observations(self):
        rng = np.random.default_rng(45)
        fs = np.eye(4); fs[:3, :3] = cv2.Rodrigues(np.array([.1, -.6, .03]))[0]; fs[:3, 3] = [.4, -.1, .1]
        obj = np.array([[x*.05, y*.05, 0] for y in range(1, 6) for x in range(1, 6)])
        rows = []
        for i in range(9):
            fb = np.eye(4); fb[:3, :3] = cv2.Rodrigues(np.array([.10+.06*i, -.15+.04*(i%3), .02*i]))[0]
            fb[:3, 3] = [-.12+.04*i, -.12+.03*(i%3), .72+.02*(i%2)]
            row = {}
            for role, pose, focal in [('front', fb, 387.), ('side', np.linalg.inv(fs)@fb, 393.)]:
                p = obj@pose[:3, :3].T+pose[:3, 3]
                norm = p[:, :2]/p[:, 2:]+rng.normal(0, .10/focal, (25, 2))
                seed = pose.copy(); seed[:3, 3] += rng.normal(0, .0005, 3)
                row[role] = {'object_points_m': obj.tolist(), 'normalized_pixels': norm.tolist(),
                             'T_camera_board': seed.tolist(), 'timing': {'intrinsics': {'fx': focal, 'fy': focal}}}
            rows.append(row)
        return fs, rows

    def test_bundle_fit_recovers_rig_and_validates_unseen_board_poses(self):
        truth, rows = self.observations(); result = fit_relative(rows, list(range(6)), [6, 7, 8])
        d, angle = pose_difference(truth, result['T_front_side'])
        self.assertLess(d, .003); self.assertLess(angle, .3)
        self.assertTrue(result['relative_calibration_validated'])

    def test_bad_heldout_geometry_cannot_be_marked_valid(self):
        _, rows = self.observations()
        rows[8]['side']['normalized_pixels'] = (np.array(rows[8]['side']['normalized_pixels'])+.04).tolist()
        result = fit_relative(rows, list(range(6)), [6, 7, 8])
        self.assertFalse(result['relative_calibration_validated'])
        with self.assertRaises(ValueError): fit_relative(rows, [0, 1, 2, 3, 4], [4, 5])
        with self.assertRaises(ValueError): fit_relative([copy.deepcopy(rows[0]) for _ in rows], list(range(6)), [6, 7, 8])

    def test_steady_checks_complete_window_in_both_views(self):
        history = []
        for i in range(12):
            row = {r: {'ids': list(range(12)), 'pixels': [[j+i*.01, j] for j in range(12)]} for r in ['front', 'side']}
            history.append((i/15, row))
        self.assertTrue(steady(history))
        history[-1][1]['side']['pixels'] = [[j+10, j] for j in range(12)]
        self.assertFalse(steady(history))


if __name__ == '__main__': unittest.main()
