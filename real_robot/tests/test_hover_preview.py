import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from real_robot.__main__ import candidate_calibration_for
from real_robot.controller import hover_preview_target
from real_robot.geometry import transform_point
from real_robot.tracking import VelocityTracker


class HoverPreviewTests(unittest.TestCase):
    def ready_track(self):
        tracker = VelocityTracker(frame='right_base_estimate', stationary_updates=False)
        for i in range(25):
            state = tracker.update(i/15, [.4+.01*i/15, .05, .1+.02*i/15])
        self.assertTrue(state.ready)
        return tracker, state

    def test_offset_follows_base_vertical_with_oblique_camera_and_moving_depth(self):
        _, state = self.ready_track()
        # Camera Z is base -Y; an offset in camera Z would move sideways.
        t = np.array([[1, 0, 0, .2], [0, 0, -1, 0], [0, 1, 0, .6], [0, 0, 0, 1.]])
        result = hover_preview_target(state)
        predicted = state.position_m+.1*state.velocity_mps
        np.testing.assert_allclose(np.asarray(result['target_m'])-predicted, [0, 0, .05])
        camera_target = transform_point(np.linalg.inv(t), result['target_m'])
        camera_object = transform_point(np.linalg.inv(t), predicted)
        np.testing.assert_allclose(camera_target-camera_object, [0, .05, 0], atol=1e-12)
        self.assertFalse(result['motion_sent'])

    def test_no_target_during_loss_warmup_or_outlier(self):
        tracker, state = self.ready_track()
        self.assertIsNone(hover_preview_target(None)['target_m'])
        missing = tracker.update(state.timestamp_s+1/15, None)
        self.assertIsNone(hover_preview_target(missing)['target_m'])
        outlier = tracker.update(state.timestamp_s+2/15, [2, 2, 2])
        self.assertIsNone(hover_preview_target(outlier)['target_m'])
        warmup = VelocityTracker(frame='right_base_estimate').update(0, [.4, 0, .1])
        self.assertIsNone(hover_preview_target(warmup)['target_m'])

    def test_camera_frame_and_uncertain_prediction_are_not_targets(self):
        _, state = self.ready_track()
        state.frame = 'camera_optical'
        self.assertIsNone(hover_preview_target(state)['target_m'])
        state.frame = 'right_base_estimate'
        state.covariance *= 100000
        self.assertIsNone(hover_preview_target(state)['target_m'])

    def test_candidate_requires_correct_arm_camera_and_provisional_status(self):
        data = {'schema': 'aloha_handeye_candidate_v1', 'arm': 'right',
                'front_camera_serial': '123', 'destination_frame': 'right_base_estimate',
                'calibration_validated': False,
                'T_right_base_front_camera_candidate': np.eye(4).tolist()}
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)/'candidate.json'
            p.write_text(json.dumps(data))
            _, report = candidate_calibration_for(p, '123', 'right')
            self.assertFalse(report['calibration_validated'])
            for serial, arm in [('wrong', 'right'), ('123', 'left')]:
                with self.assertRaises(ValueError):
                    candidate_calibration_for(p, serial, arm)
            data['calibration_validated'] = True
            p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                candidate_calibration_for(p, '123', 'right')


if __name__ == '__main__':
    unittest.main()
