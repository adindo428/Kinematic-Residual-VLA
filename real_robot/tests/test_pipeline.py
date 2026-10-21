import unittest
import numpy as np
import cv2
from real_robot.geometry import validate_transform, transform_point, fit_rigid_transform
from real_robot.tracking import VelocityTracker
from real_robot.perception import detect_mask, depth_measurement


class GeometryTests(unittest.TestCase):
    def test_rigid_fit_recovers_rotation_and_translation(self):
        points = np.array([[0, 0, 0], [.2, 0, 0], [0, .2, 0], [.2, .2, 0], [.1, .1, .1]])
        t = np.array([[0, -1, 0, .3], [1, 0, 0, -.1], [0, 0, 1, .2], [0, 0, 0, 1.]])
        targets = np.array([transform_point(t, p) for p in points])
        actual, errors = fit_rigid_transform(points, targets)
        np.testing.assert_allclose(actual, t, atol=1e-12)
        self.assertLess(max(errors), 1e-12)

    def test_degenerate_calibration_rejected(self):
        with self.assertRaises(ValueError):
            fit_rigid_transform(np.zeros((5, 3)), np.zeros((5, 3)))
        mirror = np.eye(4); mirror[0, 0] = -1
        with self.assertRaises(ValueError): validate_transform(mirror)

    def test_wrist_ego_motion_requires_transform_before_velocity(self):
        tracker = VelocityTracker(frame='robot_base')
        fixed_object = np.array([.2, .1, .4])
        for i in range(100):
            t = np.eye(4); t[:3, 3] = [.001 * i, 0, 0]
            camera_point = fixed_object - t[:3, 3]
            state = tracker.update(i / 30, transform_point(t, camera_point))
        self.assertTrue(state.ready)
        np.testing.assert_allclose(state.velocity_mps, 0, atol=1e-12)


class TrackingTests(unittest.TestCase):
    def test_stationary_update_reduces_noise_and_clears_on_missing_measurements(self):
        rng = np.random.default_rng(10)
        tracker, baseline = VelocityTracker(), VelocityTracker(stationary_updates=False)
        updated, original = [], []
        for i in range(120):
            point = np.array([.2,.1,.8])+rng.normal(0,.001,3)
            state = tracker.update(i/15,point)
            raw = baseline.update(i/15,point)
            if i>30:
                updated.append(np.linalg.norm(state.velocity_mps))
                original.append(np.linalg.norm(raw.velocity_mps))
        self.assertTrue(state.stationary_assumed)
        self.assertLess(np.percentile(updated,95),.1*np.percentile(original,95))
        self.assertGreater(np.linalg.eigvalsh(state.covariance).min(),0)
        state = tracker.update(8,None)
        self.assertFalse(state.stationary_assumed)
        self.assertFalse(state.ready)
        tracker.update(8.4,None)
        state = tracker.update(8.5,[.2,.1,.8])
        self.assertFalse(state.stationary_assumed)
        self.assertFalse(state.ready)

    def test_stationary_update_releases_on_slow_and_fast_motion_then_reenters(self):
        for speed, max_delay in [(.005,1.2),(.01,.8),(.08,.3)]:
            with self.subTest(speed=speed):
                rng = np.random.default_rng(21)
                tracker = VelocityTracker()
                release, velocities, settled = None, [], []
                for i in range(180):
                    time = i/15
                    displacement = speed*min(max(time-3,0),4)
                    point = np.array([.2+displacement,.1,.8])+rng.normal(0,.001,3)
                    state = tracker.update(time,point)
                    if i==40: self.assertTrue(state.stationary_assumed)
                    if time>3 and release is None and not state.stationary_assumed:
                        release=time-3
                    if 5<time<6.5: velocities.append(state.velocity_mps[0])
                    if time>9: settled.append(state.stationary_assumed)
                self.assertIsNotNone(release)
                self.assertLessEqual(release,max_delay)
                self.assertAlmostEqual(np.mean(velocities),speed,delta=.006)
                self.assertGreater(np.mean(settled),.8)

    def test_fixed_base_velocity_rotates_without_translation_offset(self):
        camera_tracker = VelocityTracker()
        base_tracker = VelocityTracker(frame='robot_base')
        transform = np.array([[0, -1, 0, .4], [1, 0, 0, -.2],
                              [0, 0, 1, .8], [0, 0, 0, 1.]])
        rotation = transform[:3, :3]
        camera_velocity = np.array([.06, -.02, .01])
        for i in range(90):
            timestamp = i/15
            point = np.array([.1, .2, 1.]) + timestamp*camera_velocity
            camera = camera_tracker.update(timestamp, point)
            base = base_tracker.update(timestamp, transform_point(transform, point))
        self.assertTrue(base.ready)
        np.testing.assert_allclose(base.velocity_mps, rotation@camera.velocity_mps, atol=1e-12)
        np.testing.assert_allclose(base.velocity_mps, [.02, .06, .01], atol=1e-6)
        self.assertAlmostEqual(np.linalg.norm(base.velocity_mps), np.linalg.norm(camera.velocity_mps))

    def test_velocity_accuracy_with_irregular_timestamps_and_noise(self):
        rng = np.random.default_rng(7)
        tracker = VelocityTracker()
        velocity = np.array([.08, -.03, .015]); time = 0; errors = []
        for i in range(150):
            time += rng.uniform(.025, .05)
            point = [.2, .1, .8] + velocity * time + rng.normal(0, .001, 3)
            state = tracker.update(time, point)
            if i > 30:
                self.assertTrue(state.ready)
                errors.append(np.linalg.norm(state.velocity_mps - velocity))
        self.assertLess(np.sqrt(np.mean(np.square(errors))), .015)
        self.assertGreater(np.linalg.eigvalsh(state.covariance).min(), 0)

    def test_outlier_is_rejected_without_target_jump(self):
        tracker = VelocityTracker()
        for i in range(30): state = tracker.update(i / 30, [.2, .1, .8])
        state = tracker.update(1., [1, 1, 1])
        self.assertFalse(state.accepted)
        self.assertFalse(state.ready)
        np.testing.assert_allclose(state.position_m, [.2, .1, .8])

    def test_dropout_and_reacquisition_reset_velocity(self):
        tracker = VelocityTracker()
        for i in range(30): tracker.update(i / 30, [i * .002, .1, .8])
        self.assertFalse(tracker.update(1., None).ready)
        self.assertIsNone(tracker.update(1.4, None))
        state = tracker.update(1.5, [.3, .1, .8])
        self.assertFalse(state.ready)
        np.testing.assert_allclose(state.velocity_mps, 0)

    def test_invalid_time_and_coordinates_rejected(self):
        tracker = VelocityTracker(); tracker.update(1., [0, 0, 1])
        for t in [1., .9, float('nan')]:
            with self.assertRaises(ValueError): tracker.update(t, [0, 0, 1])
        with self.assertRaises(ValueError): tracker.update(2., [0, float('nan'), 1])

    def test_stationary_noise_does_not_create_large_velocity(self):
        rng = np.random.default_rng(2); tracker = VelocityTracker()
        for i in range(80):
            state = tracker.update(i / 15, np.array([.2, .1, .8]) + rng.normal(0, .001, 3))
        self.assertTrue(state.ready)
        self.assertLess(np.linalg.norm(state.velocity_mps), .02)


class PerceptionTests(unittest.TestCase):
    def test_upright_blue_roll_ignores_clipped_cable(self):
        image = np.zeros((150, 200, 3), np.uint8)
        cv2.ellipse(image, (110, 75), (10, 23), -15, 0, 360, (200, 80, 20), -1)
        cv2.ellipse(image, (108, 75), (4, 13), -15, 0, 360, (150, 150, 150), -1)
        cv2.rectangle(image, (0, 35), (12, 110), (200, 80, 20), -1)
        mask = detect_mask(image, 'blue_tape')
        self.assertIsNotNone(mask)
        self.assertEqual(mask[75, 5], 0)
        self.assertEqual(mask[75, 108], 0)
        self.assertGreater(np.count_nonzero(mask[:, 90:130]), 80)
        # A second fully visible roll must still be treated as ambiguous.
        cv2.ellipse(image, (160, 75), (10, 23), -15, 0, 360, (200, 80, 20), -1)
        self.assertIsNone(detect_mask(image, 'blue_tape'))

    def test_blue_tape_roll_preserves_hole_and_ignores_orange(self):
        image = np.zeros((150, 200, 3), np.uint8)
        cv2.circle(image, (60, 70), 30, (200, 80, 20), -1)
        cv2.circle(image, (60, 70), 17, (150, 150, 150), -1)
        cv2.circle(image, (150, 70), 30, (0, 140, 255), -1)
        mask = detect_mask(image, 'blue_tape')
        self.assertIsNotNone(mask)
        self.assertEqual(mask[70, 60], 0)
        self.assertEqual(mask[70, 84], 255)
        self.assertEqual(mask[70, 150], 0)
        depth = np.full((150, 200), .9)
        depth[mask > 0] = .7
        point, _ = depth_measurement(mask, depth, lambda pixel, z: [0, 0, z])
        self.assertAlmostEqual(point[2], .7)

    def test_blue_tape_strip_ambiguity_and_clipping(self):
        image = np.zeros((120, 180, 3), np.uint8)
        image[30:60, 20:80] = (200, 80, 20)
        self.assertIsNotNone(detect_mask(image, 'blue_tape'))
        self.assertIsNone(detect_mask(image, 'blue_tape', roi=[40, 0, 100, 100]))
        image[80:110, 100:160] = (200, 80, 20)
        self.assertIsNone(detect_mask(image, 'blue_tape'))

    def test_orange_ball_ignores_red_and_orange_rectangles(self):
        image = np.zeros((180, 240, 3), np.uint8)
        cv2.circle(image, (45, 45), 20, (0, 140, 255), -1)
        cv2.circle(image, (150, 45), 20, (0, 0, 255), -1)
        cv2.rectangle(image, (120, 100), (170, 150), (0, 140, 255), -1)
        mask = detect_mask(image, 'orange_ball')
        self.assertIsNotNone(mask)
        self.assertEqual(mask[45, 45], 255)
        self.assertEqual(mask[45, 150], 0)
        self.assertEqual(mask[125, 145], 0)

    def test_orange_ball_rejects_ambiguity_and_clipping(self):
        image = np.zeros((150, 200, 3), np.uint8)
        cv2.circle(image, (45, 65), 25, (0, 140, 255), -1)
        cv2.circle(image, (145, 65), 25, (0, 140, 255), -1)
        self.assertIsNone(detect_mask(image, 'orange_ball'))
        self.assertIsNotNone(detect_mask(image, 'orange_ball', roi=[0, 0, 100, 150]))
        self.assertIsNone(detect_mask(image, 'orange_ball', roi=[30, 0, 70, 150]))
        image[:] = 0
        cv2.circle(image, (10, 65), 25, (0, 140, 255), -1)
        self.assertIsNone(detect_mask(image, 'orange_ball'))

    def test_mask_depth_and_units(self):
        image = np.zeros((100, 100, 3), np.uint8)
        image[20:60, 30:70] = [0, 0, 255]
        mask = detect_mask(image, 'hsv')
        depth = np.full((100, 100), .75)
        depth[30:35, 40:45] = 0
        project = lambda pixel, z: [(pixel[0]-50)*z/100, (pixel[1]-50)*z/100, z]
        p, q = depth_measurement(mask, depth, project)
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p[2], .75)
        self.assertLess(abs(p[0]), .01)
        self.assertLess(p[1], 0)
        self.assertIsNone(q['reason'])

    def test_ambiguous_color_objects_rejected(self):
        image = np.zeros((100, 100, 3), np.uint8)
        image[10:40, 10:40] = [0, 0, 255]
        image[60:90, 60:90] = [0, 0, 255]
        self.assertIsNone(detect_mask(image, 'hsv'))
        self.assertIsNotNone(detect_mask(image, 'hsv', roi=[0, 0, 50, 50]))

    def test_missing_and_discontinuous_depth_rejected(self):
        mask = np.full((30, 30), 255, np.uint8)
        for depth in [np.zeros((30, 30)), np.full((30, 30), np.nan),
                      np.vstack([np.full((15, 30), .4), np.full((15, 30), .8)])]:
            p, _ = depth_measurement(mask, depth, lambda pixel, z: [0, 0, z])
            self.assertIsNone(p)

    def test_side_depth_range_accepts_sixty_cm_without_accepting_bad_depth(self):
        mask = np.full((30, 30), 255, np.uint8)
        project = lambda pixel, z: [0, 0, z]
        depth = np.full((30, 30), .60)
        point, quality = depth_measurement(mask, depth, project, .07, .50)
        self.assertIsNone(point)
        self.assertEqual(quality['valid_fraction'], 0)
        self.assertAlmostEqual(quality['nonzero_depth_median_m'], .60)
        self.assertEqual(quality['accepted_depth_range_m'], [.07, .50])
        point, quality = depth_measurement(mask, depth, project, .07, .80)
        self.assertAlmostEqual(point[2], .60)
        for bad in (np.zeros((30,30)), np.full((30,30), .9),
                    np.vstack([np.full((15,30), .55), np.full((15,30), .75)])):
            point, _ = depth_measurement(mask, bad, project, .07, .80)
            self.assertIsNone(point)

    def test_marker_identity(self):
        marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 7, 100)
        image = np.full((150, 150), 255, np.uint8); image[25:125, 25:125] = marker
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.assertIsNotNone(detect_mask(bgr, marker_id=7))
        self.assertIsNone(detect_mask(bgr, marker_id=0))


if __name__ == '__main__':
    unittest.main()
