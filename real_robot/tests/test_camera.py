"""Hardware-free checks for startup pairing and bounded frame waits."""
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import numpy as np
from real_robot.camera import RGBDCamera


def frames(color_ms, depth_ms):
    intrinsics = SimpleNamespace(width=4, height=3, fx=2., fy=2., ppx=2., ppy=1.,
                                 coeffs=[0.] * 5, model='none')
    profile = Mock()
    profile.as_video_stream_profile.return_value.get_intrinsics.return_value = intrinsics
    def frame(timestamp, array):
        return SimpleNamespace(get_timestamp=lambda: timestamp,
                               get_frame_timestamp_domain=lambda: 'hardware_clock',
                               get_frame_number=lambda: 1, get_data=lambda: array,
                               profile=profile)
    color = frame(color_ms, np.zeros((3, 4, 3), np.uint8))
    depth = frame(depth_ms, np.full((3, 4), 1000, np.uint16))
    return SimpleNamespace(get_color_frame=lambda: color, get_depth_frame=lambda: depth)


class CameraTests(unittest.TestCase):
    def camera(self):
        cam = RGBDCamera.__new__(RGBDCamera)
        cam.serial, cam.scale, cam.frames_started = 'test', .001, False
        cam.pipeline, cam.align = Mock(), Mock()
        cam.align.process.side_effect = lambda x: x
        cam.rs = SimpleNamespace(rs2_deproject_pixel_to_point=lambda i, pixel, z: [0, 0, z])
        return cam

    def test_startup_timeout_and_mismatched_pair_are_not_accepted(self):
        cam = self.camera()
        cam.pipeline.try_wait_for_frames.side_effect = [
            (False, None), (True, frames(1000, 1228)), (True, frames(1400, 1410))]
        bgr, depth, metadata, _ = cam.read()
        self.assertEqual(metadata['capture_wait_timeouts'], 1)
        self.assertEqual(metadata['skipped_unsynchronized_pairs'], 1)
        self.assertAlmostEqual(metadata['rgb_depth_skew_s'], .01)
        np.testing.assert_allclose(depth, 1.)
        self.assertTrue(cam.frames_started)

    def test_no_frames_eventually_aborts(self):
        cam = self.camera()
        cam.pipeline.try_wait_for_frames.return_value = (False, None)
        with patch('real_robot.camera.time.monotonic', side_effect=[0., 10.1]):
            with self.assertRaises(RuntimeError): cam.read()

    def test_persistent_bad_pair_aborts_without_relaxing_tolerance(self):
        cam = self.camera()
        cam.pipeline.try_wait_for_frames.return_value = (True, frames(1000, 1228))
        with patch('real_robot.camera.time.monotonic', side_effect=[0., .1, 10.1]):
            with self.assertRaises(RuntimeError): cam.read()


if __name__ == '__main__':
    unittest.main()
