import unittest

import numpy as np
import cv2

from real_robot.recover_marker_session import marker_mask, measure_marker


class MarkerRecoveryTests(unittest.TestCase):
    def image(self):
        hsv = np.zeros((480,640,3),np.uint8)
        hsv[300:310,250:300] = (80,200,100)
        return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    def test_background_and_ambiguous_markers_rejected(self):
        image = self.image()
        self.assertIsNotNone(marker_mask(image)[0])
        image[350:360,350:400] = image[300:310,250:300]
        self.assertIsNone(marker_mask(image)[0])
        self.assertIsNone(marker_mask(np.zeros_like(image))[0])

    def test_sparse_depth_is_measured_not_filled(self):
        mask,_ = marker_mask(self.image())
        depth = np.zeros(mask.shape,np.float32)
        depth[300:302,250:300] = .7
        point,quality = measure_marker(mask,depth,lambda p,z:[p[0]*z,p[1]*z,z])
        self.assertAlmostEqual(point[2],.7,places=6)
        self.assertTrue(quality['sparse_depth_recovery'])
        depth[:] = 0
        self.assertIsNone(measure_marker(mask,depth,lambda p,z:[0,0,z])[0])

    def test_inconsistent_depth_rejected(self):
        mask,_ = marker_mask(self.image())
        depth = np.zeros(mask.shape,np.float32)
        depth[300:305,250:300] = .6
        depth[305:310,250:300] = .9
        self.assertIsNone(measure_marker(mask,depth,lambda p,z:[0,0,z])[0])


if __name__ == '__main__':
    unittest.main()
