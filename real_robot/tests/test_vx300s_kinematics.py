import unittest
import numpy as np
from real_robot.vx300s_kinematics import forward, inverse


class KinematicsTests(unittest.TestCase):
    def test_zero_pose_matches_manufacturer_ee_reference(self):
        # Official ee_gripper_link home M is x=.536494, z=.42705;
        # its standard gripper offset is .1072 m from gripper_link.
        t = forward(np.zeros(6), [.1072, 0, 0])
        np.testing.assert_allclose(t[:3, 3], [.536494, 0, .42705], atol=1e-9)
        np.testing.assert_allclose(t[:3, :3], np.eye(3), atol=1e-9)

    def test_tcp_offset_rotates_with_wrist(self):
        q = np.deg2rad([30, 10, -40, 20, 70, 15])
        g = forward(q)
        tip = forward(q, [.16, 0, 0])
        np.testing.assert_allclose(tip[:3, 3]-g[:3, 3], g[:3, 0]*.16)

    def test_inverse_preserves_tcp_pose(self):
        q = np.deg2rad([10, -20, 30, 15, 65, -10])
        offset = [.16, 0, 0]
        target = forward(q, offset)
        solved = inverse(target, q+.02, offset)
        np.testing.assert_allclose(forward(solved, offset), target, atol=1e-5)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            forward(np.zeros(5))
        with self.assertRaises(ValueError):
            forward(np.zeros(6), [float('nan'), 0, 0])
