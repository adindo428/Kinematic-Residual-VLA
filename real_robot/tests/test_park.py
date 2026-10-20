import unittest
import numpy as np
from real_robot.aloha_park import ParkFollower


class ParkFollowerTests(unittest.TestCase):
    def test_retains_holding_goal_and_limits_each_step(self):
        held = np.deg2rad([0, 24, -17, 0, 80, 0]); sag = np.deg2rad([0, 1.6, 1.2, 0, 0, 0])
        servo = ParkFollower(held)
        np.testing.assert_allclose(servo.step(held, held+sag, 1/30), held)
        last = held.copy()
        for i in range(100):
            nominal = held-np.deg2rad([0, i/100, 0, 0, 0, 0])
            goal = servo.step(nominal, nominal+sag, 1/30)
            self.assertLessEqual(np.max(np.abs(goal-last)), np.deg2rad(3)/30+1e-12)
            last = goal

    def test_bad_feedback_timing_and_out_of_range_goal_stop_before_update(self):
        held = np.zeros(6); servo = ParkFollower(held)
        with self.assertRaises(ValueError): servo.step(held, held, .2)
        with self.assertRaises(RuntimeError): servo.step(held, held+.1, 1/30)
        outside = np.array([0, 2, 0, 0, 0, 0])
        with self.assertRaises(RuntimeError): servo.step(outside, outside, 1/30)
        np.testing.assert_allclose(servo.goal, held)


if __name__ == '__main__': unittest.main()
