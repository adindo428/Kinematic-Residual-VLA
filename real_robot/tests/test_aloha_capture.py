"""Control/capture concurrency checks using simulated serial buses only."""
import time
import unittest
import numpy as np
from real_robot.aloha_charuco_capture import LeaderFollowing, following_target


class Bus:
    def __init__(self, value):
        self.calibration = {}
        self.motor_names = [str(i) for i in range(9)]
        self.position = np.full(9, value, dtype=float)
        self.writes = []
        self.fail = False

    def read(self, register):
        if self.fail:
            raise OSError('Simulated serial failure')
        return np.ones(9) if register == 'Torque_Enable' else self.position.copy()

    def apply_calibration(self, raw, names):
        return raw.copy()

    def write(self, register, target):
        assert register == 'Goal_Position'
        assert np.max(np.abs(target-self.position)) <= 5
        self.writes.append(target.copy())
        self.position = target.copy()


class CaptureControlTests(unittest.TestCase):
    def test_relative_target_matches_lerobot_limit(self):
        np.testing.assert_allclose(following_target([100,-100,.2],[0,0,0]),[5,-5,.2])
        with self.assertRaises(ValueError):
            following_target([float('nan')],[0])

    def test_camera_wait_and_snapshot_do_not_pause_following(self):
        leader, follower = Bus(20), Bus(0)
        control = LeaderFollowing(leader, follower)
        control.start()
        try:
            time.sleep(.05)
            self.assertEqual(len(follower.writes),0)  # Starts with no movement.
            self.assertTrue(control.toggle())
            time.sleep(.4)  # Simulate a blocked camera acquisition/image save.
            control.snapshot()
            self.assertGreaterEqual(len(follower.writes),6)
            self.assertTrue(control.enabled.is_set())
            np.testing.assert_allclose(follower.position,leader.position)
            control.toggle()
            n = len(follower.writes)
            time.sleep(.1)
            self.assertEqual(len(follower.writes),n)
        finally:
            control.close()
        n = len(follower.writes)
        time.sleep(.05)
        self.assertEqual(len(follower.writes),n)

    def test_serial_failure_disables_following(self):
        leader, follower = Bus(1), Bus(0)
        control = LeaderFollowing(leader, follower)
        control.start()
        try:
            control.toggle()
            follower.fail = True
            self.assertTrue(control.stop.wait(1))
            self.assertFalse(control.enabled.is_set())
            with self.assertRaises(RuntimeError):
                control.snapshot()
        finally:
            control.close()


if __name__ == '__main__':
    unittest.main()
