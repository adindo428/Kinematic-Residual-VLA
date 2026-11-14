from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from real_robot.calibrate_leader_gripper import (
    measure_endpoints,
    sample_endpoint,
    save_calibration,
    with_gripper_endpoints,
)


def calibration():
    return dict(motor_names=['waist', 'gripper'], calib_mode=['DEGREE', 'LINEAR'],
                start_pos=[2068, 2013], end_pos=[-1025, 2555],
                homing_offset=[2048, -2048], drive_mode=[1, 0])


class LeaderGripperCalibrationTests(unittest.TestCase):
    def test_follower_open_only_preserves_closed_and_other_joints(self):
        original = calibration()
        original['start_pos'][1] = 2475
        original['end_pos'][1] = 3113
        with patch('builtins.input') as prompt, patch('builtins.print'), \
                patch('real_robot.calibrate_leader_gripper.sample_endpoint', return_value=3143) as sample:
            updated = measure_endpoints(object(), original, 'follower', open_only=True)
        prompt.assert_called_once()
        sample.assert_called_once()
        self.assertEqual(updated['end_pos'][1], 3143)
        updated['end_pos'][1] = 3113
        self.assertEqual(updated, original)

    def test_reported_position_maps_to_open_after_recalibration(self):
        original = calibration()
        before = deepcopy(original)
        updated = with_gripper_endpoints(original, 2013, 2631)
        self.assertEqual(original, before)
        self.assertGreater((2631 - original['start_pos'][1]) /
                           (original['end_pos'][1] - original['start_pos'][1]) * 100, 110)
        self.assertEqual((2631 - updated['start_pos'][1]) /
                         (updated['end_pos'][1] - updated['start_pos'][1]) * 100, 100)
        updated['end_pos'][1] = original['end_pos'][1]
        self.assertEqual(updated, original)

    def test_reject_invalid_endpoints(self):
        for closed, opened in [(2000, 2000), (2631, 2013), (2013, 6727)]:
            with self.subTest(closed=closed, opened=opened), self.assertRaises(ValueError):
                with_gripper_endpoints(calibration(), closed, opened)

    def test_save_preserves_backup_and_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'right_leader.json'
            original = json.dumps(calibration()).encode()
            path.write_bytes(original)
            updated = with_gripper_endpoints(calibration(), 2013, 2631)
            backup = save_calibration(path, original, updated)
            self.assertEqual(backup.read_bytes(), original)
            self.assertEqual(json.loads(path.read_bytes()), updated)
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                save_calibration(path, original, updated)

    def test_sampling_reads_raw_without_writes(self):
        class Bus:
            def read(self, register, motor):
                if motor != 'gripper':
                    raise AssertionError('Unexpected motor')
                return [0 if register == 'Torque_Enable' else 2631]

        with patch('real_robot.calibrate_leader_gripper.time.sleep'):
            self.assertEqual(sample_endpoint(Bus()), 2631)

    def test_sampling_rejects_enabled_torque(self):
        class Bus:
            def read(self, register, motor):
                if register != 'Torque_Enable':
                    raise AssertionError('Position read while torque is enabled')
                return [1]

        with self.assertRaisesRegex(RuntimeError, 'torque is enabled'):
            sample_endpoint(Bus())

    def test_sampling_rejects_motion(self):
        class Bus:
            position = 2000

            def read(self, register, motor):
                if register == 'Torque_Enable':
                    return [0]
                self.position += 10
                return [self.position]

        with patch('real_robot.calibrate_leader_gripper.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'moved while sampling'):
                sample_endpoint(Bus())


if __name__ == '__main__':
    unittest.main()
