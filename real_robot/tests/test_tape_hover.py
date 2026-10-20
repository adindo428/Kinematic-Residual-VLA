import unittest
import numpy as np
import cv2
from real_robot.aloha_tape_hover import CurrentGuard, HoverServo, TapeArm, signed, observation_transform
from real_robot.geometry import transform_point
from real_robot.perception import detect_mask, depth_measurement


class TapeHoverTests(unittest.TestCase):
    def test_signed_motor_units(self):
        self.assertEqual(signed(65535, 16), -1)
        self.assertEqual(signed(0xffffffff, 32), -1)
        self.assertEqual(signed(2048, 32), 2048)

    def test_wrist_motion_is_compensated_before_tracking_object(self):
        x = np.eye(4); x[:3, 3] = [.07, .01, .08]
        plan = {'observation_camera': 'right_wrist', 'T_gripper_wrist_camera': x.tolist()}
        point = np.array([.47, .02, .04])
        measurements = []
        for q in [np.deg2rad([0, 18, -7, 0, 76, 0]), np.deg2rad([0, 25, -16, 0, 81, 0])]:
            t = observation_transform(plan, q)
            camera_point = transform_point(np.linalg.inv(t), point)
            measurements.append(camera_point)
            np.testing.assert_allclose(transform_point(t, camera_point), point, atol=1e-12)
        self.assertGreater(np.linalg.norm(measurements[0]-measurements[1]), .005)

    def test_wrist_saturation_filter_excludes_blue_tinted_table_reflection(self):
        hsv = np.full((480, 640, 3), (20, 120, 140), np.uint8)
        cv2.ellipse(hsv, (295, 315), (50, 90), 0, 0, 360, (110, 100, 220), -1)
        cv2.circle(hsv, (370, 220), 55, (110, 230, 140), -1)
        depth = np.full((480, 640), .23)
        cv2.circle(depth, (370, 220), 55, .16, -1)
        mask = detect_mask(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), 'blue_tape', hsv_low=(90, 150, 45))
        p, quality = depth_measurement(mask, depth, lambda xy, z: [xy[0]/1000, xy[1]/1000, z], .07, .5)
        self.assertIsNotNone(p)
        self.assertAlmostEqual(quality['depth_m'], .16)
        np.testing.assert_allclose(quality['pixel'], [370, 220], atol=1)
        self.assertFalse(mask[315, 295])

    def test_current_guard_handles_both_paired_shoulders_and_dwell(self):
        for index in (1, 2):
            guard = CurrentGuard(); currents = np.zeros(9); currents[index] = -.95
            for _ in range(20): guard.check(currents, 1/30)
            with self.assertRaises(RuntimeError):
                for _ in range(5): guard.check(currents, 1/30)
        guard = CurrentGuard(); currents = np.zeros(9); currents[2] = 1.6
        with self.assertRaises(RuntimeError): guard.check(currents, 1/30)
        with self.assertRaises(ValueError): guard.check(np.zeros(9), .2)

    def test_servo_limits_compensation_and_command_slew_with_loaded_feedback(self):
        initial = np.deg2rad([0, 20, -10, 0, 80, 0])
        servo = HoverServo(initial, .01)
        bias = np.deg2rad([0, 1., .8, 0, 0, 0]); actual = initial+bias
        last = initial.copy()
        for _ in range(600):
            goal = servo.step(initial, actual, 1/30)
            self.assertLessEqual(np.max(np.abs(goal-last)), np.deg2rad(3.5)/30+1e-10)
            self.assertLessEqual(np.max(np.abs(servo.correction)), np.deg2rad(1.5)+1e-10)
            actual += .15*(goal+bias-actual); last = goal
        self.assertLess(np.max(np.abs(actual-initial)), np.deg2rad(.1))
        with self.assertRaises(RuntimeError): servo.step(initial, actual+.1, 1/30)
        with self.assertRaises(ValueError): servo.step(initial, actual, .2)

    def test_activation_refuses_live_torque_before_any_goal_write(self):
        arm = TapeArm.__new__(TapeArm)
        arm.read_register = lambda motor, address, width: 1
        arm.goal = lambda q: self.fail('Position write must not precede torque-off verification')
        with self.assertRaises(RuntimeError): arm.enable_supported_hold(np.zeros(6))

    def test_resumed_servo_retains_existing_load_holding_goal(self):
        q = np.deg2rad([0, 20, -10, 0, 80, 0])
        held = q+np.deg2rad([0, -.9, -1.2, 0, 0, 0])
        servo = HoverServo(q, .01, held)
        np.testing.assert_allclose(servo.step(q, q, 1/30), held, atol=1e-12)
        with self.assertRaises(ValueError): HoverServo(q, .01, q+.1)

    def test_activation_uses_measured_holding_goals_before_torque(self):
        arm = TapeArm.__new__(TapeArm); events = []
        q = np.zeros(6); raw = {i: 2048 for i in range(1, 10)}; raw[9] = 2466
        arm.read_register = lambda motor, address, width: (raw[motor] if address in (116, 132) else 0)
        arm.telemetry = lambda: (q.copy(), np.zeros(9), raw.copy())
        arm.goal = lambda target: (np.testing.assert_allclose(target, q), events.append('hold goals'))
        def gripper(port, motor, address, value):
            self.assertEqual((motor, address, value), (9, 116, raw[9]))
            events.append('unchanged gripper goal'); return 0, 0
        arm.packet = type('Packet', (), {'write4ByteTxRx': staticmethod(gripper)})()
        arm.dxl = type('SDK', (), {'COMM_SUCCESS': 0})(); arm.port = object()
        arm.torque_packet = lambda value: events.append(('torque', value))
        arm.check_health = lambda: events.append('health')
        arm.positions = lambda: q.copy()
        arm.enable_supported_hold(q)
        self.assertEqual(events, ['hold goals', 'unchanged gripper goal', ('torque', 1), 'health'])

    def test_activation_rejects_nonlocal_holding_goal_without_enabling_torque(self):
        arm = TapeArm.__new__(TapeArm); q = np.zeros(6); raw = {i: 2048 for i in range(1, 10)}
        arm.read_register = lambda motor, address, width: (2060 if address == 116 else (2048 if address == 132 else 0))
        arm.telemetry = lambda: (q.copy(), np.zeros(9), raw.copy())
        arm.goal = lambda q: None
        arm.packet = type('Packet', (), {'write4ByteTxRx': staticmethod(lambda *args: (0, 0))})()
        arm.dxl = type('SDK', (), {'COMM_SUCCESS': 0})(); arm.port = object()
        arm.torque_packet = lambda value: self.fail('Must not enable torque with a displaced holding goal')
        with self.assertRaises(RuntimeError): arm.enable_supported_hold(q)


if __name__ == '__main__':
    unittest.main()
