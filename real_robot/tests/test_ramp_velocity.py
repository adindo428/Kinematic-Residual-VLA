import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from real_robot import ramp_velocity
from real_robot.ramp_velocity import MotionDetector, analyze, save, read_complete_record, seek_live_tail


def row(t, speed, status='estimated'):
    return {'host_processed_monotonic_s': t, 'pair': {'timestamp_s': t},
            'velocity_report': {'status': status, 'speed_mps': speed, 'velocity_mps': [speed, 0, 0]}}


class RampVelocityTest(unittest.TestCase):
    def test_time_weighting_and_no_gap_bridging(self):
        rows = [row(1, 1), row(1.1, 3), row(1.2, 0, 'unavailable'), row(1.3, 9), row(1.4, 9)]
        result = analyze(rows, 1, 1.4)
        self.assertAlmostEqual(result['valid_duration_s'], .2)
        self.assertAlmostEqual(result['mean_speed_mps'], 5.5)
        self.assertAlmostEqual(result['coverage'], .5)

    def test_automatic_motion_and_stop(self):
        detector = MotionDetector()
        events = []
        for t, speed in [(1, 0), (1.3, 0), (1.6, 0), (1.7, .05), (1.8, .08),
                         (1.9, .10), (2.0, .02), (2.2, .01), (2.5, .01)]:
            event = detector.update(row(t, speed))
            if event: events.append(event)
        self.assertEqual([e[0] for e in events], ['ready', 'started', 'complete'])
        self.assertEqual(events[-1][1], (1.7, 2.0))

    def test_lost_tracking_does_not_end_run(self):
        detector = MotionDetector()
        for t, speed in [(1, 0), (1.3, 0), (1.6, 0), (1.7, .05), (1.8, .06)]:
            detector.update(row(t, speed))
        self.assertIsNone(detector.update(row(2, 0, 'unavailable')))
        self.assertEqual(detector.update(row(2.4, 0, 'unavailable'))[0], 'invalid')

    def test_requires_stillness_before_start(self):
        detector = MotionDetector()
        for t in [1, 1.2, 1.4, 1.6]:
            self.assertIsNone(detector.update(row(t, .1)))
        self.assertEqual(detector.state, 'settling')

    def test_log_record_split_during_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fusion.jsonl'
            with path.open('wb', buffering=0) as writer, path.open('rb') as reader:
                writer.write(b'{"status": "esti')
                self.assertIsNone(read_complete_record(reader))
                self.assertEqual(reader.tell(), 0)
                writer.write(b'mated"}\n')
                self.assertEqual(read_complete_record(reader), {'status': 'estimated'})
                self.assertIsNone(read_complete_record(reader))

    def test_arming_during_record_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fusion.jsonl'
            with path.open('wb', buffering=0) as writer, path.open('rb') as reader:
                writer.write(b'{"old": 1}\n{"new": "' + b'x'*20000)
                seek_live_tail(reader)
                self.assertIsNone(read_complete_record(reader))
                writer.write(b'"}\n')
                self.assertEqual(read_complete_record(reader), {'new': 'x'*20000})
                seek_live_tail(reader)
                self.assertIsNone(read_complete_record(reader))

    def test_ball_mask_check_never_imports_tape_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            child = MagicMock()
            child.wait.return_value = 0
            with patch.object(ramp_velocity.subprocess, 'Popen', return_value=child) as popen, \
                 patch.object(ramp_velocity, 'wait_for_perception'), \
                 patch.object(ramp_velocity, 'stop_child'), \
                 patch.object(ramp_velocity, 'LeaseOwner'), \
                 patch.object(ramp_velocity, 'has_prompt', return_value=True), \
                 patch.object(ramp_velocity, 'find_tape_reference') as tape:
                code = ramp_velocity.main(['--object', 'striped_ball', '--mask-check',
                                           '--output', str(Path(directory)/'check')])
            self.assertEqual(code, 0)
            tape.assert_not_called()
            command = popen.call_args.args[0]
            self.assertIn('--new-mask', command)
            self.assertIn('--front-mask-only', command)
            self.assertTrue(command[command.index('--object-profile')+1].endswith('striped_ball/on_ramp'))

    def test_green_ball_uses_its_own_profile_and_only_front_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            child = MagicMock()
            child.wait.return_value = 0
            with patch.object(ramp_velocity.subprocess, 'Popen', return_value=child) as popen, \
                 patch.object(ramp_velocity, 'wait_for_perception'), \
                 patch.object(ramp_velocity, 'stop_child'), \
                 patch.object(ramp_velocity, 'LeaseOwner'), \
                 patch.object(ramp_velocity, 'has_prompt', return_value=True), \
                 patch.object(ramp_velocity, 'find_tape_reference') as tape:
                code = ramp_velocity.main(['--object', 'green_rubber_ball', '--mask-check',
                                           '--output', str(Path(directory)/'check')])
            self.assertEqual(code, 0)
            tape.assert_not_called()
            command = popen.call_args.args[0]
            self.assertIn('--new-mask', command)
            self.assertIn('--front-mask-only', command)
            self.assertTrue(command[command.index('--object-profile')+1].endswith('green_rubber_ball/on_ramp'))

            self.assertIn('--front-camera-only',command)

    def test_yllow_baseball_selects_a_fresh_independent_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            child = MagicMock()
            child.wait.return_value = 0
            with patch.object(ramp_velocity.subprocess, 'Popen', return_value=child) as popen, \
                 patch.object(ramp_velocity, 'wait_for_perception'), \
                 patch.object(ramp_velocity, 'stop_child'), \
                 patch.object(ramp_velocity, 'LeaseOwner'), \
                 patch.object(ramp_velocity, 'has_prompt', return_value=True), \
                 patch.object(ramp_velocity, 'find_tape_reference') as tape:
                code = ramp_velocity.main(['--object', 'yllow_baseball', '--mask-check',
                                           '--output', str(Path(directory)/'check')])
            self.assertEqual(code, 0)
            tape.assert_not_called()
            command = popen.call_args.args[0]
            self.assertIn('--new-mask', command)
            self.assertIn('--front-mask-only', command)
            self.assertTrue(command[command.index('--object-profile')+1].endswith('yllow_baseball/on_ramp'))

            self.assertIn('--front-camera-only',command)

    def test_summary_uses_sample_standard_deviation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = [dict(ramp='ramp_1', run=i, mean_speed_mps=v, valid_duration_s=1, marked_duration_s=1,
                         coverage=1, valid_samples=3, samples=[], start_monotonic_s=0) for i, v in [(1, 1), (2, 3)]]
            save(root, runs)
            result = json.loads((root/'summary.json').read_text())
            self.assertAlmostEqual(result['ramps']['ramp_1']['sample_std_dev_of_run_means_mps'], 2**.5)
            self.assertTrue((root/'velocity.svg').exists())


if __name__ == '__main__': unittest.main()
