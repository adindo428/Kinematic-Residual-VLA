"""Grasp recording failure containment and coordinate/label checks; no hardware."""
import json
import hashlib
from pathlib import Path
import queue
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from real_robot.grasp_capture import TeachingControl, event_snapshot
from real_robot.grasp_prior import EVENTS, fit_session, follower_pose, fresh_observation, tape_frame
from real_robot.vx300s_kinematics import forward


class FakeRobot:
    def __init__(self):
        self.torque = np.ones(9, dtype=int)
        self.actual = np.zeros(9)
        self.leader = np.zeros(9)
        self.writes = []
        self.fail = False

    def health(self):
        if self.fail:
            raise OSError('serial failed')
        return {'torque': self.torque.copy(), 'hardware_errors': np.zeros(9, dtype=int)}

    def read(self):
        if self.fail:
            raise OSError('serial failed')
        return {'leader_raw_ticks': [2048]*9, 'follower_raw_ticks': [2048]*9,
                'leader_calibrated': self.leader.tolist(), 'follower_calibrated': self.actual.tolist(),
                'currents_a': [0.]*9, 'velocities_rad_s': [0.]*9,
                'voltages_v': [12.]*9, 'temperatures_c': [25]*9}

    def goal(self, target):
        self.writes.append(np.array(target))
        self.actual = np.array(target)


def wait_until(predicate, timeout=1):
    end = time.monotonic()+timeout
    while not predicate():
        if time.monotonic() >= end:
            raise AssertionError('Condition did not become true')
        time.sleep(.005)


class ControlTests(unittest.TestCase):
    def start_control(self, execute=True):
        io = FakeRobot()
        control = TeachingControl(io, execute, [.16, 0, 0])
        control.start()
        self.addCleanup(control.close)
        wait_until(lambda: control.latest is not None)
        return io, control

    def test_preview_and_paused_start_never_command(self):
        io, control = self.start_control(execute=False)
        with self.assertRaisesRegex(ValueError, 'Preview only'):
            control.toggle()
        self.assertEqual(io.writes, [])

    def test_enable_requires_matching_leader_and_all_torque(self):
        io, control = self.start_control()
        io.torque[0] = 0
        with self.assertRaisesRegex(ValueError, 'holding torque'):
            control.toggle()
        io.torque[:] = 1
        io.leader[:] = 10
        wait_until(lambda: control.latest['leader_calibrated'][0] == 10)
        with self.assertRaisesRegex(ValueError, 'Match leader'):
            control.toggle()
        self.assertEqual(io.writes, [])

    def test_relative_clamp_pause_and_no_writes_after_close(self):
        io, control = self.start_control()
        self.assertTrue(control.toggle())
        io.leader[:] = 50
        wait_until(lambda: len(io.writes) >= 2)
        self.assertLessEqual(np.max(np.abs(io.writes[0])), 5)
        for a, b in zip(io.writes, io.writes[1:]):
            self.assertLessEqual(np.max(np.abs(b-a)), 5)
        control.close()
        count = len(io.writes)
        time.sleep(.05)
        self.assertEqual(len(io.writes), count)

    def test_serial_failure_stops_goals(self):
        io, control = self.start_control()
        control.toggle()
        io.fail = True
        self.assertTrue(control.stop.wait(1))
        self.assertFalse(control.enabled)
        with self.assertRaisesRegex(RuntimeError, 'serial failed'):
            control.snapshot()

    def test_lost_ui_heartbeat_stops_goals(self):
        io, control = self.start_control()
        control.toggle()
        with control.lock:
            control.heartbeat = time.monotonic()-1
        self.assertTrue(control.stop.wait(1))
        self.assertIn('heartbeat', control.error)

    def test_log_overflow_stops_instead_of_silently_dropping(self):
        io, control = self.start_control()
        control.toggle()
        with control.lock:
            while not control.pending.full():
                control.pending.put_nowait({})
        self.assertTrue(control.stop.wait(1))
        self.assertFalse(control.enabled)


class PriorTests(unittest.TestCase):
    def test_encoder_pose_uses_raw_model_convention(self):
        raw = np.array([2100, 2600, 2600, 1800, 1800, 2048, 2300, 2200, 2800])
        q, pose = follower_pose(raw, [.16, 0, 0])
        expected = (raw[[0,1,3,5,6,7]]-2048)*2*np.pi/4096
        np.testing.assert_allclose(q, expected)
        np.testing.assert_allclose(pose, forward(expected, [.16, 0, 0]))

    def test_anchor_rejects_stale_identity_and_wrong_rig(self):
        record = {'schema': 'front_side_observation_v1', 'running': True, 'stale': False,
                  'rig_sha256': 'rig', 'output_frame': 'right_base_estimate',
                  'timing': {'oldest_host_arrival_monotonic_s': 10},
                  'fusion': {'stale': False, 'identity_recent': True, 'sources': ['front', 'side'],
                             'output_frame': 'right_base_estimate', 'position_output_m': [.3, 0, .05]}}
        np.testing.assert_allclose(fresh_observation(record, 'rig', 10.1), [.3, 0, .05])
        for digest, now in [('wrong', 10.1), ('rig', 11), ('rig', 9)]:
            with self.assertRaises(ValueError):
                fresh_observation(record, digest, now)
        record['fusion']['identity_recent'] = False
        with self.assertRaises(ValueError):
            fresh_observation(record, 'rig', 10.1)

    def test_relative_grasp_transfers_heading_and_failed_episodes_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'episodes').mkdir()
            (root/'manifest.json').write_text(json.dumps({'schema': 'blue_tape_grasp_capture_v1',
                'rig_sha256': 'rig', 'tcp_offset_m': [.16, 0, 0]}))
            local = np.eye(4)
            local[:3, 3] = [.01, .02, .03]
            for i, heading in enumerate([0, 90, 180]):
                reference = tape_frame([.3+i*.05, .1, .05], heading)
                events = {name: {'sample': {'T_base_tcp_candidate': (reference@local).tolist(),
                    'follower_calibrated': [0]*8+[25], 'follower_raw_ticks': [2048]*9,
                    'currents_a': [0]*9}} for name in EVENTS}
                episode = {'episode_id': i, 'box_id': 'box_1' if i == 0 else 'box_2',
                    'T_base_tape_reference': reference.tolist(), 'events': events,
                    'outcome': 'success' if i < 2 else 'failed'}
                (root/'episodes'/f'{i}.json').write_text(json.dumps(episode))
            prior = fit_session(root)
            self.assertEqual(prior['successful_episodes'], 2)
            self.assertEqual(prior['excluded_episodes'], 1)
            self.assertFalse(prior['validated_for_execution'])
            np.testing.assert_allclose(prior['grasp_stages_in_tape_reference']['before_close']['T'], local, atol=1e-10)
            self.assertEqual(prior['box_locations_in_base']['box_2']['sample_count'], 1)
            # The frozen visible-surface origin must not be silently renamed a centroid.
            self.assertIn('visible-surface', prior['frame_convention'])

    def test_no_success_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'episodes').mkdir()
            (root/'manifest.json').write_text('{"schema":"blue_tape_grasp_capture_v1"}')
            with self.assertRaisesRegex(ValueError, 'No complete'):
                fit_session(root)


class TimingTests(unittest.TestCase):
    def test_recorded_frames_are_required_and_clock_domains_not_equated(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            timing = {'host_arrival_monotonic_s': 10, 'sensor_timestamp_s': 100,
                      'color_timestamp_s': 100.01, 'timestamp_domain': 'global_time'}
            for role in ('front', 'side'):
                (root/(role+'.npz')).touch()
            observation = {'rgbd': {'save_every': 1, 'files': {r: r+'.npz' for r in ('front', 'side')},
                'views': {r: {'timing': timing} for r in ('front', 'side')}}}
            sample = {'host_read_end_monotonic_s': 10.05}
            wrist = [{'file': 'wrist/frame.npz', 'timing': {**timing, 'timestamp_domain': 'hardware_clock'}}]
            event = event_snapshot(sample, observation, wrist, root/'observation.json', 10.1)
            self.assertFalse(event['cross_camera_clock_comparable'])
            self.assertIsNone(event['three_camera_native_span_s'])
            with self.assertRaisesRegex(ValueError, 'stale'):
                event_snapshot(sample, observation, wrist, root/'observation.json', 11)
            observation['rgbd']['save_every'] = 30
            with self.assertRaisesRegex(ValueError, 'record-rgbd'):
                event_snapshot(sample, observation, wrist, root/'observation.json', 10.1)


class LauncherTests(unittest.TestCase):
    def test_rgbd_flag_routes_to_every_pair_without_starting_cameras(self):
        from real_robot.tracking_session import main
        config = {'perception': {'rig': 'rig.json', 'output_root': '/tmp/unused',
                   'seconds': 60, 'max_skew_s': .035, 'save_every': 30, 'preview': False,
                   'side_depth_range_m': [.07, .80]},
                  'arm': {'output_root': '/tmp/unused'}}
        with tempfile.TemporaryDirectory() as d:
            config['perception']['output_root'] = d
            with patch('sys.argv', ['tracking_session', 'perception', '--record-rgbd']), \
                    patch('real_robot.tracking_session.load_config', return_value=config), \
                    patch('real_robot.multiview_track.main', return_value=None) as run:
                main()
            argv = run.call_args.args[0]
            self.assertEqual(argv[argv.index('--save-every')+1], '1')
            index = argv.index('--side-depth-range')
            self.assertEqual(argv[index+1:index+3], ['0.07', '0.8'])
            self.assertEqual(config['perception']['save_every'], 30)

    def test_rgbd_flag_cannot_start_follow(self):
        from real_robot.tracking_session import main
        with patch('sys.argv', ['tracking_session', 'follow', '--record-rgbd']), \
                patch('real_robot.tracking_session.load_config', return_value={}), \
                self.assertRaises(SystemExit) as error:
            main()
        self.assertEqual(error.exception.code, 2)


class SessionTests(unittest.TestCase):
    def test_stale_startup_never_opens_robot_or_wrist(self):
        from real_robot.grasp_capture import main
        with tempfile.TemporaryDirectory() as d:
            parent = Path(d)
            rig = parent/'rig.json'
            rig.write_text('{}')
            observation = parent/'observation.json'
            observation.write_text(json.dumps({'schema': 'front_side_observation_v1', 'running': False}))
            output = parent/'capture'
            with patch('real_robot.grasp_capture.load_config', return_value={'perception': {'rig': str(rig)}}), \
                    patch('real_robot.grasp_capture.connect_buses') as buses, \
                    patch('real_robot.grasp_capture.WristRecording') as wrist, \
                    self.assertRaises(SystemExit) as error:
                main(['--observation', str(observation), '--heading-deg', '180',
                      '--perception-wait-s', '0', '--output', str(output)])
            self.assertEqual(error.exception.code, 2)
            buses.assert_not_called()
            wrist.assert_not_called()
            self.assertFalse(output.exists())

    def test_two_box_collection_and_offline_fit_without_hardware(self):
        from real_robot.grasp_capture import main
        with tempfile.TemporaryDirectory() as d:
            parent = Path(d)
            rigpath = parent/'rig.json'
            rigpath.write_text('{}')
            config_path = parent/'config.yaml'
            config_path.write_text('test configuration provenance')
            perception = parent/'perception'
            perception.mkdir()
            for role in ('front', 'side'):
                (perception/(role+'.npz')).touch()
            now = time.monotonic()
            timing = {'host_arrival_monotonic_s': now, 'sensor_timestamp_s': 100,
                      'color_timestamp_s': 100.01, 'timestamp_domain': 'global_time'}
            observation = {'schema': 'front_side_observation_v1', 'running': True, 'stale': False,
                'rig_sha256': hashlib.sha256(rigpath.read_bytes()).hexdigest(),
                'output_frame': 'right_base_estimate', 'host_published_monotonic_s': now,
                'timing': {'oldest_host_arrival_monotonic_s': now},
                'fusion': {'stale': False, 'identity_recent': True, 'sources': ['front', 'side'],
                          'output_frame': 'right_base_estimate', 'position_output_m': [.3, 0, .05]},
                'velocity_report': {'speed_mps': 0},
                'rgbd': {'save_every': 1, 'files': {r: r+'.npz' for r in ('front', 'side')},
                         'views': {r: {'timing': timing} for r in ('front', 'side')}}}
            observation_file = perception/'observation.json'
            observation_file.write_text(json.dumps(observation))
            sample = FakeRobot().read()
            sample.update(host_read_end_monotonic_s=now,
                          T_base_tcp_candidate=np.eye(4).tolist())
            control = SimpleNamespace(start=Mock(), close=Mock(), pulse=Mock(),
                snapshot=lambda: dict(sample), stationary=lambda: True,
                lock=threading.Lock(), enabled=False, pending=queue.Queue(), error=None)
            control.pending.put(sample)
            wrist = SimpleNamespace(start=Mock(), close=Mock(), snapshot=lambda: (
                [{'file': 'wrist/frame.npz', 'timing': timing}], np.zeros((480,640,3), np.uint8)))
            buses = {r: Mock(is_connected=True) for r in ('leader', 'follower')}
            config = {'perception': {'rig': str(rigpath)}}
            session = parent/'capture'
            keys = [ord(k) for k in 'npgclarv2npgclarvq']
            with patch('real_robot.grasp_capture.load_config', return_value=config), \
                    patch('real_robot.grasp_capture.WristRecording', return_value=wrist), \
                    patch('real_robot.grasp_capture.connect_buses', return_value=(buses, {})), \
                    patch('real_robot.grasp_capture.RobotIO', return_value=SimpleNamespace(provenance={})), \
                    patch('real_robot.grasp_capture.TeachingControl', return_value=control), \
                    patch('real_robot.grasp_capture.cv2.imshow'), \
                    patch('real_robot.grasp_capture.cv2.destroyAllWindows'), \
                    patch('real_robot.grasp_capture.cv2.waitKey', side_effect=keys):
                code = main(['--config', str(config_path), '--observation', str(observation_file),
                             '--heading-deg', '180', '--output', str(session)])
            self.assertEqual(code, 0)
            control.close.assert_called_once()
            wrist.close.assert_called_once()
            for bus in buses.values():
                bus.disconnect.assert_called_once()
            result = fit_session(session)
            self.assertEqual(result['successful_episodes'], 2)
            self.assertEqual(result['box_locations_in_base']['box_1']['sample_count'], 1)
            self.assertEqual(result['box_locations_in_base']['box_2']['sample_count'], 1)
            self.assertFalse(json.loads((session/'status.json').read_text())['running'])


if __name__ == '__main__':
    unittest.main()
