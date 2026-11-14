"""Pilot controls tested without camera or motor access."""
import os
import termios
import hashlib
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from real_robot.dynamic_capture import main as capture
from real_robot.collection_preview import TerminalKeys
from real_robot.collection_control import CollectionControl
from real_robot.tests.test_easy_collection import FakeIO, wait_until


class PilotTests(unittest.TestCase):
    def capture_keys(self, keys, max_successes=10, viewer='opencv', start_delay=0, fixed_ready=None,task='striped_ball',limits=10):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rig=root/'rig.json';rig.write_text('{}')
            config=root/'config.yaml';config.write_text('test')
            lease=root/'lease.json';obs=root/'observation.json'
            obs.write_text(json.dumps(dict(running=True,stale=True,track=None,
                host_published_monotonic_s=time.monotonic(),recording_lease=str(lease),
                camera_fps=30,output_frame='right_base_estimate',
                rig_sha256=hashlib.sha256(rig.read_bytes()).hexdigest())))
            bus=Mock(is_connected=True)
            control=Mock(enabled=False,error=None,pending=queue.Queue(),torque_enabled_here=False)
            control.toggle.side_effect=lambda:setattr(control,'enabled',not control.enabled)
            audio=Mock()
            preview=Mock()
            sequence=iter(keys)
            readiness=iter(fixed_ready) if fixed_ready is not None else None
            def next_key(*_):
                if readiness is not None:
                    payload=json.loads(obs.read_text());payload['fixed_cameras_ready']=next(readiness)
                    obs.write_text(json.dumps(payload))
                value=next(sequence)
                if isinstance(value,Exception):raise value
                return value
            preview.poll.side_effect=next_key
            wrist=Mock(enabled=threading.Event())
            wrist.snapshot.side_effect=lambda:(np.zeros((48,64,3),np.uint8),time.monotonic())
            with patch('real_robot.tracking_config.load_config',return_value={'perception':{'rig':str(rig)}}), \
                 patch('real_robot.grasp_capture.connect_buses',return_value=({'leader':bus,'follower':bus},{})), \
                 patch('real_robot.grasp_capture.RobotIO',return_value=Mock(provenance={})), \
                 patch('real_robot.collection_control.CollectionControl',return_value=control) as factory, \
                 patch('real_robot.collection_preview.RerunPreview',return_value=preview), \
                 patch('real_robot.dynamic_capture.AudioCues',return_value=audio), \
                 patch('real_robot.dynamic_capture.EpisodeWrist',return_value=wrist), \
                 patch('real_robot.dynamic_capture.cv2.namedWindow'), \
                 patch('real_robot.dynamic_capture.cv2.imshow'), \
                 patch('real_robot.dynamic_capture.cv2.destroyAllWindows'), \
                 patch('real_robot.dynamic_capture.cv2.waitKey',side_effect=next_key):
                result=capture(['--config',str(config),'--task',task,'--split','train',
                    '--output',str(root/'capture'),'--observation',str(obs),
                    '--recording-lease',str(lease),'--clutched','--execute','--default-success',
                    '--start-delay',str(start_delay),'--robot-max-relative-target',json.dumps(limits),'--viewer',viewer,'--max-successes',str(max_successes)])
            self.assertEqual(factory.call_args.kwargs['max_relative_target'],limits)
            manifest=json.loads((root/'capture/manifest.json').read_text())
            self.assertEqual(manifest['max_relative_target'],limits)
            self.manifest=manifest
            episodes=[json.loads(p.read_text()) for p in sorted((root/'capture/episodes').glob('*.json'))]
            self.audio_calls=[c.args[0] for c in audio.play.call_args_list]
            self.preview_states=[call.args[0] for call in preview.set_status.call_args_list]
            return result,episodes

    def test_marker_session_uses_marker_labels_and_existing_space_enter_controls(self):
        code,episodes=self.capture_keys([32,13,ord('q')],task='marker',viewer='rerun')
        self.assertEqual(code,0)
        self.assertEqual([e['task_id'] for e in episodes],['marker'])
        self.assertEqual([e['outcome'] for e in episodes],['success'])
        self.assertEqual(self.manifest['task']['instruction'],'Pick up the moving green marker and place it in the box.')
        self.assertEqual(self.manifest['object_instance'],'marker')

    def test_pick_only_bottle_preserves_nine_limits_and_saves_without_release(self):
        limits=[20]*8+[35]
        code,episodes=self.capture_keys([32,13,ord('q')],task='blue_bottle',viewer='rerun',limits=limits)
        self.assertEqual(code,0)
        self.assertEqual([e['outcome'] for e in episodes],['success'])
        self.assertEqual(self.manifest['task']['task_type'],'pick')
        self.assertEqual(self.manifest['task']['instruction'],'Pick up the rolling blue bottle.')
        self.assertIsNone(self.manifest['box_id'])

    def test_each_motor_keeps_its_requested_bound_during_following(self):
        limits=np.array([20]*8+[35])
        io=FakeIO();control=CollectionControl(io,True,[.16,0,0],max_relative_target=limits.tolist())
        control.start()
        try:
            wait_until(lambda:control.latest is not None)
            control.toggle()
            io.leader[:8]+=100;io.leader[-1]=100
            wait_until(lambda:np.all(io.actual>=100))
        finally:control.close()
        goals=[v for k,v in io.events if k=='Goal_Position']
        steps=np.abs(np.diff(np.vstack([np.zeros(9),*goals]),axis=0))
        self.assertTrue(np.all(steps<=limits+1e-6))
        self.assertAlmostEqual(steps[:,:8].max(),20)
        self.assertAlmostEqual(steps[:,-1].max(),35)

    def test_terminal_keys_without_global_listener_and_restore(self):
        master,slave=os.openpty()
        try:
            with os.fdopen(os.dup(slave),'r') as stream:
                original=termios.tcgetattr(slave)
                keys=TerminalKeys(stream)
                try:
                    self.assertEqual(keys.poll(),255)
                    for data,expected in [(b'\n',13),(b' ',32),(b'x',ord('x')),(b'L',ord('l')),(b'r',ord('r')),(b'q',ord('q'))]:
                        os.write(master,data)
                        wait_until(lambda:bool(__import__('select').select([slave],[],[],0)[0]))
                        self.assertEqual(keys.poll(),expected)
                finally:keys.close()
                self.assertEqual(termios.tcgetattr(slave),original)
        finally:
            os.close(master);os.close(slave)

    def test_space_starts_enter_stops_and_repeated_keys_do_not_create_episodes(self):
        code,episodes=self.capture_keys([13,32,32,13,13,32,ord('x'),32,13,ord('q')])
        self.assertEqual(code,0)
        self.assertEqual([e['outcome'] for e in episodes],['success','failed','success'])

    def test_ten_successes_stop_without_eleventh_episode(self):
        code,episodes=self.capture_keys([32,13]*10)
        self.assertEqual(code,0)
        self.assertEqual(len(episodes),10)
        self.assertTrue(all(e['outcome']=='success' for e in episodes))

    def test_fault_is_incomplete_even_with_default_success(self):
        code,episodes=self.capture_keys([32,RuntimeError('injected camera fault')])
        self.assertEqual(code,1)
        self.assertEqual(episodes[0]['outcome'],'incomplete')

    def test_rerun_save_ack_persists_during_next_episode_and_final_save(self):
        code,_=self.capture_keys([32,13,255,32,13,ord('q')],viewer='rerun')
        self.assertEqual(code,0)
        self.assertTrue(any(s['last_saved']=='EPISODE 1 SAVED - SUCCESS' and s['episode']==2
                            and s['successes']==1 for s in self.preview_states))
        self.assertTrue(any(s['last_saved']=='EPISODE 1 SAVED - SUCCESS' and s['state']=='READY'
                            and s['episode'] is None for s in self.preview_states))
        self.assertEqual(self.preview_states[-1]['last_saved'],'EPISODE 2 SAVED - SUCCESS')
        self.assertEqual(self.preview_states[-1]['state'],'FINISHED')

    def test_quit_does_not_label_an_unstopped_episode_successful(self):
        code,episodes=self.capture_keys([32,ord('q')])
        self.assertEqual(code,0)
        self.assertEqual(episodes[0]['outcome'],'incomplete')

    def test_countdown_keeps_recording_closed_then_cues_start_and_end(self):
        code,episodes=self.capture_keys([32]+[255]*7+[13,ord('q')],viewer='rerun',start_delay=.04)
        self.assertEqual(code,0)
        self.assertEqual(len(episodes),1)
        countdown=[s for s in self.preview_states if s['state'].startswith('START IN')]
        self.assertTrue(countdown)
        self.assertTrue(all(s['episode'] is None for s in countdown))
        self.assertEqual(self.audio_calls,['start','end'])

    def test_cancel_countdown_does_not_create_episode_or_start_sound(self):
        code,episodes=self.capture_keys([32,13,ord('q')],viewer='rerun',start_delay=1)
        self.assertEqual(code,0)
        self.assertEqual(episodes,[])
        self.assertEqual(self.audio_calls,[])

    def test_fixed_camera_gap_interrupts_episode_and_does_not_auto_resume(self):
        # Readiness updates become visible on the following recorder iteration.
        code,episodes=self.capture_keys([32,255,32,255,32,13,ord('q')],viewer='rerun',
                                       fixed_ready=[True,False,False,True,True,True,True])
        self.assertEqual(code,0)
        self.assertEqual([e['outcome'] for e in episodes],['incomplete','success'])
        self.assertEqual(episodes[0]['reason'],'fixed_camera_gap')
        self.assertTrue(any(s['state']=='CAMERA RECONNECTING' for s in self.preview_states))
        self.assertIn('failed',self.audio_calls)
        self.assertTrue(any(s['state']=='PAUSED' and s['last_outcome']=='incomplete'
                            and s['episode'] is None for s in self.preview_states))

    def test_requested_ten_unit_command_bound(self):
        io=FakeIO()
        control=CollectionControl(io,True,[.16,0,0],max_relative_target=10)
        control.start()
        try:
            wait_until(lambda:control.latest is not None)
            control.toggle()
            io.leader[:8]+=30
            wait_until(lambda:io.actual[0]>=30)
        finally:control.close()
        goals=[value for key,value in io.events if key=='Goal_Position']
        steps=np.diff(np.vstack([np.zeros(9),*goals]),axis=0)
        self.assertLessEqual(np.abs(steps).max(),10)
        self.assertAlmostEqual(np.abs(steps).max(),10)


if __name__=='__main__':unittest.main()
