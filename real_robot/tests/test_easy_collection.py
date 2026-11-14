"""One-command collection regressions; all robot devices are simulated."""
from collections import deque
import hashlib
import io
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock,patch

import cv2
import numpy as np

from real_robot.collection_control import ClutchedIO,CollectionControl
from real_robot.dynamic_capture import main as capture,FixedRecordingBuffer
from real_robot.multiview import CameraPair,FramePairer
from real_robot.object_profiles import load_prompt,save_prompt


class FakeIO:
    def __init__(self,torque=1):
        self.leader=np.full(9,90.); self.actual=np.zeros(9)
        self.torque=np.full(9,torque); self.events=[]
        self.follower=Mock()
        self.follower.write.side_effect=self.write

    def write(self,key,values):
        assert key=='Torque_Enable'
        self.events.append((key,np.array(values)))
        self.torque=np.array(values)

    def health(self):
        return dict(torque=self.torque.copy(),hardware_errors=np.zeros(9))

    def goal(self,target):
        self.events.append(('Goal_Position',np.array(target)))
        self.actual=np.array(target)

    def read(self):
        return dict(leader_raw_ticks=[2048]*9,follower_raw_ticks=[2048]*9,
                    leader_calibrated=self.leader.tolist(),follower_calibrated=self.actual.tolist(),
                    temperatures_c=[25]*9)


def wait_until(predicate):
    deadline=time.monotonic()+1
    while not predicate():
        if time.monotonic()>deadline: raise AssertionError('Timed out')
        time.sleep(.005)


class EasyCollectionTests(unittest.TestCase):
    def test_async_files_keep_original_observation_times_and_order(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'perception';source.mkdir()
            output=root/'recording';output.mkdir()
            stream=io.StringIO();buffer=FixedRecordingBuffer(source,output,stream)
            def observation(i):
                return dict(host_published_monotonic_s=float(i),
                            rgbd={'files':{'front':f'{i}_front.npz','side':f'{i}_side.npz'}})
            buffer.receive(observation(1),100.)
            buffer.receive(observation(2),101.)
            self.assertEqual(buffer.drain(),0)
            for name in ('2_front.npz','2_side.npz'):
                np.savez(source/name,bgr=np.zeros((2,2,3),np.uint8),depth_m=np.ones((2,2),np.float32))
            self.assertEqual(buffer.drain(),0)
            for name in ('1_front.npz','1_side.npz'):
                np.savez(source/name,bgr=np.zeros((2,2,3),np.uint8),depth_m=np.ones((2,2),np.float32))
            buffer.finish()
            rows=[json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertEqual([row['host_recorded_monotonic_s'] for row in rows],[100.,101.])
            self.assertEqual([row['host_published_monotonic_s'] for row in rows],[1.,2.])
            self.assertTrue(all((output/path).exists() for row in rows for path in row['rgbd']['files'].values()))

    def control(self,torque=1,execute=True):
        io=FakeIO(torque)
        control=CollectionControl(io,execute,[.16,0,0])
        control.start();self.addCleanup(control.close)
        wait_until(lambda:control.latest is not None)
        return io,control

    def test_large_leader_mismatch_clutches_without_enable_jump(self):
        io,control=self.control()
        self.assertTrue(control.toggle())
        wait_until(lambda:len(io.events)>0)
        np.testing.assert_allclose(io.events[0][1],np.zeros(9))
        io.leader[:8]+=12
        wait_until(lambda:io.actual[0]>0)
        control.close()
        goals=[value for key,value in io.events if key=='Goal_Position']
        previous=np.zeros(9)
        for value in goals:
            self.assertLessEqual(float(np.max(np.abs(value-previous))),5.)
            previous=value

    def test_torque_off_seeds_current_pose_before_explicit_enable(self):
        io,control=self.control(torque=0)
        self.assertTrue(control.toggle())
        self.assertEqual([key for key,_ in io.events[:2]],['Goal_Position','Torque_Enable'])
        np.testing.assert_allclose(io.events[0][1],np.zeros(9))
        self.assertTrue(control.torque_enabled_here)

    def test_preview_and_mixed_torque_do_not_energize(self):
        io,control=self.control(execute=False)
        with self.assertRaisesRegex(ValueError,'Preview'):
            control.toggle()
        self.assertEqual(io.events,[])
        io,control=self.control(torque=0)
        io.torque[0]=1
        with self.assertRaisesRegex(ValueError,'Mixed'):
            control.toggle()
        self.assertEqual(io.events,[])

    def test_gripper_clutch_is_continuous_and_reaches_both_endpoints(self):
        io=ClutchedIO(FakeIO())
        io.anchor([90]*8+[50],[20]*8+[25])
        np.testing.assert_allclose(io.map_leader([90]*8+[50]),[20]*8+[25])
        self.assertEqual(io.map_leader([90]*8+[100])[-1],100)
        self.assertEqual(io.map_leader([90]*8+[0])[-1],0)

    def test_latest_pair_does_not_replay_backlog(self):
        camera=CameraPair.__new__(CameraPair)
        camera.lock=threading.Lock(); camera.errors={}; camera.pairer=FramePairer()
        def frame(i):
            return dict(timing=dict(sensor_timestamp_s=float(i),color_timestamp_s=float(i),
                                    timestamp_domain='global_time',host_arrival_monotonic_s=time.monotonic()))
        camera.pending={role:deque([frame(i) for i in range(1,4)]) for role in ('front','side')}
        pair=camera.poll_latest()
        self.assertEqual(pair[2]['timestamp_s'],3.)
        self.assertEqual(camera.pairer.dropped,{'front':2,'side':2})

    def test_saved_profile_preserves_mask_and_exact_reference_and_serial(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'source'; folder=source/'prompt_front';folder.mkdir(parents=True)
            image=np.full((20,30,3),[101,3,7],np.uint8)
            mask=np.zeros((20,30),np.uint8); mask[3:18,5:25]=255
            cv2.imwrite(str(folder/'reference.png'),image); cv2.imwrite(str(folder/'mask.png'),mask)
            (source/'front-prompt.json').write_text(json.dumps(dict(timing={'camera_serial':'front'},box_xywh=[5,3,20,15])))
            save_prompt(source,root/'profile','front')
            actual,actual_mask,_=load_prompt(root/'profile','front','front')
            np.testing.assert_array_equal(actual,image);np.testing.assert_array_equal(actual_mask,mask)
            with self.assertRaisesRegex(ValueError,'another camera'):
                load_prompt(root/'profile','front','wrong')

    def test_attempt_can_begin_with_stale_or_absent_object(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); rig=root/'rig.json';rig.write_text('{}')
            config=root/'config.yaml';config.write_text('test')
            lease=root/'lease.json';obs=root/'observation.json'
            obs.write_text(json.dumps(dict(running=True,stale=True,track=None,
                           host_published_monotonic_s=time.monotonic(),recording_lease=str(lease),
                           camera_fps=30,output_frame='right_base_estimate',
                           rig_sha256=hashlib.sha256(rig.read_bytes()).hexdigest())))
            bus=Mock(is_connected=True)
            control=Mock(enabled=False,error=None,pending=queue.Queue())
            control.toggle.side_effect=lambda:setattr(control,'enabled',True)
            control.torque_enabled_here=False
            wrist=Mock(enabled=threading.Event())
            wrist.snapshot.return_value=(np.zeros((48,64,3),np.uint8),time.monotonic())
            with patch('real_robot.tracking_config.load_config',return_value={'perception':{'rig':str(rig)}}), \
                 patch('real_robot.grasp_capture.connect_buses',return_value=({'leader':bus,'follower':bus},{})), \
                 patch('real_robot.grasp_capture.RobotIO',return_value=Mock(provenance={})), \
                 patch('real_robot.grasp_capture.TeachingControl',return_value=control), \
                 patch('real_robot.collection_control.CollectionControl',return_value=control), \
                 patch('real_robot.dynamic_capture.EpisodeWrist',return_value=wrist), \
                 patch('real_robot.dynamic_capture.cv2.namedWindow'), \
                 patch('real_robot.dynamic_capture.cv2.imshow'), \
                 patch('real_robot.dynamic_capture.cv2.destroyAllWindows'), \
                 patch('real_robot.dynamic_capture.cv2.waitKey',side_effect=[ord('n'),ord('q')]):
                result=capture(['--config',str(config),'--task','blue_tape','--split','train',
                                '--output',str(root/'capture'),'--observation',str(obs),
                                '--recording-lease',str(lease),'--clutched','--execute'])
            self.assertEqual(result,0)
            episode=json.loads((root/'capture/episodes/episode_000000.json').read_text())
            self.assertEqual(episode['outcome'],'incomplete')
            self.assertIsNotNone(episode['start_monotonic_s'])
            control.toggle.assert_called_once()

    def test_wrist_gap_preserves_success_and_requires_new_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); rig=root/'rig.json';rig.write_text('{}')
            config=root/'config.yaml';config.write_text('test')
            lease=root/'lease.json';obs=root/'observation.json'
            obs.write_text(json.dumps(dict(running=True,stale=True,track=None,
                           host_published_monotonic_s=time.monotonic(),recording_lease=str(lease),
                           camera_fps=30,output_frame='right_base_estimate',
                           rig_sha256=hashlib.sha256(rig.read_bytes()).hexdigest())))
            bus=Mock(is_connected=True)
            control=Mock(enabled=False,error=None,pending=queue.Queue())
            control.toggle.side_effect=lambda:setattr(control,'enabled',not control.enabled)
            control.torque_enabled_here=False
            wrist=Mock(enabled=threading.Event())
            ages=iter([0,0,0,0,2,0,0,0])
            wrist.snapshot.side_effect=lambda:(np.zeros((48,64,3),np.uint8),time.monotonic()-next(ages))
            with patch('real_robot.tracking_config.load_config',return_value={'perception':{'rig':str(rig)}}), \
                 patch('real_robot.grasp_capture.connect_buses',return_value=({'leader':bus,'follower':bus},{})), \
                 patch('real_robot.grasp_capture.RobotIO',return_value=Mock(provenance={})), \
                 patch('real_robot.grasp_capture.TeachingControl',return_value=control), \
                 patch('real_robot.collection_control.CollectionControl',return_value=control), \
                 patch('real_robot.dynamic_capture.EpisodeWrist',return_value=wrist), \
                 patch('real_robot.dynamic_capture.cv2.namedWindow'), \
                 patch('real_robot.dynamic_capture.cv2.imshow'), \
                 patch('real_robot.dynamic_capture.cv2.destroyAllWindows'), \
                 patch('real_robot.dynamic_capture.cv2.waitKey',side_effect=[ord('n'),ord('v'),ord('n'),255,ord('n'),ord('x'),ord('q')]):
                result=capture(['--config',str(config),'--task','blue_tape','--split','train',
                                '--output',str(root/'capture'),'--observation',str(obs),
                                '--recording-lease',str(lease),'--clutched','--execute'])
            self.assertEqual(result,0)
            episodes=[json.loads(path.read_text()) for path in sorted((root/'capture/episodes').glob('*.json'))]
            self.assertEqual([row['outcome'] for row in episodes],['success','incomplete','failed'])
            self.assertEqual(episodes[1]['reason'],'wrist_camera_gap')
            self.assertEqual(control.toggle.call_count,3)
            self.assertTrue(control.enabled)


if __name__=='__main__':
    unittest.main()
