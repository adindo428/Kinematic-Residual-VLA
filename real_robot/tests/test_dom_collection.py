"""Offline contract tests with real parquet, AV1 and the pinned LeRobot reader."""
import json
import os
import hashlib
import queue
from pathlib import Path
import tempfile
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch, Mock

import numpy as np
from scipy.spatial.transform import Rotation

from real_robot.dom_dataset import (align_episode, command_action, command_raw, dom_euler,
                                    export_sessions, features, jsonl_write, json_write,
                                    TimeIndex, validate_dataset)
from real_robot.dynamic_capture import Episodes
from real_robot.grasp_prior import follower_pose
from real_robot.recording_lease import recording_active, renew, LeaseOwner


def make_session(root, split='train', offset=100):
    root = Path(root)
    root.mkdir()
    calibration = dict(motor_names=[str(i) for i in range(9)], calib_mode=['DEGREE']*8+['LINEAR'],
                       drive_mode=[0,1,1,0,0,0,1,0,0], homing_offset=[-2048]*9,
                       start_pos=[0]*8+[2000], end_pos=[0]*8+[3000])
    manifest = dict(schema='dom_dynamic_capture_v1', mode='execute', task_id='blue_tape', split=split,
                    session_id=root.name, collector='human_leader', heading_deg=180,
                    gripper_open_percent=100, gripper_closed_percent=0,
                    tcp_offset_m=[.16,0,0], native_camera_hz=15, wrist_serial='fixture',
                    rig={'source':'synthetic test calibration', 'base_calibration_validated':True},
                    task={'instruction':'Pick up the rolling blue tape and place it in the box.'},
                    buses={'follower':{'calibration':calibration}})
    json_write(root/'manifest.json', manifest)
    robot = []
    for i in range(76):
        t = offset+i/30
        commanded = [0.,0.,0.,0.,0.,0.,0.,0.,float(i%2)*100]
        raw = command_raw(commanded, calibration)
        _, pose = follower_pose(raw, manifest['tcp_offset_m'])
        robot.append(dict(host_read_start_monotonic_s=t, host_read_end_monotonic_s=t+.002,
                          host_cycle_end_monotonic_s=t+.004, following=True,
                          commanded_calibrated=commanded, T_base_tcp_candidate=pose.tolist()))
    jsonl_write(root/'robot.jsonl', robot)
    obs, wrist = [], []
    for i in range(37):
        t = offset+i/15
        files, views = {}, {}
        for role in ('front','side','wrist'):
            name = f'{role}/{i:04d}.npz'
            path = root/name
            path.parent.mkdir(exist_ok=True)
            bgr = np.zeros((48,64,3),np.uint8)
            bgr[...,i%3] = 180
            np.savez_compressed(path,bgr=bgr,depth_m=np.ones((48,64),np.float32))
            timing = dict(host_arrival_monotonic_s=t, sensor_timestamp_s=t,
                          color_timestamp_s=t, timestamp_domain='fixture', camera_serial=role)
            if role == 'wrist':
                wrist.append(dict(file=name, timing=timing, available_monotonic_s=t+.005))
            else:
                files[role] = name
                views[role] = dict(timing=timing)
        obs.append(dict(host_recorded_monotonic_s=t+.01, host_published_monotonic_s=t+.008,
                        stale=False, rgbd=dict(files=files, views=views),
                        fusion=dict(position_output_m=[.3+i*.001,0,.05], stale=False),
                        velocity_report=dict(velocity_mps=[.015,0,0])))
    jsonl_write(root/'observations.jsonl',obs)
    jsonl_write(root/'wrist/frames.jsonl',wrist)
    episode = dict(episode_id=0,task_id='blue_tape',split=split,start_monotonic_s=offset,
                   end_monotonic_s=offset+2.4,outcome='success',events={})
    json_write(root/'episodes/episode_000000.json',episode)
    return manifest, episode


class CollectionTests(unittest.TestCase):
    def test_recorder_failure_closes_control_before_serial_disconnect(self):
        from real_robot.dynamic_capture import main
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            rig=root/'rig.json';rig.write_text('{}')
            config=root/'config.yaml';config.write_text('test configuration')
            lease=root/'lease.json'
            obs=root/'observation.json'
            obs.write_text(json.dumps(dict(running=True,host_published_monotonic_s=time.monotonic(),
                           recording_lease=str(lease),camera_fps=30,output_frame='right_base_estimate',
                           rig_sha256=hashlib.sha256(rig.read_bytes()).hexdigest())))
            order=[]
            bus=Mock(is_connected=True)
            bus.disconnect.side_effect=lambda:order.append('disconnect')
            leader_bus=Mock(is_connected=True)
            leader_bus.disconnect.side_effect=lambda:order.append('leader_disconnect')
            control=Mock(error=None,pending=queue.Queue())
            control.pulse.side_effect=RuntimeError('injected serial loss')
            control.close.side_effect=lambda:order.append('stop_control')
            wrist=Mock(enabled=threading.Event())
            wrist.snapshot.return_value=(np.zeros((48,64,3),np.uint8),time.monotonic())
            with patch('real_robot.tracking_config.load_config',return_value={'perception':{'rig':str(rig)}}), \
                 patch('real_robot.grasp_capture.connect_buses',return_value=({'leader':leader_bus,'follower':bus},{})), \
                 patch('real_robot.grasp_capture.RobotIO',return_value=Mock(provenance={})), \
                 patch('real_robot.grasp_capture.TeachingControl',return_value=control), \
                 patch('real_robot.dynamic_capture.EpisodeWrist',return_value=wrist), \
                 patch('real_robot.dynamic_capture.cv2.namedWindow'), \
                 patch('real_robot.dynamic_capture.cv2.destroyAllWindows'):
                result=main(['--config',str(config),'--task','orange','--split','train',
                             '--output',str(root/'capture'),'--observation',str(obs),
                             '--recording-lease',str(lease)])
            self.assertEqual(result,1)
            self.assertEqual(order,['stop_control','leader_disconnect','disconnect'])
            self.assertFalse(json.loads(lease.read_text())['active'])
            self.assertIn('injected serial loss',json.loads((root/'capture/summary.json').read_text())['error'])
            bus.write.assert_not_called()
            wrist.close.assert_called_once()

    def test_temporal_causality_and_gap(self):
        stream = TimeIndex([{'t':1},{'t':2}], lambda r:r['t'])
        self.assertEqual(stream.past(1.9)[0]['t'],1)
        with self.assertRaises(ValueError):
            stream.past(.9)
        with self.assertRaises(ValueError):
            stream.past(3,.1)
        self.assertEqual(stream.next(1.1,1)[0]['t'],2)

    def test_lease_expiry_boot_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'lease.json'
            self.assertFalse(recording_active(path,10))
            renew(path,True,10)
            self.assertTrue(recording_active(path,10.1))
            self.assertFalse(recording_active(path,11))
            self.assertFalse(recording_active(path,10.1,boot='other-boot'))
            renew(path,False,10)
            self.assertFalse(recording_active(path,10.1))
            path.write_text('{')
            self.assertFalse(recording_active(path,10))

    def test_only_one_recorder_can_own_a_lease(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'lease.json'
            first=LeaseOwner(path)
            try:
                with self.assertRaises(ValueError):
                    LeaseOwner(path)
            finally:
                first.close()
            second=LeaseOwner(path)
            second.close()

    def test_attempts_survive_failure_and_markers_require_no_holds(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=Episodes(d,'orange','train')
            ledger.begin(10)
            ledger.event('launch',10.1)
            ledger.finish('failed',11,'missed')
            ledger.begin(12)
            ledger.finish('incomplete',13,'quit')
            attempts=[json.loads(x) for x in (Path(d)/'attempts.jsonl').read_text().splitlines()]
            self.assertEqual([x['outcome'] for x in attempts],['failed','incomplete'])
            self.assertEqual(attempts[0]['events']['launch'],10.1)

    def test_fk_command_and_dom_conventions(self):
        with tempfile.TemporaryDirectory() as d:
            manifest,_ = make_session(Path(d)/'raw')
            row=dict(commanded_calibrated=[0.]*8+[100.])
            action=command_action(row,manifest)
            self.assertEqual(action.shape,(7,))
            self.assertEqual(action[-1],1)
            row['commanded_calibrated'][-1]=0
            self.assertEqual(command_action(row,manifest)[-1],-1)
            matrix=np.eye(4)
            matrix[:3,:3]=Rotation.from_euler('xyz',[-.2,.3,-.4]).as_matrix()
            np.testing.assert_allclose(dom_euler(matrix),[2*np.pi-.2,.3,2*np.pi-.4])
            # Compare against the actual pinned bus inverse without connecting hardware.
            from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
            bus=object.__new__(DynamixelMotorsBus)
            bus.calibration=manifest['buses']['follower']['calibration']
            bus.motors={str(i):(i,'fixture') for i in range(9)}
            bus.model_resolution={'fixture':4096}
            target=np.array([12.,-5.,4.,9.,6.,1.,3.,2.,37.])
            expected=bus.revert_calibration(target.copy(),list(bus.motors))
            np.testing.assert_array_equal(command_raw(target,bus.calibration),expected)

    def test_alignment_keeps_regular_time_and_reports_occlusion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'raw'
            manifest,episode=make_session(root)
            rows,paths,quality=align_episode(root,manifest,episode)
            self.assertGreater(len(rows),20)
            self.assertTrue(all(q['camera_ages_s']['front']>=0 for q in quality))
            self.assertTrue(all(not q['orientation_measured'] for q in quality))
            np.testing.assert_allclose(np.diff([r['timestamp'] for r in rows]),.04)
            obs=[json.loads(x) for x in (root/'observations.jsonl').read_text().splitlines()]
            for row in obs[10:]:
                row['fusion']['position_output_m']=None
                row['stale']=True
                row['velocity_report']['velocity_mps']=None
            jsonl_write(root/'observations.jsonl',obs)
            _,_,quality=align_episode(root,manifest,episode)
            self.assertTrue(any(q['velocity_zero_filled'] for q in quality))
            robot=[json.loads(x) for x in (root/'robot.jsonl').read_text().splitlines()]
            jsonl_write(root/'robot.jsonl',[r for r in robot if not 100.9<r['host_read_end_monotonic_s']<101.4])
            with self.assertRaises(ValueError):
                align_episode(root,manifest,episode)

    def test_collection_preflight_checks_headers_without_decoding_old_video_frames(self):
        import av
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);make_session(root/'raw')
            export_sessions([root/'raw'],root/'export')
            real_open=av.open
            class HeaderOnlyContainer:
                def __init__(self,*args,**kwargs):
                    self.container=real_open(*args,**kwargs);self.streams=self.container.streams
                def __enter__(self):return self
                def __exit__(self,*_):self.container.close()
                def decode(self,**_):raise AssertionError('Full frame decode requested')
            with patch('av.open',side_effect=HeaderOnlyContainer):
                report=validate_dataset(root/'export',decode_videos=False)
                self.assertEqual(report['episodes'],1)
                with self.assertRaisesRegex(AssertionError,'Full frame decode requested'):
                    validate_dataset(root/'export',use_video_cache=False)
            self.assertEqual(validate_dataset(root/'export'),report)

    def test_full_export_reads_with_lerobot_and_preserves_splits(self):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.common.datasets.video_utils import decode_video_frames
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            sources=[]
            for i,split in enumerate(('train','validation','test')):
                source=root/split
                make_session(source,split,100+i*10)
                sources.append(source)
            output=root/'dataset'
            report=export_sessions(sources,output)
            self.assertTrue(report['complete'])
            self.assertEqual(validate_dataset(output)['splits'],{'train':1,'validation':1,'test':1})
            meta=LeRobotDatasetMetadata('local/dom-fixture',root=output)
            self.assertEqual(meta.total_episodes,3)
            self.assertEqual(meta.features['action']['shape'],(7,))
            with patch('datasets.config.HF_DATASETS_CACHE',root/'hf-cache'):
                dataset=LeRobotDataset('local/dom-fixture',root=output,episodes=[0],video_backend='pyav')
            sample=dataset[2]
            self.assertEqual(tuple(sample['action'].shape),(7,))
            self.assertEqual(tuple(sample['observation.state'].shape),(6,))
            self.assertEqual(tuple(sample['observation.environment_state'].shape),(9,))
            self.assertEqual(tuple(sample['observation.images.opst_cam'].shape),(3,360,480))
            # DOM's two-timestep input can be decoded at exactly t-2 and t.
            video=output/'videos/chunk-000/observation.images.wrist_cam/episode_000000.mp4'
            pair=decode_video_frames(video,[0.,2/25],tolerance_s=1e-4,backend='pyav')
            self.assertEqual(tuple(pair.shape),(2,3,360,480))
            self.assertEqual(json.loads((output/'meta/splits.json').read_text()),
                             {'train':[0],'validation':[1],'test':[2]})
            self.assertTrue((output/'meta/train_stats.json').is_file())
            self.assertEqual(features()['action']['names'][-1],'gripper')
            # Import the actual DOM training reader in an isolated process. Namespace
            # aliases only bridge this workstation's older LeRobot package layout.
            code = '''
import importlib,sys
from pathlib import Path
import lerobot
sys.path.insert(0,str(Path('third_party/DynamicVLA').resolve()))
for alias,target in [('constants','lerobot.common.constants'),('datasets','lerobot.common.datasets')]:
    module=importlib.import_module(target)
    sys.modules['lerobot.'+alias]=module
    setattr(lerobot,alias,module)
for name in ['compute_stats','lerobot_dataset','utils']:
    sys.modules['lerobot.datasets.'+name]=importlib.import_module('lerobot.common.datasets.'+name)
from utils.datasets import LeRobotDataset
from real_robot.dataset_quality import process
processed=Path(sys.argv[1]).with_name('processed')
process(Path(sys.argv[1]),processed,repair=True)
try:
    LeRobotDataset('local/dom-fixture',root=Path(sys.argv[1]),episodes=[0],
                   required_features=['observation.environment_state'])
except ValueError as exc:
    assert 'quality sidecars' in str(exc)
else:
    raise AssertionError('Unprocessed tracking sidecars must not be ignored')
reader=LeRobotDataset('local/dom-fixture',root=processed,episodes=[0],split='train',
    pin_memory=False,delta_action=True,
    euler_delta_convention='wrapped_xyz_v1',
    required_features=['action','observation.state','observation.environment_state',
                       'observation.images.opst_cam','observation.images.wrist_cam'],
    delta_timestamps={'observation':[-2,0],'action':list(range(20))})
sample=reader[5]
assert tuple(sample['action'].shape)==(20,7)
assert tuple(sample['observation.state'].shape)==(2,6)
assert tuple(sample['observation.environment_state'].shape)==(2,9)
assert tuple(sample['dynamic_object.velocity_valid'].shape)==(2,)
assert tuple(sample['observation.images.opst_cam'].shape)==(2,3,360,480)
assert tuple(sample['observation.images.wrist_cam'].shape)==(2,3,360,480)
tail=reader[len(reader)-1]
assert tail['action_is_pad'].sum().item()==19
assert isinstance(sample['task'],str)
from utils.dom_data import DOMData
reader.quality_indices=reader.quality_indices[::2]
wrapped=DOMData(reader, horizon=20)
assert sum(wrapped.lengths)==len(reader)
single=wrapped[5]
assert int(single['frame_index'])==10
batched=wrapped.__getitems__([3,5])
import torch
torch.testing.assert_close(single['action'],batched[1]['action'])
torch.testing.assert_close(single['observation.images.wrist_cam'],batched[1]['observation.images.wrist_cam'])
print('Actual DOM training reader: two views, t-2/t, 20-step delta actions, tail padding OK')
'''
            env=dict(os.environ,HF_HUB_OFFLINE='1',HF_DATASETS_CACHE=str(root/'dom-hf-cache'))
            # This workstation's Conda FFmpeg 4.3 lacks AV1 decoders. Use its
            # installed system FFmpeg 4.4 libraries in this isolated check only;
            # no packages or the working robot environment are changed.
            system_libs=[Path('/usr/lib/x86_64-linux-gnu')/name for name in
                         ('libavutil.so.56','libswresample.so.3','libavcodec.so.58',
                          'libavformat.so.58','libswscale.so.5','libavfilter.so.7')]
            if all(p.exists() for p in system_libs):
                env['LD_PRELOAD']=':'.join(map(str,system_libs))
            result=subprocess.run([sys.executable,'-c',code,str(output)],
                                  cwd=Path(__file__).resolve().parents[2],env=env,
                                  capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            with self.assertRaises(ValueError):
                export_sessions(sources,output)


if __name__ == '__main__':
    unittest.main()
