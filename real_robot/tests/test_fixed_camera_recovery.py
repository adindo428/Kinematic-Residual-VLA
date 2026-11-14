import time
import unittest
from unittest.mock import patch

import numpy as np
from real_robot.multiview import CameraPair


class FixedCameraRecoveryTests(unittest.TestCase):
    def test_timeout_reopens_same_camera_and_stays_nonfatal(self):
        pair=CameraPair(reconnect_attempts=1)
        calls=[]
        class Camera:
            def __init__(self,serial,fps):calls.append(serial);self.number=len(calls)
            def read(self):
                if self.number==1:raise RuntimeError('No synchronized RGB/depth pair')
                pair.stop.set()
                return np.zeros((2,2,3),np.uint8),np.ones((2,2)),dict(sensor_timestamp_s=10.,timestamp_domain='global_time',host_arrival_monotonic_s=time.monotonic()),None
            def close(self):pass
        with patch('real_robot.camera.RGBDCamera',Camera):pair._read('front')
        self.assertEqual(len(calls),2)
        self.assertEqual(pair.restarts['front'],1)
        self.assertEqual(pair.errors,{})
        self.assertEqual(pair.states['front'],'running')
        self.assertTrue(any(e['event']=='reconnecting' for e in pair.take_events()))

    def test_front_only_never_opens_side_or_claims_a_side_frame(self):
        pair=CameraPair(front_only=True)
        opened=[]
        class Camera:
            def __init__(self,serial,fps):opened.append(serial)
            def read(self):
                pair.stop.set()
                return np.zeros((2,2,3),np.uint8),np.ones((2,2)),dict(sensor_timestamp_s=10.,color_timestamp_s=10.001,
                    timestamp_domain='global_time',host_arrival_monotonic_s=time.monotonic()),None
            def close(self):pass
        with patch('real_robot.camera.RGBDCamera',Camera):pair._read('front')
        front,side,timing=pair.poll_latest()
        self.assertIsNone(side)
        self.assertEqual(opened,[pair.serials['front']])
        self.assertEqual(set(pair.views()),{'front'})
        self.assertTrue(pair.health()['ready'])
        from real_robot.multiview_track import live_observation
        self.assertFalse(live_observation({},pair.views(),time.monotonic(),time.monotonic(),required_views=('front',))['stale'])
        self.assertTrue(live_observation({},pair.views(),time.monotonic(),time.monotonic())['stale'])

    def test_persistent_failure_stops_after_bounded_retries(self):
        pair=CameraPair(reconnect_attempts=1)
        class Camera:
            def __init__(self,*args,**kwargs):pass
            def read(self):raise RuntimeError('persistent synchronization failure')
            def close(self):pass
        with patch('real_robot.camera.RGBDCamera',Camera):pair._read('side')
        self.assertEqual(pair.restarts['side'],1)
        self.assertIn('side',pair.errors)
        self.assertFalse(pair.health()['ready'])
        with self.assertRaises(RuntimeError):pair.poll()


if __name__=='__main__':unittest.main()
