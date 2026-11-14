"""Exercise SDK hangs in disposable processes without opening cameras or motors."""
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from real_robot.wrist_capture import EpisodeWrist


def fake_camera(root,serial,fps,restart,enabled,stop,pixels,sequence,timing,phase,error):
    while not stop.value:
        sequence.value+=1
        np.frombuffer(pixels,dtype=np.uint8)[:]=restart+1
        timing[0]=timing[1]=time.monotonic()
        sequence.value+=1
        phase.value=2
        if restart==0:
            # A native call that ignores graceful shutdown.
            time.sleep(60)
        time.sleep(.03)


def failing_camera(root,serial,fps,restart,enabled,stop,pixels,sequence,timing,phase,error):
    error.value=b'Injected camera disconnected'
    phase.value=7


class WristProcessTests(unittest.TestCase):
    def wait_until(self,predicate,timeout=10):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if predicate():return
            time.sleep(.03)
        self.fail('Camera supervisor condition timed out')

    def test_hung_sdk_is_killed_and_fresh_frames_return(self):
        with tempfile.TemporaryDirectory() as d:
            wrist=EpisodeWrist(Path(d)/'wrist','fake',stall_s=.2,startup_s=5,
                               worker_target=fake_camera)
            wrist.thread.start()
            try:
                self.wait_until(lambda:wrist.snapshot()[0] is not None and wrist.snapshot()[0][0,0,0]==2)
                self.assertEqual(wrist.restarts,1)
                self.assertLess(time.monotonic()-wrist.snapshot()[1],.2)
                events=[json.loads(row) for row in (wrist.root/'camera_events.jsonl').read_text().splitlines()]
                self.assertTrue(any(row['event']=='restarting' and 'aligning_depth' in row['detail'] for row in events))
            finally:wrist.close()
            self.assertFalse(wrist.process.is_alive())
            self.assertFalse(wrist.thread.is_alive())

    def test_close_terminates_native_hang_with_bounded_wait(self):
        with tempfile.TemporaryDirectory() as d:
            wrist=EpisodeWrist(Path(d)/'wrist','fake',stall_s=30,worker_target=fake_camera)
            wrist.thread.start()
            try:self.wait_until(lambda:wrist.snapshot()[0] is not None)
            finally:
                started=time.monotonic();wrist.close()
            self.assertLess(time.monotonic()-started,4)
            self.assertFalse(wrist.process.is_alive())

    def test_persistent_disconnect_reports_root_error_and_stops(self):
        with tempfile.TemporaryDirectory() as d:
            wrist=EpisodeWrist(Path(d)/'wrist','fake',max_restarts=1,worker_target=failing_camera)
            wrist.thread.start()
            try:
                self.wait_until(lambda:wrist.error is not None)
                with self.assertRaisesRegex(RuntimeError,'Injected camera disconnected'):wrist.snapshot()
                self.assertEqual(wrist.restarts,1)
            finally:wrist.close()
            self.assertFalse(wrist.process.is_alive())


if __name__=='__main__':unittest.main()
