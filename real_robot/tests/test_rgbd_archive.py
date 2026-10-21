"""Disk recording must not stall live perception or hide write failures."""
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from real_robot.rgbd_archive import RGBDArchiveWriter, save_rgbd


class ArchiveTests(unittest.TestCase):
    def test_complete_compatible_npz_is_published_atomically(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'frame.npz'
            bgr = np.ones((3,4,3), np.uint8)
            depth = np.full((3,4), .25)
            save_rgbd(path, bgr, depth)
            with np.load(path) as data:
                np.testing.assert_array_equal(data['bgr'], bgr)
                np.testing.assert_allclose(data['depth_m'], depth)
                self.assertEqual(data['depth_m'].dtype, np.float32)
            with zipfile.ZipFile(path) as archive:
                self.assertTrue(all(i.compress_type == zipfile.ZIP_STORED for i in archive.infolist()))
            self.assertEqual(list(Path(d).glob('*.tmp.npz')), [])

    def test_background_writer_flushes_every_submitted_pair(self):
        with tempfile.TemporaryDirectory() as d:
            writer = RGBDArchiveWriter(d)
            frame = {'bgr': np.zeros((3,4,3), np.uint8), 'depth': np.ones((3,4))}
            writer.submit({'front': 'front.npz', 'side': 'side.npz'}, {'front': frame, 'side': frame})
            writer.close()
            self.assertEqual(sorted(p.name for p in Path(d).glob('*.npz')), ['front.npz', 'side.npz'])

    def test_slow_disk_does_not_block_submit_and_overflow_is_explicit(self):
        entered, release = threading.Event(), threading.Event()
        def slow_save(*args):
            entered.set()
            release.wait(2)
        with tempfile.TemporaryDirectory() as d, patch('real_robot.rgbd_archive.save_rgbd', side_effect=slow_save):
            writer = RGBDArchiveWriter(d, capacity=1)
            try:
                frames = {'front': {'bgr': None, 'depth': None}}
                writer.submit({'front': 'a.npz'}, frames)
                self.assertTrue(entered.wait(1))
                writer.submit({'front': 'b.npz'}, frames)
                with self.assertRaisesRegex(RuntimeError, 'backlog full'):
                    writer.submit({'front': 'c.npz'}, frames)
            finally:
                release.set()
                writer.close()

    def test_disk_failure_reaches_producer_and_shutdown(self):
        with tempfile.TemporaryDirectory() as d, patch('real_robot.rgbd_archive.save_rgbd', side_effect=OSError('disk full')):
            writer = RGBDArchiveWriter(d)
            writer.submit({'front': 'a.npz'}, {'front': {'bgr': None, 'depth': None}})
            deadline = time.monotonic()+1
            while writer.error is None and time.monotonic() < deadline:
                time.sleep(.005)
            with self.assertRaisesRegex(RuntimeError, 'disk full'):
                writer.check()
            with self.assertRaisesRegex(RuntimeError, 'disk full'):
                writer.close()


if __name__ == '__main__':
    unittest.main()
