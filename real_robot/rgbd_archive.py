"""Bounded RGB-D disk writer; never run compression on the perception loop."""
from pathlib import Path
import queue
import threading

import numpy as np


def save_rgbd(path, bgr, depth):
    """Publish complete, uncompressed NPZ archives using atomic replacement."""
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp.npz')
    np.savez(temporary, bgr=bgr, depth_m=depth.astype(np.float32))
    temporary.replace(path)


class RGBDArchiveWriter:
    def __init__(self, root, capacity=8):
        self.root = Path(root)
        self.pending = queue.Queue(maxsize=capacity)
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name='rgbd-archive', daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError('RGB-D recording failed: '+self.error)

    def submit(self, files, frames):
        self.check()
        if self.stop.is_set():
            raise RuntimeError('RGB-D writer is closed')
        try:
            # Camera adapters return independent arrays; retain references until saved.
            self.pending.put_nowait((dict(files), dict(frames)))
        except queue.Full as exc:
            raise RuntimeError('RGB-D disk writer backlog full; recording cannot keep up') from exc

    def _run(self):
        try:
            while not self.stop.is_set() or not self.pending.empty():
                try:
                    files, frames = self.pending.get(timeout=.05)
                except queue.Empty:
                    continue
                for role, name in files.items():
                    save_rgbd(self.root/name, frames[role]['bgr'], frames[role]['depth'])
        except BaseException as exc:
            self.error = f'{type(exc).__name__}: {exc}'

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError('RGB-D disk writer did not finish')
        self.check()
