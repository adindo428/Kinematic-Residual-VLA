"""Bounded RGB-D disk writer; never run compression on the perception loop."""
from pathlib import Path
import io
import queue
import threading
import time
import zipfile

import numpy as np


def save_rgbd(path, bgr, depth, *, compressed=False):
    """Publish complete NPZ archives atomically; compression is opt-in."""
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp.npz')
    if compressed:
        # NumPy's default zlib level 6 cannot sustain three real camera streams.
        # Level 1 preserves exactly the same arrays and standard NPZ reader API.
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for key,array in [('bgr',bgr),('depth_m',depth.astype(np.float32))]:
                buffer=io.BytesIO()
                np.lib.format.write_array(buffer,array,allow_pickle=False)
                archive.writestr(key+'.npy',buffer.getbuffer())
    else:
        np.savez(temporary, bgr=bgr, depth_m=depth.astype(np.float32))
    temporary.replace(path)


class RGBDArchiveWriter:
    def __init__(self, root, capacity=8, *, compressed=False, workers=1):
        if workers<1:
            raise ValueError('At least one archive worker is required')
        self.root = Path(root)
        self.pending = queue.Queue(maxsize=capacity)
        self.stop = threading.Event()
        self.error = None
        self.compressed = compressed
        self.stats_lock=threading.Lock()
        self.submitted=self.completed=self.peak_pending=0
        self.write_seconds=0.
        self.threads=[threading.Thread(target=self._run,name=f'rgbd-archive-{i}',daemon=True) for i in range(workers)]
        self.thread=self.threads[0]
        for thread in self.threads:
            thread.start()

    def stats(self):
        with self.stats_lock:
            return dict(workers=len(self.threads),submitted_pairs=self.submitted,completed_pairs=self.completed,
                        pending_pairs=self.pending.qsize(),peak_pending_pairs=self.peak_pending,
                        mean_pair_write_ms=None if not self.completed else 1000*self.write_seconds/self.completed)

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
            with self.stats_lock:
                self.submitted+=1
                self.peak_pending=max(self.peak_pending,self.pending.qsize())
        except queue.Full as exc:
            raise RuntimeError('RGB-D disk writer backlog full; recording cannot keep up') from exc

    def _run(self):
        try:
            while not self.stop.is_set() or not self.pending.empty():
                try:
                    files, frames = self.pending.get(timeout=.05)
                except queue.Empty:
                    continue
                before=time.monotonic()
                for role, name in files.items():
                    if self.compressed:
                        save_rgbd(self.root/name, frames[role]['bgr'], frames[role]['depth'], compressed=True)
                    else:
                        save_rgbd(self.root/name, frames[role]['bgr'], frames[role]['depth'])
                with self.stats_lock:
                    self.completed+=1
                    self.write_seconds+=time.monotonic()-before
        except BaseException as exc:
            self.error = f'{type(exc).__name__}: {exc}'

    def close(self):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=10)
        if any(thread.is_alive() for thread in self.threads):
            raise RuntimeError('RGB-D disk writer did not finish')
        self.check()
