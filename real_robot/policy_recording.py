"""Bounded lossless inference recording, outside the GPU/control loops."""
import io
from pathlib import Path
import queue
import threading
import zipfile
import numpy as np


class InferenceArchive:
    def __init__(self,root,capacity=4):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.queue=queue.Queue(maxsize=capacity);self.error=None;self.stop=threading.Event()
        self.submitted=self.completed=0
        self.thread=threading.Thread(target=self._run,name='policy-inference-archive',daemon=True);self.thread.start()

    def check(self):
        if self.error is not None:raise RuntimeError('Inference recording failed: '+self.error)

    def submit(self,name,arrays):
        self.check()
        if self.stop.is_set():raise RuntimeError('Inference archive is closed')
        if Path(name).name!=name:raise ValueError('Archive filename must be a basename')
        try:self.queue.put_nowait((name,dict(arrays)))
        except queue.Full as exc:raise RuntimeError('Inference recording backlog full') from exc
        self.submitted+=1

    def _run(self):
        try:
            while not self.stop.is_set() or not self.queue.empty():
                try:name,arrays=self.queue.get(timeout=.05)
                except queue.Empty:continue
                path=self.root/name;temp=path.with_suffix('.tmp')
                with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
                    for key,value in arrays.items():
                        buffer=io.BytesIO();np.lib.format.write_array(buffer,np.asarray(value),allow_pickle=False)
                        archive.writestr(key+'.npy',buffer.getbuffer())
                temp.replace(path);self.completed+=1
        except BaseException as exc:self.error=f'{type(exc).__name__}: {exc}'

    def close(self):
        self.stop.set();self.thread.join(timeout=10)
        if self.thread.is_alive():raise RuntimeError('Inference archive did not finish')
        self.check()
