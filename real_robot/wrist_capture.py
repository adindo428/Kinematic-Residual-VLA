"""Restartable wrist acquisition, isolated from robot control and native SDK hangs."""
from collections import deque
import ctypes
import json
import multiprocessing as mp
from pathlib import Path
import threading
import time

import numpy as np

PHASES={0:'starting',1:'waiting_for_frames',2:'aligning_depth',3:'copying_frame',
        4:'recording',5:'closing_camera',6:'stopped',7:'error'}


class SharedFlag:
    """Lock-free single-byte flag; killing a camera worker cannot strand a lock."""
    def __init__(self,value): self.value=value
    def set(self): self.value.value=1
    def clear(self): self.value.value=0
    def is_set(self): return bool(self.value.value)


def camera_worker(root,serial,fps,restart,enabled,stop,pixels,sequence,timing,phase,error):
    from .camera import RGBDCamera
    from .rgbd_archive import RGBDArchiveWriter
    root=Path(root)
    camera=archive=None
    def stage(name):
        phase.value={'wait_frames':1,'align_depth':2,'copy_arrays':3}.get(name,3)
    try:
        camera=RGBDCamera(serial,fps=fps)
        archive=RGBDArchiveWriter(root,compressed=True,workers=2)
        previous=domain=None
        pending=deque()
        index=0
        with (root/'frames.jsonl').open('a',buffering=1) as log:
            while not stop.value:
                bgr,depth,meta,_=camera.read(report_stage=stage)
                stamp=meta['sensor_timestamp_s']
                if ((previous is not None and stamp<=previous) or
                        (domain is not None and meta['timestamp_domain']!=domain)):
                    raise RuntimeError('Wrist clock repeated or changed within one camera connection')
                previous,domain=stamp,meta['timestamp_domain']
                if bgr.shape!=(480,640,3) or bgr.dtype!=np.uint8:
                    raise RuntimeError('Unexpected wrist camera image shape/type')
                available=time.monotonic()
                sequence.value+=1
                np.frombuffer(pixels,dtype=np.uint8)[:]=bgr.reshape(-1)
                timing[0]=meta['host_arrival_monotonic_s']
                timing[1]=available
                sequence.value+=1
                archive.check()
                phase.value=4
                if enabled.value:
                    name=f'frame_{restart:03d}_{index:07d}.npz'
                    archive.submit({'wrist':name},{'wrist':{'bgr':bgr,'depth':depth}})
                    pending.append(dict(file='wrist/'+name,timing=meta,
                                        available_monotonic_s=available,camera_restart=restart))
                    index+=1
                while pending and (root/Path(pending[0]['file']).name).is_file():
                    row=pending.popleft();row['archive_ready_monotonic_s']=time.monotonic()
                    log.write(json.dumps(row,allow_nan=False)+'\n')
            archive.close();archive=None
            while pending:
                row=pending.popleft()
                if not (root/Path(row['file']).name).is_file():
                    raise RuntimeError('Wrist archive did not publish a queued frame')
                row['archive_ready_monotonic_s']=time.monotonic()
                log.write(json.dumps(row,allow_nan=False)+'\n')
    except BaseException as exc:
        error.value=f'{type(exc).__name__}: {exc}'.encode()[:1023]
        phase.value=7
    finally:
        if archive is not None:
            try: archive.close()
            except BaseException: pass
        if camera is not None:
            phase.value=5
            try: camera.close()
            except BaseException as exc: error.value=f'Camera close: {exc}'.encode()[:1023]
        phase.value=6


class EpisodeWrist:
    """Same recorder interface, with a bounded watchdog around the SDK process."""
    def __init__(self,root,serial,fps=30,*,stall_s=2.,startup_s=15.,max_restarts=3,
                 worker_target=camera_worker):
        self.root,self.serial,self.fps=Path(root),serial,fps
        self.root.mkdir(parents=True)
        self.context=mp.get_context('spawn')
        self.enabled=SharedFlag(self.context.RawValue(ctypes.c_ubyte,0))
        self.stop=threading.Event()
        self.lock=threading.Lock()
        self.error=None
        self.restarts=0
        self.last_worker_error=None
        self.last_phase='starting'
        self.recovering=True
        self.cached_image=self.cached_arrival=None
        self.cached_sequence=-1
        self.stall_s,self.startup_s,self.max_restarts=stall_s,startup_s,max_restarts
        self.worker_target=worker_target
        self.process=None
        self.thread=threading.Thread(target=self.run,name='wrist-supervisor',daemon=True)

    @staticmethod
    def stop_process(process,stop):
        stop.value=1
        process.join(timeout=1)
        if process.is_alive():
            process.terminate();process.join(timeout=1)
        if process.is_alive():
            process.kill();process.join(timeout=1)
        if process.is_alive(): raise RuntimeError('Wrist camera process could not be stopped')

    def save_status(self,event):
        status=dict(event=event,restarts=self.restarts,phase=self.last_phase,
                    detail=self.last_worker_error,host_monotonic_s=time.monotonic())
        with (self.root/'camera_events.jsonl').open('a',buffering=1) as stream:
            stream.write(json.dumps(status)+'\n')

    def run(self):
        try:
            while not self.stop.is_set():
                pixels=self.context.RawArray(ctypes.c_ubyte,480*640*3)
                sequence=self.context.RawValue(ctypes.c_uint64,0)
                timing=self.context.RawArray(ctypes.c_double,2)
                phase=self.context.RawValue(ctypes.c_int,0)
                error=self.context.RawArray(ctypes.c_char,1024)
                worker_stop=self.context.RawValue(ctypes.c_ubyte,0)
                process=self.context.Process(target=self.worker_target,
                    args=(str(self.root),self.serial,self.fps,self.restarts,self.enabled.value,
                          worker_stop,pixels,sequence,timing,phase,error),daemon=True)
                self.process=process
                process.start()
                started=time.monotonic();last_sequence=-1;reason=None
                try:
                    self.last_phase='starting'
                    self.save_status('worker_started')
                    while not self.stop.wait(.02):
                        self.last_phase=PHASES.get(phase.value,'unknown')
                        now=time.monotonic()
                        seq=sequence.value
                        if seq and not seq%2 and seq!=last_sequence:
                            image=np.frombuffer(pixels,dtype=np.uint8).reshape(480,640,3).copy()
                            arrival=timing[0]
                            if sequence.value==seq:
                                with self.lock:
                                    self.cached_image,self.cached_arrival=image,arrival
                                    self.recovering=False
                                last_sequence=seq
                        if not process.is_alive():
                            reason=error.value.decode(errors='replace') or f'Camera worker exited ({process.exitcode})'
                            break
                        if (not timing[0] and now-started>self.startup_s) or (timing[0] and now-timing[0]>self.stall_s):
                            reason=f'Wrist stalled during {self.last_phase}'
                            break
                finally:
                    self.stop_process(process,worker_stop)
                if self.stop.is_set(): break
                self.last_worker_error=reason
                self.recovering=True
                self.save_status('restarting')
                if self.restarts>=self.max_restarts:
                    raise RuntimeError(f'Wrist recovery failed after {self.max_restarts} restarts: {reason}')
                self.restarts+=1
                self.stop.wait(.3)
        except BaseException as exc:
            with self.lock: self.error=f'{type(exc).__name__}: {exc}'
            try: self.save_status('failed')
            except OSError: pass

    def snapshot(self):
        with self.lock:
            if self.error: raise RuntimeError(self.error)
            return self.cached_image,self.cached_arrival

    def close(self):
        self.enabled.clear();self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=5)
            if self.thread.is_alive(): raise RuntimeError('Wrist supervisor did not stop')
