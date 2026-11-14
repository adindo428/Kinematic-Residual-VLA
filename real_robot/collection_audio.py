"""Nonblocking start/end cues; sound playback never blocks robot control."""
import queue
import shutil
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np


class AudioCues:
    def __init__(self):
        self.folder=tempfile.TemporaryDirectory(prefix='krvla-cues-')
        self.players=[p for p in ('paplay','aplay') if shutil.which(p)]
        self.queue=queue.Queue(maxsize=8)
        self.stop=threading.Event()
        self.warned=False
        self.paths={}
        for name,notes in {'start':[660,990], 'end':[660,440], 'failed':[330,220]}.items():
            parts=[]
            rate=24000
            for frequency in notes:
                t=np.arange(int(rate*.12))/rate
                envelope=np.minimum(1,t/.01)*np.minimum(1,(.12-t)/.02)
                parts += [.3*np.sin(2*np.pi*frequency*t)*envelope,np.zeros(int(rate*.035))]
            samples=(np.concatenate(parts)*32767).astype('<i2')
            path=Path(self.folder.name)/(name+'.wav')
            with wave.open(str(path),'wb') as f:
                f.setnchannels(1);f.setsampwidth(2);f.setframerate(rate);f.writeframes(samples.tobytes())
            self.paths[name]=path
        self.thread=threading.Thread(target=self._run,name='collection-sound',daemon=True)
        self.thread.start()

    def play(self,name):
        if name not in self.paths:raise ValueError('Unknown collection sound')
        try:self.queue.put_nowait(name)
        except queue.Full:
            if not self.warned:print('Audio cues delayed: sound queue full',flush=True);self.warned=True

    def _run(self):
        while not self.stop.is_set() or not self.queue.empty():
            try:name=self.queue.get(timeout=.1)
            except queue.Empty:continue
            played=False
            for player in self.players:
                command=[player,str(self.paths[name])] if player=='paplay' else [player,'-q',str(self.paths[name])]
                try:
                    result=subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=2)
                    if result.returncode==0:played=True;break
                except (OSError,subprocess.TimeoutExpired):pass
            if not played and not self.warned:
                print('Audio cue unavailable: check desktop sound output. Visual confirmations remain active.',flush=True)
                self.warned=True

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
        if not self.thread.is_alive():self.folder.cleanup()
