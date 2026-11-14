"""Rerun collection preview and dedicated Enter/X/T/Q keyboard controls."""
import os
import sys
import select
import termios
import tty
import queue
import threading
import time
from pathlib import Path

import cv2

from .episode_feedback import render_banner


class TerminalKeys:
    """Read individual keys from the launching terminal, restoring its settings."""
    def __init__(self, stream=None):
        self.stream = sys.stdin if stream is None else stream
        if not self.stream.isatty():
            raise ValueError('Terminal keyboard input requires an interactive terminal. Run the command in your terminal, or use --keyboard-source global.')
        self.fd = self.stream.fileno()
        self.original = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def poll(self):
        if not select.select([self.fd], [], [], 0)[0]: return 255
        key = os.read(self.fd, 1)
        if not key: raise RuntimeError('Collection terminal input closed')
        if key in (b'\r', b'\n'): return 13
        if key == b' ': return 32
        key = key.lower()
        return key[0] if key in (b'n', b'v', b'x', b't', b'q', b'h', b'p', b'c', b'l', b'r') else 255

    def close(self):
        termios.tcsetattr(self.fd, termios.TCSANOW, self.original)


class RerunPreview:
    """Keep viewer serialization and disk preview reads off the recorder loop."""
    def __init__(self, perception_dir, session_name, keyboard_source='terminal'):
        import rerun as rr
        import rerun.blueprint as rrb
        self.rr = rr
        self.perception_dir = Path(perception_dir)
        self.keys = queue.Queue(maxsize=32)
        self.frames = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.error = None
        self.held = set()
        self.status = None
        os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH','')
        rr.init('real_robot_collection', recording_id=session_name)
        rr.spawn(memory_limit='1GB')
        rr.send_blueprint(rrb.Blueprint(rrb.Vertical(
            rrb.Spatial2DView(origin='episode_status', name='Episode confirmation'),
            rrb.Horizontal(rrb.Spatial2DView(origin='cameras/front'), rrb.Spatial2DView(origin='cameras/side'),
                           rrb.Spatial2DView(origin='cameras/wrist')),
            rrb.TextDocumentView(origin='status'), row_shares=[1.4, 4, 1])))

        self.terminal = TerminalKeys() if keyboard_source == 'terminal' else None
        self.listener = None
        if self.terminal is None:
            self._start_global_listener()
        self.thread = threading.Thread(target=self._run, daemon=True, name='rerun-collection-preview')
        self.thread.start()

    def _start_global_listener(self):
        from pynput import keyboard

        def keycode(key):
            if key == keyboard.Key.enter: return 13
            if key == keyboard.Key.space: return 32
            if key == keyboard.Key.esc: return 27
            char = getattr(key, 'char', None)
            return ord(char.lower()) if char and char.lower() in 'nxvtqhpclr' else None

        def press(key):
            code = keycode(key)
            if code is not None and code not in self.held:
                self.held.add(code)
                try: self.keys.put_nowait(code)
                except queue.Full: self.error = 'Collection keyboard queue overflow'

        def release(key):
            self.held.discard(keycode(key))

        self.listener = keyboard.Listener(on_press=press, on_release=release)
        self.listener.start()
        self.listener.wait()

    def _run(self):
        version = None
        rendered_status = None
        try:
            while True:
                try: frame_item = self.frames.get(timeout=.1)
                except queue.Empty: frame_item = None
                rr = self.rr
                status = self.status
                if status is not None and status != rendered_status:
                    rr.log('episode_status', rr.Image(render_banner(status)), static=True)
                    rendered_status = status
                if frame_item is None:
                    if self.stop.is_set(): break
                    continue
                image, labels = frame_item
                rr.set_time_seconds('session', time.monotonic())
                if image is not None:
                    rr.log('cameras/wrist', rr.Image(image[:, :, ::-1]).compress(jpeg_quality=80))
                path = self.perception_dir/'live_frame.jpg'
                if path.exists():
                    stamp = path.stat().st_mtime_ns
                    if stamp != version:
                        frame = cv2.imread(str(path))
                        if frame is not None and frame.shape[0] >= 480 and frame.shape[1] >= 1280:
                            for role, x in [('front', 0), ('side', 640)]:
                                rr.log('cameras/'+role, rr.Image(frame[:480, x:x+640, ::-1]).compress(jpeg_quality=80))
                            version = stamp
                rr.log('status', rr.TextDocument('\n'.join(labels)), static=True)
                self.stop.wait(.1)
        except BaseException as exc:
            self.error = f'Rerun preview failed: {type(exc).__name__}: {exc}'

    def set_status(self, status):
        # Publish independently of the lossy camera-preview queue.
        self.status = dict(status)

    def poll(self, image, labels):
        if self.error: raise RuntimeError(self.error)
        try: self.frames.put_nowait((image, labels))
        except queue.Full: pass
        if self.terminal is not None: return self.terminal.poll()
        try: return self.keys.get_nowait()
        except queue.Empty: return 255

    def close(self):
        if self.listener is not None: self.listener.stop()
        if self.terminal is not None: self.terminal.close()
        self.stop.set()
        self.thread.join(timeout=2)
        self.rr.disconnect()
