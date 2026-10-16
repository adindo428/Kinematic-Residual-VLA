"""Tracking display; the motion arrow uses an approximate pinhole projection."""
from collections import deque

import cv2
import numpy as np


class TrackingOverlay:
    def __init__(self, lookahead_s=0.25, trail_s=1.5):
        self.lookahead_s = lookahead_s
        self.trail_s = trail_s
        self.history = deque()
        self.previous_time = None

    def draw(self, bgr, mask, quality, state, timing, processing_s,
             transform=None, roi=None, frame='camera_optical'):
        view = bgr.copy()
        height, width = view.shape[:2]
        now = timing['sensor_timestamp_s']
        dt = None if self.previous_time is None else now - self.previous_time
        self.previous_time = now
        color = (60, 220, 60) if state is not None and state.ready else (0, 190, 255)
        if roi is not None:
            x, y, w, h = roi
            cv2.rectangle(view, (x, y), (x+w-1, y+h-1), (160, 160, 160), 1)
        if mask is not None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(view, contours, -1, color, 2)
        pixel = quality.get('pixel')
        accepted = state is not None and state.accepted and pixel is not None
        if not accepted or (self.history and now-self.history[-1][0] > 0.25):
            self.history.clear()
        while self.history and now-self.history[0][0] > self.trail_s:
            self.history.popleft()
        if accepted:
            point = tuple(np.rint(pixel).astype(int))
            self.history.append((now, point))
            if len(self.history) > 1:
                cv2.polylines(view, [np.array([p for _, p in self.history], np.int32)],
                              False, (255, 210, 40), 2)
            cv2.circle(view, point, 4, color, -1)

        if state is not None and state.ready:
            position, velocity = state.position_m, state.velocity_mps
            if transform is not None:
                # The filter may use base coordinates; display arrows in camera coordinates.
                rotation, translation = transform[:3, :3], transform[:3, 3]
                position = rotation.T @ (position-translation)
                velocity = rotation.T @ velocity
            future = position + self.lookahead_s*velocity
            if position[2] > 0 and future[2] > 0:
                intr = timing['intrinsics']
                def project(p):
                    return (int(round(intr['fx']*p[0]/p[2]+intr['ppx'])),
                            int(round(intr['fy']*p[1]/p[2]+intr['ppy'])))
                start, end = project(position), project(future)
                visible, start, end = cv2.clipLine((0, 0, width, height), start, end)
                if visible:
                    cv2.arrowedLine(view, start, end, (255, 80, 255), 3, tipLength=0.3)

        panel = np.full((height, 370, 3), 24, np.uint8)
        lines = [
            ('OBJECT TRACKER', (240, 240, 240)),
            ('Status: '+('lost' if state is None else state.status), color),
            ('Frame: '+(frame if state is None else state.frame), (210, 210, 210)),
        ]
        if frame.endswith('_base_estimate'):
            lines.append(('UNVALIDATED frame estimate', (0, 190, 255)))
        if state is not None:
            lines.append(('Motion: '+('stationary assumed' if state.stationary_assumed else 'moving / uncertain'), (210, 210, 210)))
        if state is None and quality.get('reason'):
            lines.append(('Missing: '+quality['reason'], (0, 190, 255)))
        if state is not None:
            lines += [('Position (m)', (220, 220, 220))]
            lines += [(f'  {axis}: {value:+.3f}', (220, 220, 220))
                      for axis, value in zip('XYZ', state.position_m)]
            lines += [('Velocity (m/s)', (220, 220, 220))]
            lines += [(f'  v{axis.lower()}: {value:+.3f}', (220, 220, 220))
                      for axis, value in zip('XYZ', state.velocity_mps)]
            lines += [(f'Speed: {np.linalg.norm(state.velocity_mps):.3f} m/s', color)]
        else:
            lines += [('Position / velocity unavailable', color)]
        lines += [
            (f'Image FPS: {1/dt:.1f}' if dt is not None and dt > 0 else 'Image FPS: warming up', (180, 180, 180)),
            (f'Processing: {1000*processing_s:.1f} ms', (180, 180, 180)),
            ('Cyan: recent measured trail', (255, 210, 40)),
            (f'Magenta: {self.lookahead_s:g}s projected motion*', (255, 80, 255)),
            ('*Approximate pinhole projection', (150, 150, 150)),
            ('Q: close preview', (220, 220, 220)),
        ]
        for i, (label, ink) in enumerate(lines):
            cv2.putText(panel, label, (12, 25+i*23), cv2.FONT_HERSHEY_SIMPLEX,
                        .47, ink, 1, cv2.LINE_AA)
        cv2.putText(view, 'Tracking only | no robot motion', (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1, cv2.LINE_AA)
        return np.concatenate([view, panel], axis=1)
