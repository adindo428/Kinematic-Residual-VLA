"""Wrist EfficientTAM feed for continuous grasping; camera-only process."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .camera import RGBDCamera
from .efficienttam_tracker import EfficientTAMTracker
from .rgbd_archive import RGBDArchiveWriter
from .wrist_alignment import blue_seed, tape_feature


def publish(root, record):
    temporary = root/'observation.tmp.json'
    temporary.write_text(json.dumps(record, allow_nan=False)+'\n')
    temporary.replace(root/'observation.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--template', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed-box', type=int, nargs=4)
    parser.add_argument('--record-rgbd', action='store_true', help='Archive every processed wrist RGB-D frame')
    args = parser.parse_args(argv)
    template = json.loads(Path(args.template).read_text())
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    camera = None
    archive = None
    error = None
    try:
        tracker = EfficientTAMTracker(template['efficienttam_checkpoint'], root,
                                      template['efficienttam_config'], args.device)
        camera = RGBDCamera(template['wrist_serial'])
        if args.record_rgbd:
            archive = RGBDArchiveWriter(root)
        seeded = False
        last_stamp = None
        index = 0
        with (root/'observations.jsonl').open('x', buffering=1) as log:
            while not (root/'STOP').exists():
                image, depth, timing, _ = camera.read()
                stamp = timing['sensor_timestamp_s']
                if last_stamp is not None and stamp <= last_stamp:
                    raise RuntimeError('Wrist timestamp repeated or moved backwards')
                last_stamp = stamp
                processing_started = time.monotonic()
                feature = None
                reason = None
                info = None
                mask = None
                if not seeded:
                    try:
                        box = args.seed_box or blue_seed(image, template['wrist_reference']['center_px'])[0]
                        publish(root, {'schema': 'wrist_pick_observation_v1', 'running': True,
                            'timing': timing, 'feature': None, 'tracker': None,
                            'processing_state': 'seeding',
                            'reason': 'Initializing the first EfficientTAM wrist mask',
                            'host_published_monotonic_s': time.monotonic()})
                        mask, info = tracker.seed('wrist', image, stamp, box)
                        seeded = True
                    except ValueError as exc:
                        reason = str(exc)
                else:
                    mask, info = tracker.track('wrist', image, stamp)
                if mask is not None:
                    try:
                        feature = tape_feature(image, mask, depth)
                    except ValueError as exc:
                        reason = str(exc)
                if feature is None and reason is None:
                    reason = (info or {}).get('mask_reason', 'Tape not visible in wrist view')
                rgbd = None
                if archive is not None:
                    filename = f'frame_{index:06d}_wrist.npz'
                    archive.submit({'wrist': filename}, {'wrist': {'bgr': image, 'depth': depth}})
                    rgbd = {'file': filename, 'frame_index': index}
                    index += 1
                record = {'schema': 'wrist_pick_observation_v1', 'running': True,
                    'timing': timing, 'feature': feature, 'reason': reason,
                    'processing_state': 'tracking' if seeded else 'waiting_for_prompt',
                    'processing_s': time.monotonic()-processing_started,
                    'rgbd': rgbd, 'tracker': info, 'host_published_monotonic_s': time.monotonic()}
                log.write(json.dumps(record, allow_nan=False)+'\n')
                view = image.copy()
                if feature:
                    center = np.array(feature['center_px'])
                    axis = np.array([np.cos(feature['axis_rad']), np.sin(feature['axis_rad'])])
                    cv2.line(view, tuple(np.rint(center-55*axis).astype(int)),
                             tuple(np.rint(center+55*axis).astype(int)), (0,255,0), 2)
                cv2.putText(view, 'WRIST: '+(reason or 'EfficientTAM tape mask'), (8,24),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,255), 1)
                cv2.imwrite(str(root/'live.tmp.jpg'), view)
                (root/'live.tmp.jpg').replace(root/'live.jpg')
                publish(root, record)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        print(error, flush=True)
    finally:
        for resource in (camera, archive):
            if resource is not None:
                try: resource.close()
                except Exception as exc:
                    error = (error+'; ' if error else '')+f'{type(exc).__name__}: {exc}'
        publish(root, {'schema': 'wrist_pick_observation_v1', 'running': False, 'error': error})
    return int(error is not None)


if __name__ == '__main__':
    raise SystemExit(main())
