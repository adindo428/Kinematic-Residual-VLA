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
from .wrist_alignment import tape_feature
from .wrist_detection import read_hint, tape_prompt, associated_mask


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
        failures = 0
        generation = 0
        last_prompt_attempt = -float('inf')
        active_prompt_id = None
        handled_prompt_id = None
        manual_override = args.seed_box is not None
        initial_box = args.seed_box
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
                hint = read_hint(root/'target_hint.json',timing,template)
                candidate = None
                candidate_box = None
                detection = None
                try:
                    candidate_box,candidate,detection = tape_prompt(image,depth,timing,template,
                        None if manual_override else hint)
                except ValueError as exc:
                    reason = str(exc)
                request = None
                try:
                    value = json.loads((root/'prompt_request.json').read_text())
                    if value['request_id'] != handled_prompt_id:
                        request = value
                        handled_prompt_id = value['request_id']
                        active_prompt_id = handled_prompt_id
                        manual_override = True
                except (OSError,ValueError,KeyError):
                    pass
                seed_box = request['box_xywh'] if request is not None else (initial_box or candidate_box)
                prompt_due = request is not None or (not seeded and seed_box is not None
                    and time.monotonic()-last_prompt_attempt>=.75)
                if prompt_due:
                    try:
                        last_prompt_attempt = time.monotonic()
                        publish(root, {'schema': 'wrist_pick_observation_v1', 'running': True,
                            'timing': timing, 'feature': None, 'tracker': None,
                            'processing_state': 'seeding',
                            'reason': 'Prompting EfficientTAM on the associated wrist tape',
                            'host_published_monotonic_s': time.monotonic()})
                        seed_image,seed_stamp = image,stamp
                        if request is not None and request.get('frame_file'):
                            filename = Path(request['frame_file']).name
                            with np.load(root/filename) as frame:
                                seed_image = frame['bgr'].copy()
                            seed_stamp = float(request['sensor_timestamp_s'])
                        mask, info = tracker.seed('wrist', seed_image, seed_stamp, seed_box)
                        if seed_stamp < stamp:
                            mask,info = tracker.track('wrist',image,stamp)
                        generation += 1
                        initial_box = None
                        seeded = True
                        failures = 0
                        reason = None
                        detection = {'source':'manual_prompt','box_xywh':seed_box} if manual_override else detection
                    except (OSError,ValueError,KeyError) as exc:
                        seeded = False
                        reason = str(exc)
                elif seeded:
                    mask, info = tracker.track('wrist', image, stamp)
                if mask is not None:
                    try:
                        associated,why = associated_mask(mask,None if manual_override else candidate,
                            None if manual_override else hint,timing)
                        if not associated:
                            raise ValueError(why)
                        feature = tape_feature(image, mask, depth)
                        reason = ('Tape mask visible; wrist depth unavailable' if feature['depth_m'] is None
                                  else 'Tape mask visible; axis uncertain' if not feature['axis_reliable'] else None)
                    except ValueError as exc:
                        reason = str(exc)
                        mask = None
                if seeded:
                    failures = failures+1 if feature is None else 0
                    if failures>=3:
                        seeded = False
                if feature is None and reason is None:
                    reason = (info or {}).get('mask_reason', 'Tape not visible in wrist view')
                if info is not None:
                    info.update(prompt_generation=generation,prompt_id=active_prompt_id,
                        recovery_pending=not seeded)
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
                    'detection':detection,'target_hint':hint,
                    'rgbd': rgbd, 'tracker': info, 'host_published_monotonic_s': time.monotonic()}
                log.write(json.dumps(record, allow_nan=False)+'\n')
                view = image.copy()
                if mask is not None:
                    selected = mask>0
                    view[selected] = (.65*view[selected]+.35*np.array([0,255,0])).astype(np.uint8)
                if hint is not None and hint.get('in_view'):
                    cv2.drawMarker(view,tuple(np.rint(hint['center_px']).astype(int)),(255,255,0),cv2.MARKER_CROSS,18,2)
                if candidate_box is not None:
                    x,y,w,h = candidate_box
                    cv2.rectangle(view,(x,y),(x+w,y+h),(0,255,255),1)
                if feature:
                    center = np.array(feature['center_px'])
                    axis = np.array([np.cos(feature['axis_rad']), np.sin(feature['axis_rad'])])
                    if feature['axis_reliable']:
                        cv2.line(view, tuple(np.rint(center-55*axis).astype(int)),
                                 tuple(np.rint(center+55*axis).astype(int)), (0,255,0), 2)
                    x,y,w,h = feature['bbox_xywh']
                    cv2.rectangle(view,(x,y),(x+w,y+h),(0,255,0),2)
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
