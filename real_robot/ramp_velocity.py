"""Camera-only ramp object masks and automatic velocity profiling."""
import argparse
import csv
from datetime import datetime, timezone
import html
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from .collect import PROJECT, resolve, stop_child, wait_for_perception
from .object_profiles import find_tape_reference, has_prompt, save_prompt
from .recording_lease import LeaseOwner
from .tracking_config import load_config


def analyze(rows, start, end):
    """Use only accepted, ready filter velocities and never bridge missing samples."""
    samples = []
    for row in rows:
        host = row.get('host_processed_monotonic_s', 0)
        if not start <= host <= end:
            continue
        report = row.get('velocity_report') or {}
        track = row.get('track') or {}
        t = (row.get('pair') or {}).get('timestamp_s')
        speed = report.get('speed_mps')
        good = (report.get('status') == 'estimated' and isinstance(t, (int, float))
                and isinstance(speed, (int, float)) and math.isfinite(t) and math.isfinite(speed))
        samples.append({'host_s': host, 'time_s': t, 'speed_mps': speed if good else None,
                        'velocity_mps': report.get('velocity_mps') if good else None,
                        'status': report.get('status'), 'fusion_status': (row.get('fusion') or {}).get('status'),
                        'sources': (row.get('fusion') or {}).get('sources'), 'track_status': track.get('status'),
                        'measurement_method':report.get('measurement_method','rgbd_surface_proxy'),
                        'support_surface':(row.get('fusion') or {}).get('support_surface')})
    area = duration = 0.0
    for a, b in zip(samples, samples[1:]):
        if a['speed_mps'] is None or b['speed_mps'] is None:
            continue
        dt = b['time_s'] - a['time_s']
        if 0 < dt <= .25:
            area += dt * (a['speed_mps'] + b['speed_mps']) / 2
            duration += dt
    return {'samples': samples, 'mean_speed_mps': area / duration if duration else None,
            'valid_duration_s': duration, 'marked_duration_s': end-start,
            'coverage': min(1.0, duration/(end-start)) if end > start else 0,
            'valid_samples': sum(s['speed_mps'] is not None for s in samples)}



class MotionDetector:
    """Arm on a stable track; use hysteresis to detect a complete motion episode."""
    def __init__(self, start_speed=.04, stop_speed=.025, settle_s=.5, stop_s=.4,
                 loss_s=.5, max_motion_s=15):
        self.start_speed, self.stop_speed = start_speed, stop_speed
        self.settle_s, self.stop_s, self.loss_s = settle_s, stop_s, loss_s
        self.max_motion_s = max_motion_s
        self.state = 'settling'
        self.still_since = self.candidate_start = self.stop_since = None
        self.last_valid = self.start = None
        self.high_count = 0

    def update(self, row):
        host = row.get('host_processed_monotonic_s')
        report = row.get('velocity_report') or {}
        speed = report.get('speed_mps')
        if not isinstance(host, (int, float)) or not math.isfinite(host):
            return None
        valid = (report.get('status') == 'estimated' and isinstance(speed, (int, float))
                 and math.isfinite(speed))
        if self.state == 'moving' and host-self.start > self.max_motion_s:
            return ('invalid', 'motion exceeded maximum duration')
        if self.last_valid is not None and host-self.last_valid > self.loss_s:
            if self.state == 'moving':
                return ('invalid', 'tracking lost during motion')
            self.state = 'settling'
            self.still_since = self.candidate_start = None
            self.high_count = 0
        if not valid:
            if self.state == 'settling': self.still_since = None
            if self.state == 'ready': self.candidate_start = None; self.high_count = 0
            if self.state == 'moving': self.stop_since = None
            return None
        self.last_valid = host
        if self.state == 'settling':
            if speed <= self.stop_speed:
                if self.still_since is None: self.still_since = host
                if host-self.still_since >= self.settle_s:
                    self.state = 'ready'
                    return ('ready', None)
            else:
                self.still_since = None
        elif self.state == 'ready':
            if speed >= self.start_speed:
                if self.high_count == 0: self.candidate_start = host
                self.high_count += 1
                if self.high_count >= 2:
                    self.start = self.candidate_start
                    self.state = 'moving'
                    return ('started', self.start)
            else:
                self.high_count = 0; self.candidate_start = None
        elif self.state == 'moving':
            if speed <= self.stop_speed:
                if self.stop_since is None: self.stop_since = host
                if host-self.stop_since >= self.stop_s:
                    return ('complete', (self.start, self.stop_since))
            else:
                self.stop_since = None
        return None



def seek_live_tail(stream):
    """Start at EOF, or at the beginning of a record still being written (binary)."""
    end = stream.seek(0, os.SEEK_END)
    position = end
    while position:
        begin = max(0, position-8192)
        stream.seek(begin)
        chunk = stream.read(position-begin)
        newline = chunk.rfind(b'\n')
        if newline >= 0:
            stream.seek(begin+newline+1)
            return
        position = begin
    stream.seek(0)


def read_complete_record(stream):
    """Do not parse a JSONL record until its terminating newline is published."""
    position = stream.tell()
    line = stream.readline()
    if not line.endswith(b'\n'):
        stream.seek(position)
        return None
    return json.loads(line)

def detect_episode(stream, child, detector=None):
    """Consume new fusion records, checking process and stalled publication."""
    detector = detector or MotionDetector()
    rows = []
    last_row_wall = time.monotonic()
    while True:
        row = read_complete_record(stream)
        if row is None:
            if child.poll() is not None:
                raise RuntimeError('Perception exited; see perception.log')
            if time.monotonic()-last_row_wall > 2:
                raise RuntimeError('Perception stopped publishing for 2 seconds; see perception.log')
            time.sleep(.02)
            continue
        last_row_wall = time.monotonic()
        rows.append(row)
        event = detector.update(row)
        if not event: continue
        kind, value = event
        if kind == 'ready': print('Tracking stable. Push now.', flush=True)
        elif kind == 'started': print('Motion detected; recording...', flush=True)
        elif kind == 'invalid': return None, value
        elif kind == 'complete':
            start, end = value
            return (start, end, analyze(rows, start, end)), None

def save(root, runs):
    groups = {}
    for name in ('ramp_1', 'ramp_2'):
        vals = [r['mean_speed_mps'] for r in runs if r['ramp'] == name]
        groups[name] = {'runs': len(vals), 'mean_of_run_mean_speed_mps': statistics.mean(vals) if vals else None,
                        'sample_std_dev_of_run_means_mps': statistics.stdev(vals) if len(vals) > 1 else None}
    (root/'summary.json').write_text(json.dumps({'ramps': groups, 'runs': runs,
        'semantics': 'Time-weighted magnitude of accepted filtered estimated 3D velocity; measurement method is recorded per run. Standard deviation is across run means; estimates are not ground truth.'}, indent=2, allow_nan=False)+'\n')
    with (root/'runs.csv').open('w', newline='') as f:
        fields = ['measurement_method','surface_calibration','object', 'mask_context', 'ramp', 'run', 'mean_speed_mps', 'valid_duration_s', 'marked_duration_s', 'coverage', 'valid_samples']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(runs)
    plot(root, runs)


def plot(root, runs):
    # Self-contained SVG, so the collection environment needs no plotting package.
    points = [(r, s) for r in runs for s in r['samples'] if s['speed_mps'] is not None]
    xmax = max([r['marked_duration_s'] for r in runs] + [1.0])
    ymax = max([s['speed_mps'] for _, s in points] + [.01]) * 1.1
    w, h = 1000, 580
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="500" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">Object speed by run</text>']
    for j, name in enumerate(('ramp_1', 'ramp_2')):
        x0 = 70 + j*490; top = 65; pw = 410; ph = 420
        parts += [f'<text x="{x0+pw/2}" y="52" text-anchor="middle" font-family="sans-serif">{html.escape(name)}</text>',
                  f'<path d="M{x0} {top} V{top+ph} H{x0+pw}" fill="none" stroke="black"/>']
        for i in range(5):
            y = top+ph-i*ph/4
            parts += [f'<path d="M{x0} {y} H{x0+pw}" stroke="#ddd"/>', f'<text x="{x0-6}" y="{y+4}" text-anchor="end" font-size="12">{ymax*i/4:.2f}</text>']
        for r in (v for v in runs if v['ramp'] == name):
            path = []; previous = None
            for s in r['samples']:
                if s['speed_mps'] is None or s['time_s'] is None:
                    previous = None; continue
                x = x0 + (s['host_s']-r['start_monotonic_s'])/xmax*pw
                y = top+ph-s['speed_mps']/ymax*ph
                command = 'L' if previous is not None and 0 < s['time_s']-previous <= .25 else 'M'
                path.append(f'{command}{x:.1f},{y:.1f}'); previous = s['time_s']
            color = f'hsl({(r["run"]-1)*36},70%,42%)'
            parts.append(f'<path d="{" ".join(path)}" fill="none" stroke="{color}" stroke-width="1.5" opacity=".8"><title>Run {r["run"]}</title></path>')
        parts += [f'<text x="{x0}" y="{top+ph+18}" font-size="12">0</text>', f'<text x="{x0+pw}" y="{top+ph+18}" text-anchor="end" font-size="12">{xmax:.1f} s</text>']
    parts += ['<text x="500" y="540" text-anchor="middle" font-family="sans-serif">Time since detected motion (s)</text>',
              '<text x="15" y="275" transform="rotate(-90 15 275)" text-anchor="middle" font-family="sans-serif">Speed (m/s)</text>', '</svg>']
    (root/'velocity.svg').write_text('\n'.join(parts))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object', choices=['blue_tape', 'striped_ball', 'green_rubber_ball', 'yllow_baseball', 'lemon', 'marker', 'blue_bottle', 'orange'], default='blue_tape',
                   help='Separate EfficientTAM mask libraries for each object')
    p.add_argument('--surface-calibration',type=Path,help='Use measured support planes instead of stereo depth; requires --ramp and fixed geometry')
    p.add_argument('--ramp',choices=['ramp_1','ramp_2'],help='Test only this ramp; required for a per-geometry surface calibration')
    p.add_argument('--min-mask-pixels',type=int,help='Mask area threshold; defaults to 25 for yllow_baseball and 80 otherwise. Depth quality gates are unchanged.')
    p.add_argument('--mask-context', choices=['on_ramp', 'off_ramp'], default='on_ramp',
                   help='Object reference used for tracking')
    p.add_argument('--prepare-masks', action='store_true',
                   help='Create and inspect object references on and off the ramp before profiling')
    p.add_argument('--preview-only',action='store_true',help='Inspect existing mask and depth without selecting a new mask or recording runs')
    p.add_argument('--mask-check', action='store_true',
                   help='Preview only: select a fresh mask, inspect it, then Q to save/finish')
    p.add_argument('--config', default=str(PROJECT/'real_robot/config.yaml'))
    p.add_argument('--output', type=Path)
    p.add_argument('--position-source', choices=['front', 'fused'], default='front',
                   help='Metric tracking source; front RGB-D by default for partial side visibility')
    p.add_argument('--refresh-side-mask', action='store_true',
                   help='Select the tape again in the side view for this session only')
    p.add_argument('--runs', type=int, default=10, help='Accepted runs per ramp (default 10)')
    p.add_argument('--seconds', type=int, default=1800, help='Maximum camera session duration')
    args = p.parse_args(argv)
    if args.surface_calibration:
        if args.ramp is None or args.position_source!='front' or args.refresh_side_mask or args.prepare_masks:
            p.error('Surface speed mode requires --ramp ramp_1/ramp_2 and front-camera tracking without mask preparation')
        args.surface_calibration=resolve(args.surface_calibration)
        from .surface_tracking import SurfaceModel
        SurfaceModel.load(args.surface_calibration)
    if args.min_mask_pixels is None:args.min_mask_pixels=25 if args.object=='yllow_baseball' else 80
    if args.min_mask_pixels<20:p.error('--min-mask-pixels must be >=20')
    if args.runs < 1 or not 1 <= args.seconds <= 1800: p.error('runs must be positive and seconds must be between 1 and 1800')
    if args.prepare_masks:
        if args.object == 'blue_tape': p.error('--prepare-masks requires a non-tape object')
        if args.output: p.error('--prepare-masks names separate session folders automatically; omit --output')
        for context in ('on_ramp', 'off_ramp'):
            print(f'\nMASK CHECK: {context}. Put the object {"on the ramp" if context == "on_ramp" else "off the ramp in the rollout area"}.', flush=True)
            print('Select the entire object. Inspect the contour, gently rotate/move it, then Q in the preview to finish this stage.', flush=True)
            input('Press Enter when the object is positioned and visible to the front camera. ')
            code = main(['--object', args.object, '--mask-context', context, '--mask-check',
                         '--config', args.config, '--seconds', str(args.seconds),'--min-mask-pixels',str(args.min_mask_pixels)])
            if code: return code
        print(f'Both references saved separately. To profile: .venv/bin/python -m real_robot.ramp_velocity --object {args.object}', flush=True)
        return 0
    if args.mask_check and args.position_source != 'front': p.error('Mask checks use the front camera; omit --position-source fused')
    cfg = load_config(args.config)['perception']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    root = resolve(args.output) if args.output else PROJECT/'scratch/real_robot/ramp_velocity'/stamp
    root.mkdir(parents=True, exist_ok=False)
    profile = PROJECT/'scratch/real_robot/object_profiles'/args.object
    if args.object in ('striped_ball','green_rubber_ball','yllow_baseball','lemon','marker','blue_bottle','orange'): profile = profile/args.mask_context
    if args.object == 'striped_ball' and not args.mask_check and not has_prompt(profile, 'front'):
        print('Prepare ball masks first: .venv/bin/python -m real_robot.ramp_velocity --object striped_ball --prepare-masks', file=sys.stderr)
        return 1
    if args.refresh_side_mask:
        session_profile = root/'object_profile'
        if has_prompt(profile, 'front'):
            save_prompt(profile, session_profile, 'front')
        profile = session_profile
    try:
        lock = LeaseOwner(PROJECT/'scratch/real_robot/active-collection.json')
    except ValueError:
        print('Another collection or ramp tracker is running or suspended. Stop it with Ctrl+C in its terminal (if suspended, run fg first), then retry. Do not delete the lock file.', file=sys.stderr)
        return 1
    child = None; runs = []
    try:
        cmd = [sys.executable, '-u', '-m', 'real_robot.multiview_track', '--rig', str(resolve(cfg['rig'])),
               '--output', str(root/'perception'), '--mask-tracker', 'efficienttam', '--object-profile', str(profile),
               '--min-mask-pixels',str(args.min_mask_pixels),'--position-source', args.position_source, '--collection-mode', '--camera-fps', '30', '--save-every', '0', '--seconds', str(args.seconds),
               '--max-skew', str(cfg['max_skew_s']), '--observation-max-age', str(cfg.get('observation_max_age_s', .20)), '--preview']
        for role in ('front', 'side'):
            if role+'_depth_range_m' in cfg: cmd += ['--'+role+'-depth-range', *map(str, cfg[role+'_depth_range_m'])]
        front_only=args.position_source == 'front' and not args.refresh_side_mask
        if front_only:
            cmd += ['--front-mask-only','--front-camera-only']
        if args.surface_calibration:cmd += ['--surface-calibration',str(args.surface_calibration)]
        if args.mask_check:
            cmd += ['--new-mask']
        if args.object == 'blue_tape' and not args.mask_check and not args.refresh_side_mask and not all(has_prompt(profile, role) for role in ('front', 'side')):
            reference = find_tape_reference(PROJECT)
            if reference is not None: cmd += ['--seed-session', str(reference)]
        print(f'Output: {root}\nStarting fixed cameras only; robot motors are not opened.', flush=True)
        if args.refresh_side_mask:
            print('Place the tape in view at the ramp starting position. When the SIDE selection window appears, draw a tight box around the tape and press Enter or Space.', flush=True)
        with (root/'perception.log').open('w') as log:
            env = dict(os.environ); env.setdefault('DISPLAY', ':1')
            child = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            wait_for_perception(child, root/'perception/observation.json',required_views=('front',) if front_only else ('front','side'))
            if args.mask_check or args.preview_only:
                print('Inspect the FRONT contour on the object. Gently move/rotate it in this setting. Press Q in the preview to finish this mask check.', flush=True)
                code = child.wait()
                if code or not has_prompt(profile, 'front'):
                    raise RuntimeError('Mask check did not complete; inspect perception.log')
                print(f'Reference saved: {profile}\nCheck evidence: {root}', flush=True)
                return 0
            print('Keep the object visible in the selected metric camera(s). Preview Q stops cameras.', flush=True)
            for ramp in ((args.ramp,) if args.ramp else ('ramp_1', 'ramp_2')):
                height = input(f'\nSet up {ramp}; enter its height and units (or leave blank): ').strip()
                with (root/'notes.txt').open('a') as notes:
                    notes.write(f'{ramp} height: {height or "not provided"}\n')
                while sum(r['ramp'] == ramp for r in runs) < args.runs:
                    if child.poll() is not None: raise RuntimeError('Perception exited; see perception.log')
                    number = 1 + sum(r['ramp'] == ramp for r in runs)
                    input(f'{ramp} run {number}: reset object, then press Enter once to arm. ')
                    if child.poll() is not None: raise RuntimeError('Perception exited; see perception.log')
                    print('Waiting for a stable visible object...', flush=True)
                    with (root/'perception/fusion.jsonl').open('rb') as stream:
                        seek_live_tail(stream)
                        episode, reason = detect_episode(stream, child)
                    if reason:
                        print(f'Run discarded: {reason}. Reset and repeat.', flush=True)
                        continue
                    start, end, result = episode
                    print(f"Run ended. Valid samples: {result['valid_samples']}; tracked time: {result['valid_duration_s']:.2f}s; coverage: {result['coverage']:.0%}; mean speed: {result['mean_speed_mps'] if result['mean_speed_mps'] is not None else 'unavailable'} m/s", flush=True)
                    if result['valid_samples'] < 3 or result['valid_duration_s'] < .1 or result['coverage'] < .7:
                        print('Insufficient tracking coverage; run discarded. Reset and repeat.', flush=True)
                        continue
                    runs.append({'measurement_method':'rgb_support_plane' if args.surface_calibration else 'rgbd_surface_proxy',
                                 'surface_calibration':str(args.surface_calibration) if args.surface_calibration else None,
                                 'surface_calibration_sha256':hashlib.sha256(args.surface_calibration.read_bytes()).hexdigest() if args.surface_calibration else None,
                                 'object': args.object, 'mask_context': args.mask_context, 'mask_profile': str(profile), 'ramp': ramp, 'run': number, 'start_monotonic_s': start, 'end_monotonic_s': end, **result})
                    save(root, runs)
                    print(f'Run {number} saved.', flush=True)
            print(f'Finished. Results: {root}/summary.json, runs.csv, velocity.svg')
    except (KeyboardInterrupt, EOFError, Exception) as exc:
        print(f'Stopped: {type(exc).__name__}: {exc}. Saved results and diagnostics: {root}', file=sys.stderr)
        return 1
    finally:
        stop_child(child); lock.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
