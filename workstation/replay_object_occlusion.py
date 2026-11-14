"""Replay recorded tracker output through live occlusion logic; opens no hardware."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from real_robot.policy_stream import LiveObjectEstimator,live_object


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-gap-s',type=float,default=.5)
    p.add_argument('--motion-model',choices=['constant_velocity','contact_stop_assumption'],default='constant_velocity')
    a=p.parse_args();manifest=json.loads((a.session/'controller/manifest.json').read_text());rig=manifest['rig_sha256']
    trials=[json.loads(f.read_text()) for f in (a.session/'controller').glob('trial-*.json')]
    estimator=LiveObjectEstimator(a.max_gap_s,motion_model=a.motion_model);counts=Counter();reasons=Counter();max_horizon=0.;frames=[]
    for line in (a.session/'perception/fusion.jsonl').read_text().splitlines():
        row=json.loads(line);views=row['views']
        stamp=min(x['timing']['host_arrival_monotonic_s'] for x in views.values())
        now=row['host_processed_monotonic_s']
        record=dict(schema='front_side_observation_v1',running=True,rig_sha256=rig,output_frame='right_base_estimate',
            stale=not 0<=now-stamp<=.2,timing={'oldest_host_arrival_monotonic_s':stamp},
            track=row.get('track'),velocity_report=row['velocity_report'])
        try:environment=estimator.read(record,now,rig);mode=estimator.diagnostics['source'];error=None
        except ValueError as exc:mode='unavailable';environment=None;error=str(exc)
        in_trial=any(t['start_monotonic_s']<=now<=t['end_monotonic_s'] for t in trials)
        if not in_trial:continue
        try:live_object(record,now,rig);counts['fresh_measurement_frames']+=1
        except ValueError:counts['frames_without_fresh_measurement']+=1
        counts[mode]+=1
        if mode.endswith('_occlusion_prediction'):max_horizon=max(max_horizon,estimator.diagnostics['horizon_s'])
        if error:reasons[error]+=1
        frames.append(dict(host_processed_monotonic_s=now,mode=mode,diagnostics=estimator.diagnostics,
            environment=None if environment is None else environment.tolist()))
    a.output.mkdir(parents=True,exist_ok=True)
    report=dict(hardware_accessed=False,session=str(a.session.resolve()),prediction_settings=estimator.settings,
        trial_tracking_frames=len(frames),counts=dict(counts),max_used_prediction_horizon_s=max_horizon,
        remaining_hold_reasons=dict(reasons),
        interpretation='Offline replay of original per-frame tracker logs at processing time, not control-tick replay or physical success. No object-position ground truth during occlusion; camera/wrist/robot checks remain independent.')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'frames.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in frames))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
