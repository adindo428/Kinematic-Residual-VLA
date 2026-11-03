"""Separate official completion, secure-lift proxies, and finishing-pose errors."""
import argparse
import json
import pickle
from pathlib import Path
import sys
import numpy as np

def analyze(root):
    protocol=json.loads((root/'protocol.json').read_text())
    sys.path.insert(0,protocol['variants']['original']['source'])
    goals={}
    for test in protocol['tests']:
        cfg=json.loads(Path(test['path']).read_text())
        goals[test['name']]=np.asarray(cfg['terminations']['objects_placed']['params']['goal_position'])
    results={}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder/'verification.json').exists():continue
        rows=[json.loads(x) for x in (folder/'trial_results.jsonl').read_text().splitlines()]
        client=[json.loads(x) for x in (folder/'client_trace.jsonl').read_text().splitlines()]
        values=[]
        for episode,trial in enumerate(rows,1):
            frames=json.loads(Path(trial['telemetry_file']).read_text())
            ee=np.asarray([r['state']['end_effector']['pos'][0] for r in frames])
            obj=np.asarray([r['state']['object']['pos'][0] for r in frames])
            grip=np.asarray([r['action'][0][-1] for r in frames])
            gap=np.asarray([sum(r['finger_joint_positions'][0]) for r in frames])
            distance=np.linalg.norm(ee-obj,axis=1)
            # Diagnostic only: target rises >=5 cm while held close to a commanded-closed EE.
            lifted=(obj[:,2]>=obj[0,2]+.05)&(distance<=.06)&(grip<0)
            longest=current=0
            for value in lifted:
                current=current+1 if value else 0;longest=max(longest,current)
            held=longest*trial['step_dt_s']>=.20
            close=np.flatnonzero((grip[1:]<0)&(grip[:-1]>=0))+1
            first_close=int(close[0]) if len(close) else None
            first_lift=int(np.flatnonzero(lifted)[0]) if lifted.any() else None
            goal_distance=np.linalg.norm(ee-goals[trial['env_name']],axis=1)
            after_lift=goal_distance[first_lift:] if first_lift is not None else np.asarray([])
            # Match exact transmitted poses to applied server poses, including held commands.
            # These are trusted pickles produced by our own frozen evaluation client.
            dumps=sorted((folder/'actions').rglob(trial['env_name']+'-*.pkl'))
            with dumps[trial['trial']-1].open('rb') as stream: dump=pickle.load(stream)
            actions=np.asarray(dump['action'])[:,0,:]
            traces=[r for r in client if r['episode']==episode]
            assert len(actions)==len(traces)
            lookup={}
            for action,trace in zip(actions,traces):
                lookup.setdefault(tuple(action),[]).append(trace)
            lags=[]
            for frame in frames:
                matches=[r for r in lookup.get(tuple(frame['action'][0]),[]) if r['wall_time']<=frame['wall_time']]
                if matches:
                    selected=max(matches,key=lambda r:r['wall_time'])
                    lags.append(frame['index']-selected['action_index'])
            values.append({
                'env_name':trial['env_name'],'trial':trial['trial'],'official_success':trial['success'],
                'secure_lift_proxy':held,'longest_lifted_close_gripper_time_s':longest*trial['step_dt_s'],
                'maximum_object_height_change_m':float(obj[:,2].max()-obj[0,2]),
                'first_close_step':first_close,
                'first_close_ee_object_distance_m':float(distance[first_close]) if first_close is not None else None,
                'minimum_finger_gap_m':float(gap.min()),
                'final_ee_finishing_pose_error_m':float(goal_distance[-1]),
                'minimum_ee_finishing_pose_error_after_first_lift_m':float(after_lift.min()) if len(after_lift) else None,
                'server_applied_action_frames_matched':len(lags),
                'server_applied_action_lag_steps_min_median_max':[int(min(lags)),float(np.median(lags)),int(max(lags))] if lags else None,
            })
        results[folder.name]={'official_successes':sum(r['official_success'] for r in values),
                            'trials':len(values),'secure_lift_proxy_trials':sum(r['secure_lift_proxy'] for r in values),
                            'trials_detail':values}
    output={'definition':'Secure lift is a diagnostic proxy: object >=5 cm above initial root height, EE within 6 cm, close command, for >=0.20 s continuously. It does not replace official completion.',
            'finishing_pose_tolerance_m':.015,'variants':results}
    (root/'grasp_analysis.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='trials_detail'} for k,v in results.items()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();analyze(a.root)
