"""Replay recorded joint targets through the governor; never open robot devices."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from real_robot.policy_trajectory import JointTrajectoryGovernor


def differences(values,indices,order):
    values=np.asarray(values)
    result=np.diff(values,n=order,axis=0)/.04**order
    contiguous=np.array([indices[i+order]-indices[i]==order for i in range(len(indices)-order)])
    return result[contiguous]


def replay(path):
    rows=[json.loads(line) for line in path.open()]
    feedback=next(row for row in rows if row['event']=='feedback')
    initial=np.asarray(feedback['actual_q_rad']);previous=initial.copy()
    governor=JointTrajectoryGovernor(initial,[60.]*6)
    indices=[];raw=[];shaped=[];accelerations=[];velocities=[];times=[];last_index=None
    for row in rows:
        if row['event']!='action' or row['status']!='EXECUTING':continue
        index=row['index'];target=np.asarray(row['candidate_joints_rad'])
        if last_index is not None and index!=last_index+1:governor.reset(previous)
        start=time.perf_counter();candidate=governor.propose(target,.04)
        times.append(time.perf_counter()-start);governor.accept(candidate)
        indices.append(index);raw.append(target);shaped.append(candidate)
        accelerations.append(governor.acceleration.copy());velocities.append(governor.velocity.copy())
        previous=candidate;last_index=index
    report=dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),commands=len(raw),
        governor_max_velocity_deg_s=float(np.rad2deg(np.max(np.abs(velocities)))),
        governor_max_acceleration_deg_s2=float(np.rad2deg(np.max(np.abs(accelerations)))),
        calculation_p99_ms=float(np.quantile(times,.99)*1000),metrics={})
    for order,label in [(1,'joint_speed_deg_s'),(2,'joint_acceleration_deg_s2'),(3,'joint_jerk_deg_s3')]:
        before=np.abs(np.rad2deg(differences(raw,indices,order))).ravel()
        after=np.abs(np.rad2deg(differences(shaped,indices,order))).ravel()
        report['metrics'][label]=dict(raw_p95=float(np.quantile(before,.95)),governed_p95=float(np.quantile(after,.95)),
            governed_max=float(np.max(after)))
    return report,dict(indices=np.array(indices),raw=np.array(raw),governed=np.array(shaped))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('logs',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    report=dict(test_type='offline_recorded_joint_target_replay',dt_s=.04,
        limitations='Numerical reference continuity only. Recorded targets are held fixed; this is not a counterfactual closed-loop policy or physical robot test. Safety holds reset derivatives and are excluded from consecutive-sample metrics.',
        sources=dict(dynamicvla_paper='https://arxiv.org/html/2601.22153v1',
            released_dls='https://github.com/hzxie/DynamicVLA/blob/master/simulations/configs/robot_cfg.py',
            ruckig='https://github.com/pantor/ruckig'),sessions=[])
    for path in args.logs:
        result,arrays=replay(path);report['sessions'].append(result)
        np.savez(args.output/(path.parent.parent.name+'.npz'),**arrays)
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
