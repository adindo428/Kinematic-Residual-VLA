"""Compare historical and bounded DLS on identical recorded feedback/targets."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from real_robot.policy_ik import PolicyDifferentialIK,rotation_error
from real_robot.vx300s_kinematics import forward
from real_robot.policy_smoothing import PoseTargetSmoother
from real_robot.policy_trajectory import JointTrajectoryGovernor
from plot_policy_eval import read_rows,paired_actions


def lagged_reference_replay(rows,trials):
    """Illustrative first-order plant, frozen requests; not a VLA rollout."""
    pairs=paired_actions(rows);lookup={f['host_monotonic_s']:a for a,f in pairs};results=[]
    for trial in trials:
        start,end=trial['start_monotonic_s'],trial['end_monotonic_s']
        feedback=[r for r in rows if r['event']=='feedback' and r['active'] and start<=r['host_monotonic_s']<=end]
        first=next((a,f) for a,f in pairs if start<=f['host_monotonic_s']<=end)
        retained=np.array(first[1]['actual_q_rad'])+np.array(first[0]['ik']['following_error_rad'])
        for method in ['uniform_dls','bounded_dls']:
            q=np.array(feedback[0]['actual_q_rad']);diag=first[0]['ik'];rates=diag['joint_rate_ceilings_deg_s']
            ik=PolicyDifferentialIK(q,retained,[.16,0,0],rates,method=method,
                max_tcp_speed_mps=diag['tcp_speed_ceiling_mps'],max_angular_speed_deg_s=diag['angular_speed_ceiling_deg_s'])
            governor=JointTrajectoryGovernor(retained,rates);smoother=PoseTargetSmoother(.08);smoother.reset(forward(retained,[.16,0,0]))
            errors=[];moving=[];failure=None
            try:
                for frame in feedback:
                    action=lookup.get(frame['host_monotonic_s'])
                    if action is None:governor.reset(ik.goal);smoother.reset(forward(ik.goal,[.16,0,0]))
                    else:
                        raw=np.array(action['policy_absolute_action']);target=np.eye(4);target[:3,3]=raw[:3]
                        target[:3,:3]=Rotation.from_euler('xyz',raw[3:6]).as_matrix();target=smoother.step(target,.04)
                        candidate=governor.propose(ik.step(target,q,.04),.04);ik.check_candidate(candidate,q,.04)
                        ik.accept(candidate);governor.accept(candidate)
                        error=float(np.linalg.norm(raw[:3]-forward(q,[.16,0,0])[:3,3])*1000);errors.append(error)
                        if 2.2<frame['host_monotonic_s']-start<5.8:moving.append(error)
                    q+=.65*(ik.goal-q)
            except RuntimeError as exc:failure=str(exc)
            results.append(dict(trial=trial['trial'],method=method,stopped_reason=failure,
                median_position_error_mm=float(np.median(errors)),p95_position_error_mm=float(np.quantile(errors,.95)),
                moving_window_median_position_error_mm=float(np.median(moving))))
    return dict(limitations='Illustrative unvalidated first-order plant (65% tracking per 40ms), frozen recorded requests, no geometry/contact/dynamics model. Not a physical or closed-loop VLA evaluation. Mixed results must not be treated as proof of a fix.',results=results)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    session=args.session.resolve();trials=[json.loads(f.read_text()) for f in sorted((session/'controller').glob('trial-*.json'))]
    rows=read_rows(session/'controller/control.jsonl');pairs=paired_actions(rows);results=[];durations={m:[] for m in ['uniform_dls','bounded_dls']}
    for row,feedback in pairs:
        diag=row['ik'];q=np.array(feedback['actual_q_rad']);retained=q+np.array(diag['following_error_rad'])
        target=np.eye(4);target[:3,3]=row['smoothed_pose6'][:3];target[:3,:3]=Rotation.from_euler('xyz',row['smoothed_pose6'][3:6]).as_matrix()
        dt=diag['position_error_m']/diag['requested_tcp_speed_mps'] if diag['requested_tcp_speed_mps']>1e-12 else .04
        trial=next(t for t in trials if t['start_monotonic_s']<=feedback['host_monotonic_s']<=t['end_monotonic_s'])
        item=dict(trial=trial['trial'],time_s=feedback['host_monotonic_s']-trial['start_monotonic_s'],solvers={})
        for method in durations:
            controller=PolicyDifferentialIK(q,retained,[.16,0,0],diag['joint_rate_ceilings_deg_s'],method=method,
                max_tcp_speed_mps=diag['tcp_speed_ceiling_mps'],max_angular_speed_deg_s=diag['angular_speed_ceiling_deg_s'])
            started=time.perf_counter();candidate=controller.step(target,q,dt);durations[method].append(time.perf_counter()-started)
            pose=forward(candidate,[.16,0,0])
            item['solvers'][method]=dict(position_error_mm=float(np.linalg.norm(target[:3,3]-pose[:3,3])*1000),
                rotation_error_deg=float(np.rad2deg(np.linalg.norm(rotation_error(target[:3,:3],pose[:3,:3])))),
                finite_step_scale=controller.diagnostics['command_scale'])
            if method=='uniform_dls':item['legacy_replay_discrepancy_rad']=float(np.max(abs(candidate-row['ik_candidate_joints_rad'])))
        results.append(item)
    report=dict(hardware_accessed=False,commands=len(results),source=str(session),
        limitations='One-step counterfactual replay on the same measured states and retained goals, before joint trajectory shaping. Not a closed-loop model/robot test; original trajectories had unmatched starts.',
        maximum_legacy_replay_discrepancy_rad=max(r['legacy_replay_discrepancy_rad'] for r in results),solvers={},trials=[])
    for method,times in durations.items():
        errors=np.array([r['solvers'][method]['position_error_mm'] for r in results])
        report['solvers'][method]=dict(position_error_mm_median=float(np.median(errors)),position_error_mm_p95=float(np.quantile(errors,.95)),
            rotation_error_deg_median=float(np.median([r['solvers'][method]['rotation_error_deg'] for r in results])),
            calculation_ms_p99=float(np.quantile(times,.99)*1000))
    phases=json.loads((session/'plots/summary.json').read_text())['trials']
    for trial in trials:
        group=[r for r in results if r['trial']==trial['trial']];phase=next(t for t in phases if t['trial']==trial['trial'])['motion_phases']['moving']
        moving=[r for r in group if phase['first_anchor_s']<=r['time_s']<=phase['last_anchor_s']]
        report['trials'].append(dict(trial=trial['trial'],moving_commands=len(moving),
            moving_position_error_mm_median={method:float(np.median([r['solvers'][method]['position_error_mm'] for r in moving])) for method in durations},
            moving_position_error_mm_p95={method:float(np.quantile([r['solvers'][method]['position_error_mm'] for r in moving],.95)) for method in durations}))
    report['lagged_reference_replay']=lagged_reference_replay(rows,trials)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'samples.json').write_text(json.dumps(results)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(trials),1,figsize=(12,7),squeeze=False)
    for trial,ax in zip(trials,axes[:,0]):
        group=[r for r in results if r['trial']==trial['trial']]
        for method,color in [('uniform_dls','#c76044'),('bounded_dls','#23896f')]:
            ax.plot([r['time_s'] for r in group],[r['solvers'][method]['position_error_mm'] for r in group],label=method,color=color,lw=1)
        ax.set_title('Trial '+str(trial['trial']));ax.set_ylabel('XYZ target error (mm)');ax.grid(alpha=.2);ax.legend()
    axes[-1,0].set_xlabel('Recorded trial time (s)')
    fig.suptitle('IK comparison on identical recorded feedback — one-step replay, before joint smoothing')
    fig.tight_layout();fig.savefig(args.output/'ik_error_comparison.png',dpi=170);fig.savefig(args.output/'ik_error_comparison.pdf');plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
