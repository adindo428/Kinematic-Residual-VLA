"""Replay pinned Hugging Face expert Cartesian actions through the policy controller.

No VLA or object tracker is loaded. Prepare/audit/plot do not open robot devices.
Physical replay is in replay_control and starts paused with an operator present.
"""
import argparse
from bisect import bisect_left,bisect_right
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from .dom_dataset import command_raw,command_action,json_write,jsonl_read,pose6,sha256
from .vx300s_kinematics import forward,joints_from_raw_ticks
from .policy_ik import PolicyDifferentialIK
from .policy_smoothing import PoseTargetSmoother
from .policy_trajectory import JointTrajectoryGovernor
from .policy_stream import action_clock_index

PROJECT=Path(__file__).resolve().parents[1]
REPO='ailab-Clemson/KRVLA-dataset'
REVISION='439d8d9051079b0c4459f48256ac83f1760116cf'


def replay_frame(elapsed,period):
    """Nearest nominal sample for a paced device loop, not interval flooring.

    The loop samples near t=k*period. Flooring makes sub-ms timer jitter
    alternate between duplicate and skipped frames at those boundaries.
    A true delay beyond half a period still advances/skips the source clock;
    no queued catch-up bursts or replay time stretching are introduced.
    """
    return action_clock_index(elapsed,period)


def transform(pose):
    pose=np.asarray(pose,float)
    if pose.shape!=(6,) or not np.isfinite(pose).all():raise ValueError('Expected finite absolute XYZ/Euler pose')
    t=np.eye(4);t[:3,3]=pose[:3];t[:3,:3]=Rotation.from_euler('xyz',pose[3:6]).as_matrix();return t


class ReplayPipeline:
    """Same hardware-rate DLS, optional 80ms blend and Ruckig as policy_eval."""
    def __init__(self,actual,retained,rates,tcp,*,smoothing_ms=80.,governor=True,acceleration_deg_s2=120.,jerk_deg_s3=600.):
        self.tcp=np.asarray(tcp,float)
        self.ik=PolicyDifferentialIK(actual,retained,tcp,rates,max_tcp_speed_mps=None,max_angular_speed_deg_s=None)
        self.smoother=PoseTargetSmoother(smoothing_ms/1000) if smoothing_ms else None
        self.governor=JointTrajectoryGovernor(retained,rates,acceleration_deg_s2,jerk_deg_s3) if governor else None
        self.hold()

    def hold(self):
        if self.smoother is not None:self.smoother.reset(forward(self.ik.goal,self.tcp))
        if self.governor is not None:self.governor.reset(self.ik.goal)

    def propose(self,action,actual,dt):
        target=transform(np.asarray(action)[:6]);filtered=self.smoother.step(target,dt) if self.smoother else target
        ik=self.ik.step(filtered,actual,dt);command=self.governor.propose(ik,dt) if self.governor else ik
        self.ik.check_candidate(command,actual,dt)
        return command,dict(target_pose6=pose6(target).tolist(),smoothed_pose6=pose6(filtered).tolist(),
            ik_pose6=pose6(forward(ik,self.tcp)).tolist(),command_pose6=pose6(forward(command,self.tcp)).tolist(),
            ik_candidate_joints_rad=ik.tolist(),command_joints_rad=command.tolist(),ik=self.ik.diagnostics,
            trajectory=None if self.governor is None else self.governor.diagnostics)

    def accept(self,command):
        self.ik.accept(command)
        if self.governor is not None:self.governor.accept(command)


def prepare(output,episodes,revision=REVISION):
    from huggingface_hub import hf_hub_download
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);snapshot=output/'huggingface'
    def fetch(name):return Path(hf_hub_download(REPO,name,repo_type='dataset',revision=revision,local_dir=snapshot,token=False))
    sources=jsonl_read(fetch('meta/sources.jsonl'));info=json.loads(fetch('meta/info.json').read_text())
    descriptions=jsonl_read(fetch('meta/episodes.jsonl'));prepared=[];cache={}
    for ep in episodes:
        if 'marker' not in descriptions[ep]['tasks'][0]:raise ValueError('This batch is intended for recorded marker demonstrations')
        parquet=fetch(f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet')
        quality=jsonl_read(fetch(f'quality/episode_{ep:06d}.jsonl'));rows=pq.read_table(parquet).to_pylist()
        if len(rows)!=len(quality) or info['fps']!=25:raise ValueError('Episode/quality alignment or sample-rate mismatch')
        source=sources[ep];session=Path(source['raw_session'])
        if session not in cache:
            manifest=json.loads((session/'manifest.json').read_text());robot=jsonl_read(session/'robot.jsonl')
            measured=sorted(robot,key=lambda r:r['host_read_end_monotonic_s'])
            commands=sorted([r for r in robot if r.get('commanded_calibrated') is not None],key=lambda r:r['host_cycle_end_monotonic_s'])
            cache[session]=(manifest,measured,[r['host_read_end_monotonic_s']for r in measured],commands,[r['host_cycle_end_monotonic_s']for r in commands])
        manifest,measured,mt,commands,ct=cache[session];qs=[];goals=[];grips=[];command_grips=[]
        for row,q in zip(rows,quality):
            stamp=q['anchor_monotonic_s'];mi=bisect_right(mt,stamp)-1;ci=bisect_left(ct,stamp)
            if mi<0 or ci>=len(ct) or stamp-mt[mi]>.12 or ct[ci]-stamp>.12:raise ValueError('Raw joint supplement is not aligned to the HF frame')
            m,c=measured[mi],commands[ci]
            actual=joints_from_raw_ticks(m['follower_raw_ticks']);raw=command_raw(c['commanded_calibrated'],manifest['buses']['follower']['calibration'])
            target=command_action(c,manifest);state=pose6(forward(actual,manifest['tcp_offset_m']))
            if not np.allclose(target,row['action'],atol=8e-7,rtol=0) or not np.allclose(state,row['observation.state'],atol=8e-7,rtol=0):raise ValueError(f'Raw FK/action supplement differs from HF episode {ep} frame {row["frame_index"]}')
            qs.append(actual);goals.append(joints_from_raw_ticks(raw));grips.append(m['follower_raw_ticks'][-1]);command_grips.append(int(raw[-1]))
        name=f'episode_{ep:06d}.npz'
        np.savez_compressed(output/name,timestamp=np.array([r['timestamp']for r in rows]),
            actions=np.array([r['action']for r in rows]),expert_measured_pose6=np.array([r['observation.state']for r in rows]),
            expert_measured_joints=np.array(qs),expert_command_joints=np.array(goals),
            expert_measured_gripper_raw=np.array(grips),expert_command_gripper_raw=np.array(command_grips))
        prepared.append(dict(episode=ep,file=name,frames=len(rows),duration_s=len(rows)/info['fps'],
            source_url=f'https://huggingface.co/datasets/{REPO}/blob/{revision}/data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet',
            hf_parquet_sha256=sha256(parquet),archive_sha256=sha256(output/name),raw_session=str(session),
            raw_robot_sha256=sha256(session/'robot.jsonl'),tcp_offset_m=manifest['tcp_offset_m'],raw_fk_verified_against_all_hf_actions_and_states=True))
    contract=dict(schema='krvla_recorded_action_replay_v1',repo=REPO,revision=revision,fps=info['fps'],episodes=prepared,
        selection='Five specified marker episodes, spaced across the most recent marker collection; no trajectory recentering/time scaling.',
        action_source='Pinned HF absolute commanded EE pose and continuous gripper; no VLA and no object velocity conditioning.',
        joint_supplement='Local original encoder/command records, verified numerically against every HF robot/action row; joints are not claimed to be HF features.',
        source_action_semantics='FK of issued joint command, not measured future EE pose',
        stages=['HF action','smoothed target','bounded DLS','Ruckig command','live measured FK'],
        controller=dict(motion_profile='hardware',geometry_profile='grasp',ee_workspace_enforced=False,
            software_velocity_caps=False,pose_smoothing_ms=80,joint_acceleration_deg_s2=120,joint_jerk_deg_s3=600,
            reset_joint_speed_deg_s=15),hardware_accessed=False)
    if any(not np.allclose(item['tcp_offset_m'],prepared[0]['tcp_offset_m'],atol=1e-9,rtol=0)for item in prepared):raise ValueError('Mixed TCP definitions cannot share a replay batch')
    json_write(output/'replay_contract.json',contract);print(json.dumps(contract,indent=2));return contract


def error_metrics(target,actual):
    target,actual=np.asarray(target),np.asarray(actual)
    if not len(target) or not len(actual):return dict(samples=0,position_rmse_mm=None,position_median_mm=None,position_p95_mm=None,orientation_median_deg=None,orientation_p95_deg=None)
    if target.shape!=actual.shape or target.ndim!=2 or target.shape[1]!=6:raise ValueError('Compare matched Nx6 pose sequences')
    distance=np.linalg.norm(target[:,:3]-actual[:,:3],axis=1)*1000
    rotations=Rotation.from_euler('xyz',target[:,3:6])*Rotation.from_euler('xyz',actual[:,3:6]).inv()
    angle=np.rad2deg(rotations.magnitude())
    return dict(samples=len(distance),position_rmse_mm=float(np.sqrt(np.mean(distance**2))),
        position_median_mm=float(np.median(distance)),position_p95_mm=float(np.quantile(distance,.95)),
        orientation_median_deg=float(np.median(angle)),orientation_p95_deg=float(np.quantile(angle,.95)))


def audit(batch,rates=(180.,)*6):
    batch=Path(batch);contract=json.loads((batch/'replay_contract.json').read_text());results=[]
    for item in contract['episodes']:
        with np.load(batch/item['file'])as d:data={k:d[k]for k in d.files}
        actions=data['actions'];tcp=item['tcp_offset_m'];result=dict(episode=item['episode'],
            expert_command_vs_expert_measured=error_metrics(actions[:,:6],data['expert_measured_pose6']))
        modes={}
        for mode,smooth,govern in [('direct_dls_ideal_plant',0,False),('pose_blend_only_ideal_plant',80,False),
                ('governor_only_ideal_plant',0,True),('full_pipeline_ideal_plant',80,True)]:
            q=data['expert_command_joints'][0].copy();pipeline=ReplayPipeline(q,q,rates,tcp,smoothing_ms=smooth,governor=govern)
            poses=[];stage=[];error=None
            for action in actions:
                try:command,diagnostic=pipeline.propose(action,q,1/contract['fps'])
                except (ValueError,RuntimeError)as exc:error=str(exc);break
                pipeline.accept(command);q=command.copy();poses.append(pose6(forward(q,tcp)));stage.append(diagnostic)
            name=f'audit-{item["episode"]:06d}-{mode}.npz'
            np.savez_compressed(batch/name,pose6=np.array(poses),target=actions[:len(poses),:6],timestamp=data['timestamp'][:len(poses)])
            modes[mode]=dict(error_metrics(actions[:len(poses),:6],poses),completed_frames=len(poses),total_frames=len(actions),error=error)
        result['numerical_replay']=modes;results.append(result)
    report=dict(hardware_accessed=False,rates_assumed_deg_s=list(rates),episodes=results,
        note='Ideal-plant outputs are numerical simulations, never live actual EE. Geometry is not enforced in these numerical tracking curves. Hardware results are required to diagnose motor following.')
    json_write(batch/'offline_audit.json',report);print(json.dumps(report,indent=2));return report


def plot(batch,run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    batch,run=Path(batch),Path(run);out=run/'plots';out.mkdir(exist_ok=True)
    contract=json.loads((batch/'replay_contract.json').read_text());summary=[];overview=[]
    for item in contract['episodes']:
        log=run/f'episode_{item["episode"]:06d}.jsonl'
        if not log.exists():continue
        samples=[r for r in jsonl_read(log)if r.get('event')=='sample']
        if not samples:continue
        with np.load(batch/item['file'])as d:t=d['timestamp'];target=d['actions'][:,:6];expert=d['expert_measured_pose6']
        times=np.array([r['episode_elapsed_s']for r in samples]);actual=np.array([r['actual_pose6']for r in samples])
        raw=np.array([r['target_pose6']for r in samples]);ik=np.array([r['ik_pose6']for r in samples]);command=np.array([r['command_pose6']for r in samples])
        frame_indices=np.array([r['source_frame']for r in samples])
        overview.append((item['episode'],t,target,expert,times,actual,ik))
        fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
        for k,label in enumerate('XYZ'):
            ax=axes[k];ax.plot(t,target[:,k],color='black',alpha=.7,label='HF commanded EE target')
            ax.plot(t,expert[:,k],'--',color='.6',label='Recorded expert measured EE')
            ax.plot(times,ik[:,k],color='#008080',alpha=.6,label='DLS result')
            ax.plot(times,command[:,k],color='#d98600',label='Governed command FK')
            ax.plot(times,actual[:,k],color='#2463ba',label='Live measured EE (before command)')
            ax.set_ylabel(label+' (m)');ax.grid(alpha=.2)
        axes[0].legend(ncol=2,fontsize=8);axes[-1].set_xlabel('Episode time (s)');fig.suptitle(f'Recorded-action replay: marker episode {item["episode"]} — no VLA')
        fig.tight_layout();fig.savefig(out/f'episode_{item["episode"]:06d}.png',dpi=150);fig.savefig(out/f'episode_{item["episode"]:06d}.pdf');plt.close(fig)
        summary.append(dict(episode=item['episode'],target_vs_live_measured=error_metrics(raw,actual),
            target_vs_ik=error_metrics(raw,ik),ik_vs_governed=error_metrics(ik,command),
            same_tick_command_vs_measured=error_metrics(command,actual),
            expert_command_vs_recorded_expert_measured=error_metrics(target,expert),
            live_vs_recorded_expert_measured=error_metrics(expert[frame_indices],actual),
            note='Same-clock comparisons include latency and natural following error; measured feedback is read before that tick command. Inspect full curves, not only aggregate errors.'))
    if overview:
        fig,axes=plt.subplots(len(overview),3,figsize=(16,2.5*len(overview)),squeeze=False)
        for row,(episode,t,target,expert,times,actual,ik)in enumerate(overview):
            for k,label in enumerate('XYZ'):
                ax=axes[row,k];ax.plot(t,target[:,k],color='black',lw=1,label='HF command target')
                ax.plot(t,expert[:,k],'--',color='.55',lw=1,label='Original measured EE')
                ax.plot(times,ik[:,k],color='#008080',lw=1,alpha=.65,label='DLS command FK')
                ax.plot(times,actual[:,k],color='#2463ba',lw=1.2,label='Live measured EE')
                ax.set_title(f'Episode {episode}: {label}',fontsize=10);ax.set_ylabel('m');ax.grid(alpha=.2)
                if row==len(overview)-1:ax.set_xlabel('Episode time (s)')
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,bbox_to_anchor=(.5,1.),fontsize=9)
        fig.suptitle('Five expert-action replays: target vs encoder-derived EE — no VLA',y=1.025,fontsize=13)
        fig.tight_layout(rect=[0,0,1,.98]);fig.savefig(out/'overview.png',dpi=160,bbox_inches='tight');fig.savefig(out/'overview.pdf',bbox_inches='tight');plt.close(fig)
    json_write(out/'summary.json',dict(episodes=summary,source_contract=str((batch/'replay_contract.json').resolve())))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','audit','run','plot'])
    p.add_argument('--batch',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,nargs='+',default=[480,500,520,540,560]);p.add_argument('--revision',default=REVISION)
    p.add_argument('--execute',action='store_true');p.add_argument('--governor',choices=['on','off'],default='on');p.add_argument('--pose-smoothing-ms',type=float,default=80.)
    p.add_argument('--resume-from',type=Path,help='Preserve completed replay episodes and continue at the first unfinished one')
    p.add_argument('--joint-acceleration-deg-s2',type=float,default=120.)
    p.add_argument('--joint-jerk-deg-s3',type=float,default=600.)
    p.add_argument('--max-episodes',type=int,help='Replay only the first N prepared episodes')
    p.add_argument('--no-return-home',action='store_true',help='Hold after selected episodes; no automatic final return')
    args=p.parse_args()
    if args.mode=='prepare':prepare(args.output,args.episodes,args.revision)
    else:
        if args.batch is None:p.error('--batch is required')
        if args.mode=='audit':audit(args.batch)
        elif args.mode=='plot':plot(args.batch,args.output)
        else:
            from .replay_control import run
            from .recording_lease import LeaseOwner
            lease=LeaseOwner(PROJECT/'scratch/real_robot/active-collection.json')
            try:
                run(args.batch,args.output,execute=args.execute,smoothing_ms=args.pose_smoothing_ms,governor=args.governor=='on',resume_from=args.resume_from,
                    acceleration_deg_s2=args.joint_acceleration_deg_s2,jerk_deg_s3=args.joint_jerk_deg_s3,max_episodes=args.max_episodes,return_home=not args.no_return_home)
            finally:lease.close()


if __name__=='__main__':main()
