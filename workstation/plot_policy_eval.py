"""Plot recorded policy targets, measured EE motion and tracked-object motion."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from real_robot.vx300s_kinematics import forward


def read_rows(path):return [json.loads(line) for line in path.open()] if path.exists() else []


def paired_actions(rows):
    pending=None;pairs=[]
    for row in rows:
        if row['event']=='action':pending=row if row['status']=='EXECUTING' else None
        elif row['event']=='feedback' and pending is not None:pairs.append((pending,row));pending=None
    return pairs


def plot_session(session):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    output=session/'plots';output.mkdir(exist_ok=True)
    rows=read_rows(session/'controller/control.jsonl');pairs=paired_actions(rows)
    feedback=[r for r in rows if r['event']=='feedback']
    inference=read_rows(session/'exchange/inference.jsonl');fusion=read_rows(session/'perception/fusion.jsonl')
    manifest=json.loads((session/'controller/manifest.json').read_text());tcp=manifest['tcp_offset_m']
    summary=dict(session=str(session),trials=[],hardware_accessed=False)
    for trial_path in sorted((session/'controller').glob('trial-*.json')):
        trial=json.loads(trial_path.read_text());start=trial['start_monotonic_s'];end=trial['end_monotonic_s']
        measured=[r for r in feedback if r['active'] and start<=r['host_monotonic_s']<=end]
        if not measured:continue
        tt=np.array([r['host_monotonic_s']-start for r in measured]);state=np.array([r['tcp_pose6'] for r in measured])
        commands=[(a,f) for a,f in pairs if start<=f['host_monotonic_s']<=end]
        ct=np.array([f['host_monotonic_s']-start for _,f in commands])
        raw=np.array([a['policy_absolute_action'][:3] for a,_ in commands]).reshape(-1,3)
        goals=np.array([forward(a['candidate_joints_rad'],tcp)[:3,3] for a,_ in commands]).reshape(-1,3)
        ik_goals=np.array([forward(a.get('ik_candidate_joints_rad',a['candidate_joints_rad']),tcp)[:3,3] for a,_ in commands]).reshape(-1,3)
        actual=np.array([f['tcp_pose6'][:3] for _,f in commands]).reshape(-1,3)
        chunks=[r for r in inference if r.get('trial')==trial['trial'] and 'actions_absolute' in r
            and start<=r['created_monotonic_s']<=end]
        objects=[]
        for r in fusion:
            stamp=((r.get('views') or {}).get('front',{}).get('timing') or {}).get('host_arrival_monotonic_s')
            track=r.get('track') or {};velocity=r.get('velocity_report') or {}
            if stamp is not None and start<=stamp<=end and track.get('accepted') and track.get('ready'):
                objects.append((stamp-start,track['position_m'],velocity.get('velocity_mps') if velocity.get('status')=='estimated' else None))
        fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True)
        chunk_subset=[chunks[i] for i in np.linspace(0,len(chunks)-1,min(30,len(chunks)),dtype=int)] if chunks else []
        for axis,label in enumerate('XYZ'):
            ax=axes[axis]
            for j,chunk in enumerate(chunk_subset):
                values=np.asarray(chunk['actions_absolute'])
                times=(chunk['index']+np.arange(len(values)))*.04
                ax.plot(times,values[:,axis]*1000,color='#71bba0',alpha=.25,lw=.8,label='Full predicted chunks (including unexecuted steps)' if j==0 else None)
            ax.plot(tt,state[:,axis]*1000,color='#142230',lw=2,label='Measured EE feedback')
            if len(commands):
                ax.plot(ct,raw[:,axis]*1000,color='#a348a6',lw=1,alpha=.85,label='Raw VLA targets selected by LAAS')
                ax.plot(ct,ik_goals[:,axis]*1000,color='#d17b20',lw=1,ls='--',label='FK of bounded DIK commands, before joint smoothing')
                ax.plot(ct,goals[:,axis]*1000,color='#2676b8',lw=1.3,ls=':',label='FK of smoothed joint commands')
            if objects:ax.plot([r[0] for r in objects],np.array([r[1] for r in objects])[:,axis]*1000,color='#b94248',ls='--',lw=1.3,label='Tracked object surface position')
            ax.set_ylabel(label+' position (mm)');ax.grid(alpha=.2);ax.set_xlim(0,end-start)
        axes[0].legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,1.42),frameon=False)
        axes[-1].set_xlabel('Trial time (s)')
        fig.suptitle(f'Trial {trial["trial"]}: {trial["outcome"]} — {session.name}',fontsize=13,y=.995)
        fig.text(.5,.012,'Recorded predictions and feedback. Object surface estimates are not exact grasp targets; future chunk steps may never be executed.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.03,1,.94));name=f'trial-{trial["trial"]:03d}-ee'
        fig.savefig(output/(name+'.png'),dpi=170);fig.savefig(output/(name+'.pdf'))
        metrics=dict(trial=trial['trial'],outcome=trial['outcome'],reason=trial['reason'],duration_s=end-start,
            common_start_verified=bool(trial.get('common_start_verified',False)),
            start_pose_reference=trial.get('start_pose_reference'),
            measured_samples=len(measured),executed_commands=len(commands),predicted_chunks=len(chunks),
            measured_ee_span_mm=(np.ptp(state[:,:3],axis=0)*1000).tolist(),plot=name+'.png')
        if len(commands):
            metrics['ik_methods']=sorted({a['ik'].get('method','uniform_dls') for a,_ in commands})
            metrics['command_scale_scopes']=sorted({a['ik'].get('command_scale_scope','global_joint_and_cartesian_scaling') for a,_ in commands})
            metrics['raw_target_to_measured_mm_median']=float(np.median(np.linalg.norm(raw-actual,axis=1))*1000)
            metrics['governed_command_to_measured_mm_median']=float(np.median(np.linalg.norm(goals-actual,axis=1))*1000)
        if chunks:
            disp=[np.linalg.norm(np.array(r['actions_absolute'])[-1,:3]-np.array(r['actions_absolute'])[0,:3])*1000 for r in chunks]
            metrics['predicted_chunk_net_displacement_mm_median']=float(np.median(disp))
            archives=[session/'exchange'/r['input_archive'] for r in chunks if r.get('input_archive')]
            metrics['lossless_input_archives']=sum(path.is_file() for path in archives)
            metrics['input_archives_missing']=sum(not path.is_file() for path in archives)
        if objects:metrics['tracked_object_span_mm']=(np.ptp(np.array([r[1] for r in objects]),axis=0)*1000).tolist()
        speeds=[np.linalg.norm(r[2]) for r in objects if r[2] is not None]
        if speeds:metrics['tracked_speed_mps_p95']=float(np.quantile(speeds,.95))
        if chunks:
            phases={}
            for phase,predicate in [('moving',lambda v:v>.05),('stationary',lambda v:v<.02)]:
                subset=[r for r in chunks if 'observed_environment' in r and predicate(np.linalg.norm(r['observed_environment'][6:9]))]
                if not subset:continue
                plans=np.asarray([r['actions_absolute'] for r in subset]);anchors=np.asarray([r['observed_state'][:3] for r in subset])
                phase_metrics=dict(chunks=len(subset),first_anchor_s=min(r['index']*.04 for r in subset),last_anchor_s=max(r['index']*.04 for r in subset),
                    predicted_chunk_net_displacement_mm_median=float(np.median(np.linalg.norm(plans[:,-1,:3]-plans[:,0,:3],axis=1))*1000),
                    endpoint_distance_from_current_ee_mm_median=float(np.median(np.linalg.norm(plans[:,-1,:3]-anchors,axis=1))*1000))
                if objects:
                    object_times=np.array([r[0] for r in objects]);object_xyz=np.array([r[1] for r in objects]);future_times=np.array([r['index']*.04+.76 for r in subset])
                    supported=(future_times>=object_times[0])&(future_times<=object_times[-1])
                    if supported.any():
                        future=np.column_stack([np.interp(future_times[supported],object_times,object_xyz[:,k]) for k in range(3)])
                        error=np.linalg.norm(plans[supported,-1,:3]-future,axis=1)
                        phase_metrics['predicted_endpoint_to_future_object_surface_mm_median']=float(np.median(error)*1000)
                phases[phase]=phase_metrics
            metrics['motion_phases']=phases
            metrics['phase_thresholds_mps']=dict(moving_above=.05,stationary_below=.02)
            if 'moving' in phases:
                moving=phases['moving'];low=max(0,moving['first_anchor_s']-.5);high=min(end-start,moving['last_anchor_s']+.8)
                mask=(ct>=moving['first_anchor_s'])&(ct<=moving['last_anchor_s'])
                if mask.any():
                    smoothed=np.array([a['smoothed_pose6'][:3] for a,_ in commands])
                    moving['controller_command_samples']=int(mask.sum())
                    moving['median_dik_command_scale']=float(np.median([a['ik']['command_scale'] for (a,_),use in zip(commands,mask) if use]))
                    moving['smoothed_target_to_bounded_ik_mm_p95']=float(np.quantile(np.linalg.norm(smoothed[mask]-ik_goals[mask],axis=1)*1000,.95))
                    moving['ik_to_joint_governor_mm_p95']=float(np.quantile(np.linalg.norm(ik_goals[mask]-goals[mask],axis=1)*1000,.95))
                for ax in axes:
                    ax.set_xlim(low,high);ax.axvspan(moving['first_anchor_s'],moving['last_anchor_s'],color='#d4e8c8',alpha=.15)
                fig.suptitle(f'Trial {trial["trial"]}: moving-marker interval — {session.name}',fontsize=13,y=.995)
                fig.savefig(output/(name+'-moving.png'),dpi=170);fig.savefig(output/(name+'-moving.pdf'))
                metrics['moving_interval_plot']=name+'-moving.png'
        plt.close(fig)
        summary['trials'].append(metrics)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session',type=Path,required=True)
    args=parser.parse_args();plot_session(args.session.resolve())


if __name__=='__main__':main()
