"""Offline marker comparison: recorded inputs only; no robot or camera access."""
import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torchcodec.decoders import VideoDecoder
from safetensors.torch import load_model
from smoke_real_finetune import DynamicVLAPolicy,get_policy_cfg,AutoConfig,AutoTokenizer,PROJECT

sys.path.insert(0,str(PROJECT))
from real_robot.image_preprocessing import load_spec,transform
from real_robot.vx300s_kinematics import forward
from real_robot.policy_ik import PolicyDifferentialIK,rotation_error
from scipy.spatial.transform import Rotation
from utils.real_robot_contract import absolute_pose

INSTRUCTION='Pick up the moving green marker and place it in the box.'
KEYS=['front','wrist','state','environment','required','valid']


def records(path):return [json.loads(line) for line in path.open()]


def action_feedback_pairs(rows):
    # LAAS's send index can advance during one cycle. Pair by log order with
    # the following feedback row, which contains that command's measured q.
    pending=None;pairs=[]
    for row in rows:
        if row['event']=='action' and row['status']=='EXECUTING':pending=row
        elif row['event']=='feedback' and pending is not None:
            pairs.append((pending,row));pending=None
    return pairs


def prepare_demo(dataset,episode,stride):
    table=pq.read_table(dataset/f'data/chunk-000/episode_{episode:06d}.parquet')
    state=np.asarray(table['observation.state'].to_pylist())
    action=np.asarray(table['action'].to_pylist())
    environment=np.asarray(table['observation.environment_state'].to_pylist())
    quality=records(dataset/f'quality/episode_{episode:06d}.jsonl')
    usable=set(json.loads((dataset/'meta/quality_policy.json').read_text())['usable_frames'][str(episode)])
    indices=[i for i in range(2,len(state)-20,stride) if all(j in usable for j in [i-2,i-1,i])]
    payload={k:[] for k in KEYS}
    for role,key in [('front','opst_cam'),('wrist','wrist_cam')]:
        decoder=VideoDecoder(str(dataset/f'videos/chunk-000/observation.images.{key}/episode_{episode:06d}.mp4'),num_ffmpeg_threads=1)
        frames=decoder.get_frames_at(indices=sorted({j for i in indices for j in [i-2,i]})).data
        by_index=dict(zip(sorted({j for i in indices for j in [i-2,i]}),frames))
        payload[role]=np.stack([torch.stack([by_index[i-2],by_index[i]]).numpy() for i in indices])
    for i in indices:
        history=[i-2,i];payload['state'].append(state[history]);payload['environment'].append(environment[history])
        required=np.array([quality[j].get('velocity_required',True) for j in history])
        payload['required'].append(required);payload['valid'].append(np.ones(2,bool))
    for k in KEYS:payload[k]=np.asarray(payload[k])
    return payload,dict(indices=np.array(indices),time=np.array(indices)*.04,state=state[indices],
        recorded_command=action[indices],all_state=state,all_command=action,all_environment=environment,
        description=f'Recorded expert marker episode {episode}; static hold at frames 6–60, then rolling/grasping. This episode belongs to train, not a held-out performance estimate.')


def prepare_live(session,spec,stride,start_frame):
    rows=records(session/'controller/control.jsonl')
    feedback=[r for r in rows if r['event']=='feedback' and r['active']]
    capture=cv2.VideoCapture(str(session/'controller/trial-001-views.mp4'))
    if int(capture.get(cv2.CAP_PROP_FRAME_COUNT))!=len(feedback):raise ValueError('Cannot align live video to feedback rows')
    fusion=records(session/'perception/fusion.jsonl')
    fusion.sort(key=lambda r:r['views']['front']['timing']['host_arrival_monotonic_s'])
    stamps=np.array([r['views']['front']['timing']['host_arrival_monotonic_s'] for r in fusion])
    candidates=list(range(start_frame,len(feedback)-20,stride));selected=[];envs={};needed=set()
    for i in candidates:
        history=[i-2,i]
        if feedback[i]['index']-feedback[i-2]['index']!=2:continue
        valid=True
        for j in history:
            row=feedback[j];k=np.searchsorted(stamps,row['host_monotonic_s'],side='right')-1
            if k<0:valid=False;break
            track=fusion[k]['track'];velocity=fusion[k]['velocity_report']
            if (not track.get('accepted') or not track.get('ready') or velocity.get('status')!='estimated'
                    or row['host_monotonic_s']-stamps[k]>.2 or row['tracking_reason']):valid=False;break
            envs[j]=np.r_[track['position_m'],np.zeros(3),velocity['velocity_mps']]
        if valid:selected.append(i);needed.update(history)
    images={};frame_index=0
    while True:
        ok,image=capture.read()
        if not ok:break
        if frame_index in needed:
            images[frame_index]={role:transform(part,spec,role)[:,:,::-1].transpose(2,0,1).copy()
                for role,part in [('front',image[:,:640]),('wrist',image[:,640:])]}
        frame_index+=1
    capture.release();payload={k:[] for k in KEYS}
    for i in selected:
        history=[i-2,i]
        for role in ['front','wrist']:payload[role].append(np.stack([images[j][role] for j in history]))
        payload['state'].append([feedback[j]['tcp_pose6'] for j in history])
        payload['environment'].append([envs[j] for j in history]);payload['required'].append([True,True]);payload['valid'].append([True,True])
    payload={k:np.asarray(v) for k,v in payload.items()}
    origin=feedback[0]['host_monotonic_s'];pairs=action_feedback_pairs(rows);actions=[r for r,_ in pairs]
    geometry=np.array([forward(r['candidate_joints_rad'],[.16,0,0])[:3,3] for r in actions])
    before_governor=np.array([forward(r['ik_candidate_joints_rad'],[.16,0,0])[:3,3] for r in actions])
    metadata=dict(indices=np.array(selected),time=np.array([feedback[i]['host_monotonic_s']-origin for i in selected]),
        state=np.array([feedback[i]['tcp_pose6'] for i in selected]),
        all_time=np.array([r['host_monotonic_s']-origin for r in feedback]),all_state=np.array([r['tcp_pose6'] for r in feedback]),
        logged_action_time=np.array([f['host_monotonic_s']-origin for _,f in pairs]),
        logged_policy=np.array([r['policy_absolute_action'] for r in actions]),logged_ik=before_governor,logged_governed=geometry,
        object_position=payload['environment'][:,-1,:3],object_velocity=payload['environment'][:,-1,6:9],
        description='Recorded DOMV epoch-3 live trial, stationary-marker tail. Actual EE trajectory was produced by that rollout, not an expert demonstration. Video is a compressed copy of the live model views.')
    return payload,metadata


def load_checkpoint(path,bundle):
    cfg=get_policy_cfg(cfg_file=path/'config.json');cfg.enable_streaming=False;cfg.use_checkpoint_normalization=True
    with ExitStack() as stack:
        for cls in [AutoConfig,AutoTokenizer]:
            original=cls.from_pretrained
            def local(name,*args,_original=original,**kwargs):
                asset=bundle/'assets'/str(name)
                if asset.is_dir():name=str(asset);kwargs['local_files_only']=True
                return _original(name,*args,**kwargs)
            stack.enter_context(patch.object(cls,'from_pretrained',side_effect=local))
        model=DynamicVLAPolicy(cfg)
    load_model(model,str(path/'model.safetensors'),strict=True)
    return model.eval().cuda(),cfg


def predict(model,cfg,payload,batch_size):
    outputs=[];noise_rng=torch.Generator(device='cpu').manual_seed(20261109)
    for start in range(0,len(payload['state']),batch_size):
        stop=min(start+batch_size,len(payload['state']));batch={}
        for role,key in [('front','observation.images.opst_cam'),('wrist','observation.images.wrist_cam')]:
            batch[key]=torch.as_tensor(payload[role][start:stop],device='cuda').float()/255
        for name,key in [('state','observation.state'),('environment','observation.environment_state')]:
            batch[key]=torch.as_tensor(payload[name][start:stop],device='cuda',dtype=torch.float32)
        for name in ['required','valid']:batch['dynamic_object.velocity_'+name]=torch.as_tensor(payload[name][start:stop],device='cuda',dtype=torch.bool)
        batch['task']=[INSTRUCTION]*(stop-start)
        noise=torch.randn((stop-start,cfg.chunk_size,cfg.max_action_dim),generator=noise_rng).cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            result=model._get_action_chunk(model._prepare_batch(batch),noise=noise)
            if cfg.use_delta_action:result=absolute_pose(result,batch['observation.state'][:,-1:,:],cfg.euler_delta_convention)
        outputs.append(result.float().cpu().numpy())
    return np.concatenate(outputs)


def position_metrics(prediction,metadata,demo=False):
    delta=np.linalg.norm(prediction[:,0,:3]-metadata['state'][:,:3],axis=1)*1000
    horizon=np.linalg.norm(prediction[:,-1,:3]-prediction[:,0,:3],axis=1)*1000
    report=dict(first_target_distance_from_measured_mm_median=float(np.median(delta)),
        predicted_chunk_net_displacement_mm_median=float(np.median(horizon)))
    if demo:
        errors=np.linalg.norm(prediction[:,0,:3]-metadata['recorded_command'][:,:3],axis=1)*1000
        report['command_error_mm_median']=float(np.median(errors));report['command_error_mm_p95']=float(np.quantile(errors,.95))
        static=(metadata['indices']>=6)&(metadata['indices']<=60)
        report['static_prefix_command_error_mm_median']=float(np.median(errors[static]))
        for label,mask in [('static_wait',(metadata['indices']>=6)&(metadata['indices']<=60)),('moving',metadata['indices']>=75)]:
            ii=metadata['indices'][mask];truth=metadata['all_command'][ii+19,:3]-metadata['all_command'][ii,:3]
            pred=prediction[mask,-1,:3]-prediction[mask,0,:3]
            report[label]=dict(samples=int(mask.sum()),expert_chunk_net_displacement_mm_median=float(np.median(np.linalg.norm(truth,axis=1))*1000),
                predicted_chunk_net_displacement_mm_median=float(np.median(np.linalg.norm(pred,axis=1))*1000),
                endpoint_command_error_mm_median=float(np.median(np.linalg.norm(prediction[mask,-1,:3]-metadata['all_command'][ii+19,:3],axis=1))*1000))
    else:
        report['chunk_endpoint_distance_from_marker_surface_mm_median']=float(np.median(np.linalg.norm(prediction[:,-1,:3]-metadata['object_position'],axis=1))*1000)
    return report


def audit_live_controller(session,start_time):
    rows=records(session/'controller/control.jsonl');feedback=[r for r in rows if r['event']=='feedback' and r['active']]
    origin=min(r['host_monotonic_s'] for r in feedback);deltas=[];before=[];after=[];joint_following=[]
    for row,measured in action_feedback_pairs(rows):
        if measured['host_monotonic_s']-origin<start_time:continue
        diag=row['ik'];q=np.array(measured['actual_q_rad']);retained=q+np.array(diag['following_error_rad'])
        target=np.eye(4);target[:3,3]=row['smoothed_pose6'][:3];target[:3,:3]=Rotation.from_euler('xyz',row['smoothed_pose6'][3:6]).as_matrix()
        dt=diag['position_error_m']/diag['requested_tcp_speed_mps'] if diag['requested_tcp_speed_mps']>1e-9 else .04
        ik=PolicyDifferentialIK(q,retained,[.16,0,0],diag['joint_rate_ceilings_deg_s'],
            max_tcp_speed_mps=diag['tcp_speed_ceiling_mps'],max_angular_speed_deg_s=diag['angular_speed_ceiling_deg_s'],
            method=diag.get('method','uniform_dls'))
        replay=ik.step(target,q,dt);logged=np.array(row['ik_candidate_joints_rad'])
        deltas.append(float(np.max(abs(replay-logged))))
        before.append(np.linalg.norm(target[:3,3]-forward(q,[.16,0,0])[:3,3])*1000)
        after.append(np.linalg.norm(target[:3,3]-forward(logged,[.16,0,0])[:3,3])*1000)
        joint_following.append(np.linalg.norm(forward(row['candidate_joints_rad'],[.16,0,0])[:3,3]-forward(q,[.16,0,0])[:3,3])*1000)
    return dict(commands=len(deltas),maximum_replayed_ik_joint_discrepancy_rad=max(deltas),
        position_error_before_dls_mm_median=float(np.median(before)),position_error_after_dls_mm_median=float(np.median(after)),
        governed_command_to_measured_position_mm_median=float(np.median(joint_following)),
        note='Numerical replay checks DIK bookkeeping, not global physical stability. Finite-step/following bounds and trajectory shaping can leave tracking lag.')


def plots(output,predictions,metadata,models):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    colors=['#2676b8','#d97919','#21956a']
    def with_gaps(meta,values):
        steps=np.diff(meta['indices']);expected=np.min(steps) if len(steps) else 1
        breaks=np.flatnonzero(steps>expected)+1
        return np.insert(meta['time'],breaks,np.nan),np.insert(values,breaks,np.nan)
    for source,title in [('demo','Recorded marker demonstration — identical inputs'),('live','Stationary-marker live recording — identical inputs; no new robot motion')]:
        meta=metadata[source];fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True)
        for axis,label in enumerate('XYZ'):
            ax=axes[axis]
            times=np.arange(len(meta['all_state']))*.04 if source=='demo' else meta['all_time']
            ax.plot(times,meta['all_state'][:,axis]*1000,color='#18212b',lw=2.3,label='Measured EE feedback')
            if source=='demo':
                ax.plot(times,meta['all_command'][:,axis]*1000,color='#777777',lw=1.7,ls='--',label='Recorded expert EE command')
                ax.axvspan(.24,2.4,color='#ddebdc',alpha=.65,label='Static marker + expert waiting' if axis==0 else None)
            for k,(name,_) in enumerate(models):
                xx,yy=with_gaps(meta,predictions[name][source][:,0,axis]*1000)
                ax.plot(xx,yy,color=colors[k],lw=1.3,marker='.',ms=3,label=name)
            if source=='live':
                mask=meta['logged_action_time']>=meta['time'][0]
                ax.plot(meta['logged_action_time'][mask],meta['logged_policy'][mask,axis]*1000,color='#a552b8',alpha=.6,lw=.9,label='Logged live VLA targets (LAAS-selected, before IK)')
                ax.plot(meta['logged_action_time'][mask],meta['logged_governed'][mask,axis]*1000,color='#777777',ls=':',lw=1.2,label='FK of smoothed joint commands')
                ax.plot(meta['time'],meta['object_position'][:,axis]*1000,color='#bb3f48',ls='--',lw=1.5,label='Tracked marker surface position')
            ax.set_ylabel(f'{label} position (mm)');ax.grid(alpha=.2)
            ax.set_xlim(meta['time'][0],meta['time'][-1])
        axes[0].legend(loc='upper center',bbox_to_anchor=(.5,1.47),ncol=2,frameon=False)
        axes[-1].set_xlabel('Recording time (s)');fig.suptitle(title,y=.995,fontsize=14)
        fig.text(.5,.012,'Absolute targets at chunk offset 0; gaps follow tracking-quality exclusions. Replay uses recorded states, not a simulated closed loop.',ha='center',fontsize=9,color='#4c5967')
        fig.tight_layout(rect=(0,.03,1,.94));fig.savefig(output/f'{source}_positions.png',dpi=180);fig.savefig(output/f'{source}_positions.pdf');plt.close(fig)
    # A complete predicted chunk at the same frozen stationary-marker observation.
    meta=metadata['live'];sample=len(meta['indices'])//2;t=meta['time'][sample];h=np.arange(20)*.04
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
    for axis,label in enumerate('XYZ'):
        for k,(name,_) in enumerate(models):axes[axis].plot(h,predictions[name]['live'][sample,:,axis]*1000,color=colors[k],lw=2,label=name)
        actual=np.interp(t+h,meta['all_time'],meta['all_state'][:,axis])*1000
        axes[axis].plot(h,actual,color='#18212b',lw=2,label='Measured EE during original live rollout')
        axes[axis].axhline(meta['object_position'][sample,axis]*1000,color='#bb3f48',ls='--',label='Tracked marker surface position')
        axes[axis].set_ylabel(f'{label} (mm)');axes[axis].grid(alpha=.2)
    axes[0].legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,1.55),frameon=False)
    axes[-1].set_xlabel('Prediction horizon (s)');fig.suptitle(f'Frozen stationary-marker input at recording t={t:.2f}s',y=.995)
    fig.text(.5,.012,'Actual trace comes from the original DOMV rollout; it is not an expert target or a rollout of the pretrained models.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.03,1,.94));fig.savefig(output/'frozen_chunk.png',dpi=180);fig.savefig(output/'frozen_chunk.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--plot-only',action='store_true',help='Render cached predictions without running models again')
    p.add_argument('--stride',type=int,default=3);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    run=PROJECT/'scratch/real_robot/finetuning-runs/20261109T014917Z'
    dataset=Path(json.loads((run/'launch_contract.json').read_text())['local_verified_dataset'])
    bundle=PROJECT/'scratch/real_robot/finetuning-checkpoints'
    session=PROJECT/'scratch/real_robot/evaluations/marker-domv-20261109T233710Z'
    models=[('Pretrained DOM (original)',PROJECT/'scratch/real_robot/static-marker-eval/checkpoints/original-dom'),
        ('Pretrained DOMV (initialization)',bundle/'checkpoints/large-audited-197198/domv/epoch-006-u000381312'),
        ('Fine-tuned DOMV epoch 3',run/'domv/epoch-003')]
    if args.plot_only:
        metadata={source:dict(np.load(args.output/f'{source}_reference.npz')) for source in ['demo','live']}
        rows=records(session/'controller/control.jsonl');pairs=action_feedback_pairs(rows)
        origin=next(r['host_monotonic_s'] for r in rows if r['event']=='feedback' and r['active'])
        metadata['live']['logged_action_time']=np.array([r['host_monotonic_s']-origin for _,r in pairs])
        metadata['live']['logged_policy']=np.array([r['policy_absolute_action'] for r,_ in pairs])
        metadata['live']['logged_governed']=np.array([forward(r['candidate_joints_rad'],[.16,0,0])[:3,3] for r,_ in pairs])
        metadata['live']['logged_ik']=np.array([forward(r['ik_candidate_joints_rad'],[.16,0,0])[:3,3] for r,_ in pairs])
        np.savez(args.output/'live_reference.npz',**metadata['live'])
        predictions={name:{source:np.load(args.output/f'{source}_{key}_predictions.npy') for source in ['demo','live']}
            for (name,_),key in zip(models,['original_dom','pretrained_domv','finetuned_domv'])}
        report=json.loads((args.output/'report.json').read_text())
        report['controller_audit']=audit_live_controller(session,metadata['live']['time'][0])
        for name in predictions:
            for source in ['demo','live']:report['models'][name]['metrics'][source]=position_metrics(predictions[name][source],metadata[source],source=='demo')
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        plots(args.output,predictions,metadata,models)
        return
    spec=load_spec(dataset/'meta/image_preprocessing.json');payloads={};metadata={}
    payloads['demo'],metadata['demo']=prepare_demo(dataset,551,args.stride)
    payloads['live'],metadata['live']=prepare_live(session,spec,args.stride,400)
    report=dict(hardware_accessed=False,weight_updates=False,instruction=INSTRUCTION,dataset=str(dataset),demo_episode=551,
        live_session=str(session),paired_noise_seed=20261109,observation_offsets=[-2,0],action_horizon=20,dt_s=.04,
        normalization='Each checkpoint retains its own serialized normalization; no real-data statistics are substituted into pretrained models.',
        limitations=['Teacher-forced replay does not measure closed-loop task success.',
            'Original DOM and DOMV differ in training/velocity conditioning; untouched DOMV initialization controls for fine-tuning.',
            'Demo 551 is a training episode and contains an initial static waiting period followed by marker motion.',
            'Live samples use lossy archived view video, while original live targets were generated from raw camera images.'],sources={},models={})
    for source in payloads:
        np.savez_compressed(args.output/f'{source}_inputs.npz',**payloads[source])
        np.savez(args.output/f'{source}_reference.npz',**{k:v for k,v in metadata[source].items() if isinstance(v,np.ndarray)})
        report['sources'][source]=dict(samples=len(metadata[source]['time']),description=metadata[source]['description'])
    pos=metadata['live']['object_position'];report['sources']['live']['tracked_marker_span_mm']=(np.ptp(pos,axis=0)*1000).tolist()
    report['controller_audit']=audit_live_controller(session,metadata['live']['time'][0])
    predictions={}
    for name,path in models:
        print('Loading '+name,flush=True);model,cfg=load_checkpoint(path,bundle)
        predictions[name]={};report['models'][name]=dict(checkpoint=str(path),strict_weights=True,
            config_sha256=hashlib.sha256((path/'config.json').read_bytes()).hexdigest(),
            euler_delta_convention=cfg.euler_delta_convention,condition_on_object_velocity=cfg.condition_on_object_velocity,metrics={})
        for filename in ['source.json','provenance.json']:
            if (path/filename).is_file():
                report['models'][name]['source_provenance']=json.loads((path/filename).read_text());break
        for source,payload in payloads.items():
            prediction=predict(model,cfg,payload,args.batch_size);predictions[name][source]=prediction
            report['models'][name]['metrics'][source]=position_metrics(prediction,metadata[source],source=='demo')
            key=('original_dom','pretrained_domv','finetuned_domv')[len(predictions)-1]
            np.save(args.output/f'{source}_{key}_predictions.npy',prediction)
            print(name+' '+source+' '+str(report['models'][name]['metrics'][source]),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    plots(args.output,predictions,metadata,models)
    print('Saved plots and report: '+str(args.output),flush=True)


if __name__=='__main__':main()
