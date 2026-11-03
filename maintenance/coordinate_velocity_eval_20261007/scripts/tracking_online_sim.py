#!/usr/bin/env python3
"""Isaac Sim live tracking with synchronous RPC to the separate checkpoint worker."""
import argparse
import json
import logging
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from scipy.spatial.transform import Rotation
import zmq
from tracking_online_common import PosePlan, pose_command, summarize_rollout, write_json, sha256, TrackingTermination
from tracking_variants import VARIANT_LABELS


def rpc(socket, message):
    socket.send_pyobj(message)
    reply=socket.recv_pyobj()
    if not reply.get('ok'): raise RuntimeError(reply.get('error','Policy worker failed'))
    return reply.get('result')


def run(args):
    sys.path.insert(0,str(args.collector))
    from dom_tracking.cli import launch_isaac
    from dom_tracking.controller import Expert, ModeSelector
    from dom_tracking.math3d import dom_euler, mode_quaternion, angle_error
    from dom_tracking.trajectory import Trajectory
    from dom_tracking.storage import VideoWriter, jsonable
    manifest=json.loads(args.manifest.read_text())
    cfg=manifest['config']; layout=manifest['layout']
    app,software=launch_isaac(cfg,False,True)
    completed=False
    backend=None
    try:
        from dom_tracking.isaac_backend import IsaacBackend, dependency_check
        dependencies={name:dependency_check(layout[name]) for name in ('scene_usd','object_usd')}
        backend=IsaacBackend(cfg,layout)
        context=zmq.Context();socket=context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO,600000);socket.setsockopt(zmq.SNDTIMEO,600000);socket.setsockopt(zmq.LINGER,0)
        socket.connect(args.endpoint)
        write_json(args.output/'runtime.json',{'software':software,'dependencies':dependencies,
            'evaluation':'closed-loop, synchronous simulation time; simulation pauses during inference',
            'source_commit':manifest['collector_commit'],'pose_adapter':'linear XYZ and SLERP; +0.2s action horizon; replanning every 20 observations',
            'velocity_source':'simulator ground-truth current object velocity in robot-base coordinates',
            'gripper':'held open, matching tracking dataset', 'inference_clock':'latency logged; no simulated latency injection'})
        all_results=[]
        physics_hz=cfg['collection']['physics_hz'];fps=cfg['collection']['fps'];stride=physics_hz//fps
        dt=1/physics_hz
        for variant in manifest.get('models', {'baseline':None,'residual':None}):
            for speed in manifest['speed_multipliers']:
                for item in manifest['episodes']:
                    ep=item['episode']; label='iid' if speed==1 else f'ood_{speed:g}x'
                    dest=args.output/variant/label/f'episode_{ep:06d}'
                    dest.mkdir(parents=True,exist_ok=False)
                    rows=[];writers={};frames=[];latencies=[];error=None;finished=False;policy_termination=False
                    metadata=rpc(socket,{'cmd':'reset','variant':variant,'seed':42+10000*ep})
                    path=Trajectory(**{k:np.asarray(v,dtype=float) for k,v in item['trajectory'].items()})
                    path.omega=path.omega*speed
                    try:
                        backend.reset(path)
                        expert=Expert(cfg['controller'],dt)
                        for step in range(round(cfg['collection']['warmup_seconds']*physics_hz)):
                            state=backend.state();target=backend.target();target['velocity']=np.zeros(3)
                            backend.apply(expert.command(state,target,backend.jacobian(),backend.limits,backend.velocity_limits))
                            backend.step(0,render=step%stride==0)
                        # Both variants start from the same expert-settled pose; no expert assistance afterwards.
                        backend.capture(-1/fps)
                        initial=jsonable(backend.state())
                        write_json(dest/'initial_state.json',initial)
                        for name in ('opst_cam','wrist_cam','side_cam'):
                            writers[name]=VideoWriter(dest/f'{name}.mp4',fps,480,360,'libx264')
                        selector=ModeSelector(cfg['controller']['enter_speed'],cfg['controller']['exit_speed'])
                        previous_mode=None;switch_time=0.;plan=None
                        raw=(dest/'steps.jsonl').open('w')
                        try:
                            for step in range(round(args.seconds*physics_hz)):
                                now=step*dt; state=backend.state();target=backend.target()
                                mode=selector.update(target['velocity'][1])
                                if mode!=previous_mode: switch_time=now;previous_mode=mode
                                desired_q=mode_quaternion(mode,cfg['controller']['down_quaternion'])
                                current_pose=np.r_[state['ee_position'],dom_euler(state['ee_quaternion'])]
                                if step%stride==0:
                                    frame=step//stride
                                    images,cameras=backend.capture(now)
                                    response=rpc(socket,{'cmd':'observe','frame':frame,'time':now,'state':current_pose.astype('float32'),
                                        'environment_state':np.r_[backend.target_p,np.zeros(3),backend.target_v].astype('float32'),
                                        'images':{k:images[k] for k in ('opst_cam','wrist_cam')}})
                                    if response['replanned']:
                                        plan=PosePlan(now,current_pose,response['chunk_absolute'])
                                        latencies.append(response['inference_seconds'])
                                        write_json(dest/f'chunk_{frame:04d}.json',jsonable(response))
                                    frames.append({'time':now,'frame':frame,'mode':mode,'cameras':cameras})
                                    from PIL import Image, ImageDraw
                                    for name,image in images.items():
                                        canvas=Image.fromarray(image)
                                        draw=ImageDraw.Draw(canvas)
                                        draw.rectangle((0,0,480,34),fill='black')
                                        draw.text((5,3),f'{VARIANT_LABELS[variant]} | {label} | ep {ep} | t={now:.2f}s',fill='white')
                                        draw.text((5,18),f'object speed {np.linalg.norm(backend.target_v):.3f} m/s | LIVE POLICY',fill='white')
                                        writers[name].add(np.asarray(canvas,dtype='uint8'))
                                command=pose_command(state,plan.sample(now+dt),backend.jacobian(),backend.limits,
                                                     backend.velocity_limits,cfg['controller'],dt)
                                q=state['joint_position']
                                joint_violation=bool(np.any(q<backend.limits[:,0]) or np.any(q>backend.limits[:,1]))
                                workspace_violation=bool(np.any(state['ee_position']<target['workspace_lower']) or np.any(state['ee_position']>target['workspace_upper']))
                                row={'time':now,'hover_error_m':float(np.linalg.norm(state['ee_position']-target['position'])),
                                    'orientation_error_deg':float(np.rad2deg(np.linalg.norm(angle_error(desired_q,state['ee_quaternion'])))),
                                    'settled':now-switch_time>=cfg['quality']['transition_seconds'],'mode':mode,
                                    'object_speed_mps':float(np.linalg.norm(backend.target_v)),
                                    'object_position':backend.target_p.tolist(),'object_velocity':backend.target_v.tolist(),
                                    'ee_pose':current_pose.tolist(),'hover_target':target['position'].tolist(),
                                    'contact_force_n':backend.contact_force(),'joint_limit_violation':joint_violation,
                                    'workspace_violation':workspace_violation,**command}
                                rows.append(row);raw.write(json.dumps(row,allow_nan=False)+'\n')
                                backend.apply(command);backend.step(now+dt,render=step%stride==0)
                            finished=True
                        finally:
                            raw.close()
                    except Exception as exc:
                        error=f'{type(exc).__name__}: {exc}'
                        policy_termination=isinstance(exc,TrackingTermination)
                        logging.exception('Rollout failed: %s %s episode %d',variant,label,ep)
                    finally:
                        for writer in writers.values(): writer.close()
                        metrics=summarize_rollout(rows,cfg['quality'],manifest['train_max_speed_mps'],finished)
                        metrics['replans']=len(latencies)
                        metrics['mean_inference_seconds']=float(np.mean(latencies)) if latencies else None
                        visibility={name:float(np.mean([f['cameras'][name]['target_in_frame'] for f in frames])) for name in ('opst_cam','wrist_cam','side_cam')} if frames else {}
                        metrics['visible_fraction']=visibility
                        if visibility and min(visibility.values())<cfg['quality']['min_visible_fraction']: metrics['success']=False
                        result={'model':variant,'scenario':label,'speed_multiplier':speed,'episode':ep,
                            'checkpoint':metadata,'trajectory':path.serialize(),'metrics':metrics,'error':error,
                            'failure_type':'policy_termination' if policy_termination else ('infrastructure' if error else None),
                            'requested_seconds':args.seconds,'recorded_seconds':len(rows)*dt,
                            'video_frames':{name:w.count for name,w in writers.items()},'videos':{name:str(dest/f'{name}.mp4') for name in writers}}
                        write_json(dest/'metrics.json',result);write_json(dest/'frames.json',frames)
                        all_results.append(result);write_json(args.output/'rollouts.json',all_results)
                        logging.info('ROLLOUT %s %s ep=%d: %s',variant,label,ep,metrics)
                    if error and not policy_termination:
                        # Infrastructure/controller failures must not silently produce a partial successful run.
                        raise RuntimeError(f'Rollout failed; partial videos/metrics saved in {dest}: {error}')
        rpc(socket,{'cmd':'stop'});socket.close();context.term()
        completed=True
        write_json(args.output/'completion.json',{'completed':True,'rollouts':len(all_results),
            'policy_terminations':sum(r.get('failure_type')=='policy_termination' for r in all_results)})
    except BaseException:
        traceback.print_exc()
        app._app.post_quit(1)
        raise
    finally:
        if backend is not None: backend.close()
        app.close()
    return completed


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collector',required=True,type=Path);p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--endpoint',required=True)
    p.add_argument('--seconds',type=float,default=10.)
    args=p.parse_args()
    if args.seconds<=0:p.error('--seconds must be positive')
    run(args)
