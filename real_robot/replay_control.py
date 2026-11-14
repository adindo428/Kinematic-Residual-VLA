"""Paused, device-only execution of recorded expert Cartesian actions."""
import json
import os
from pathlib import Path
import subprocess
import shutil
import time

import cv2
import numpy as np
from .aloha_arm import TapeArm
from .collision import GeometryGuard
from .dom_dataset import json_write,pose6,sha256
from .policy_ik import PolicyDifferentialIK
from .policy_reset import StartPoseGate,StartPoseReturn,check_joint_path
from .policy_stream import gripper_raw
from .replay_actions import ReplayPipeline,PROJECT,replay_frame
from .tracking_config import load_config
from .vx300s_kinematics import forward,LOWER,UPPER


def render_plots(batch,output):
    """Collection Python has motor drivers; plotting lives in the FT environment."""
    try:
        with (output/'plots.log').open('w')as log:
            result=subprocess.run([str(PROJECT/'workstation/finetune-python.sh'),'-m','real_robot.replay_actions','plot',
                '--batch',str(batch),'--output',str(output)],cwd=PROJECT,
                env=dict(os.environ,MPLCONFIGDIR='/tmp/krvla-mpl'),stdout=log,stderr=subprocess.STDOUT,timeout=60)
        print('Replay plots: '+str(output/'plots')if result.returncode==0 else 'Replay plotting failed; see '+str(output/'plots.log'),flush=True)
    except (OSError,subprocess.TimeoutExpired)as exc:print('Replay plotting failed: '+str(exc),flush=True)


def run(batch,output,*,execute=False,smoothing_ms=80.,governor=True,resume_from=None,
        acceleration_deg_s2=120.,jerk_deg_s3=600.,max_episodes=None,return_home=True):
    batch,output=Path(batch).resolve(),Path(output).resolve()
    contract=json.loads((batch/'replay_contract.json').read_text());period=1/contract['fps']
    if max_episodes is not None:
        if not isinstance(max_episodes,int) or max_episodes<=0:raise ValueError('max_episodes must be positive')
        contract['episodes']=contract['episodes'][:max_episodes]
    if not contract['episodes']:raise ValueError('No episodes selected')
    if not np.isfinite([acceleration_deg_s2,jerk_deg_s3]).all() or min(acceleration_deg_s2,jerk_deg_s3)<=0:
        raise ValueError('Acceleration/jerk bounds must be finite and positive')
    if contract['fps']!=25:raise ValueError('Replay requires the dataset 25Hz action clock')
    if smoothing_ms!=0 and not 40<=smoothing_ms<=200:raise ValueError('Smoothing must be 0 or 40–200 ms')
    for item in contract['episodes']:
        if sha256(batch/item['file'])!=item['archive_sha256']:raise ValueError('Prepared replay archive changed')
    if not execute:raise ValueError('Physical replay requires --execute; use audit for a device-free run')
    output.mkdir(parents=True,exist_ok=False);settings=load_config(PROJECT/'real_robot/config_policy_eval.yaml',manipulation=True)['servo']
    calibration=json.loads(Path('/home/ailab/lerobot/.cache/calibration/aloha_default/right_follower.json').read_text())
    closed,opened=calibration['start_pos'][-1],calibration['end_pos'][-1]
    arm=None;servo=None;log=None;episode_log=None;results=[];error=None;item=None;data=None
    previous_manifest=None
    if resume_from is not None:
        resume_from=Path(resume_from).resolve();previous_manifest=json.loads((resume_from/'manifest.json').read_text())
        if (previous_manifest['source_contract_sha256']!=sha256(batch/'replay_contract.json')
                or previous_manifest['pose_smoothing_ms']!=smoothing_ms or previous_manifest['joint_governor']!=governor
                or previous_manifest.get('selected_episodes',[r['episode']for r in contract['episodes']])!=[r['episode']for r in contract['episodes']]
                or (governor and (previous_manifest['joint_acceleration_deg_s2']!=acceleration_deg_s2 or previous_manifest['joint_jerk_deg_s3']!=jerk_deg_s3))):
            raise ValueError('Cannot resume with changed replay inputs or controller mode')
        prior=json.loads((resume_from/'summary.json').read_text())
        results=[r for r in prior['episodes']if r['full_episode_completed']]
        for record in results:
            for suffix in ('.json','.jsonl'):
                shutil.copy2(resume_from/f'episode_{record["episode"]:06d}{suffix}',output/f'episode_{record["episode"]:06d}{suffix}')
    stage='waiting_start';index=0;plan=None;gate=None;pipeline=None;origin=None;last_frame=-1;samples=0;skipped=0;tail_started=None
    if results:
        done={r['episode']for r in results}
        while index<len(contract['episodes']) and contract['episodes'][index]['episode']in done:index+=1
        if index==len(contract['episodes']):raise ValueError('All requested episodes are already complete')
    status='PAUSED: SPACE moves to episode start; Q holds/quits';active_result=None
    def load_episode():
        nonlocal item,data,gate
        item=contract['episodes'][index]
        with np.load(batch/item['file'])as archive:data={k:archive[k]for k in archive.files}
        gate=StartPoseGate(data['expert_measured_joints'][0],int(data['expert_measured_gripper_raw'][0]),item['tcp_offset_m'],
            command_joints=data['expert_command_joints'][0],command_gripper=int(data['expert_command_gripper_raw'][0]),
            position_tolerance_m=.010,joint_tolerance_deg=2.,rotation_tolerance_deg=3.)
        if not min(closed,opened)<=gate.command_gripper<=max(closed,opened):raise ValueError('Recorded gripper start command is outside current calibration')
    def begin_return(target_gate,now):
        return StartPoseReturn(servo.goal,target_gate,rates,None,None,geometry,None,None,now)
    def finish_episode(outcome,reason,now):
        nonlocal active_result,episode_log,pipeline
        if active_result is not None:
            active_result.update(outcome=outcome,reason=reason,end_monotonic_s=now,samples=samples,skipped_frames=skipped,
                full_episode_completed=outcome=='complete' and skipped==0 and samples==len(data['actions']))
            json_write(output/f'episode_{item["episode"]:06d}.json',active_result);results.append(active_result);active_result=None
        if episode_log is not None:episode_log.close();episode_log=None
        if pipeline is not None:pipeline.hold()
    def complete_replay(now):
        nonlocal gate,plan,stage,status
        finish_episode('complete','all recorded action times consumed; final governed target settled' if governor else 'all recorded action times consumed',now)
        if index==len(contract['episodes'])-1:
            if not return_home:stage='completed_hold';return True
            gate=home;gate.invalidate();plan=begin_return(gate,now);stage='returning_home';status='Returning to session home after selected episodes'
        else:stage='between';status='Episode complete. SPACE prepares the next recorded start'
        return False
    try:
        arm=TapeArm();arm.check_health();actual,_,raw=arm.telemetry();retained=arm.goals()
        tcp=np.asarray(contract['episodes'][0]['tcp_offset_m']);rates=np.asarray(arm.joint_velocity_limits_deg_s(),float)
        geometry=GeometryGuard(str(PROJECT/settings['collision_model']),actual,'grasp');geometry.check(actual);geometry.check(retained)
        servo=PolicyDifferentialIK(actual,retained,tcp,rates,max_tcp_speed_mps=None,max_angular_speed_deg_s=None)
        current_gripper_command=arm.read_register(9,116,4)
        home=StartPoseGate(actual,raw[9],tcp,command_joints=retained,command_gripper=current_gripper_command,position_tolerance_m=.010,joint_tolerance_deg=2.)
        if previous_manifest is not None:
            saved=previous_manifest['initial_home']
            home=StartPoseGate(saved['joints_rad'],saved['gripper_raw'],tcp,command_joints=saved['command_joints_rad'],command_gripper=saved['command_gripper_raw'],position_tolerance_m=.010,joint_tolerance_deg=2.,rotation_tolerance_deg=3.)
        gripper_goal=current_gripper_command;gripper_rate=arm.read_register(9,44,4)*.229*4096/60
        profile=arm.read_register(9,112,4)*.229*4096/60
        if profile>0:gripper_rate=min(gripper_rate,profile)
        if not np.isfinite(gripper_rate) or gripper_rate<=0:raise RuntimeError('Invalid hardware gripper rate')
        json_write(output/'manifest.json',dict(source_contract=str(batch/'replay_contract.json'),repo=contract['repo'],revision=contract['revision'],
            source_contract_sha256=sha256(batch/'replay_contract.json'),
            control_hz=25,action_source='HF expert actions; no VLA, no tracker, no cameras',motion_profile='hardware',
            joint_rates_deg_s=rates.tolist(),tcp_speed_cap=None,angular_speed_cap=None,gripper_rate_ticks_s=gripper_rate,
            pose_smoothing_ms=smoothing_ms,joint_governor=governor,joint_acceleration_deg_s2=acceleration_deg_s2 if governor else None,
            joint_jerk_deg_s3=jerk_deg_s3 if governor else None,geometry_profile='grasp',ee_workspace_enforced=False,
            selected_episodes=[r['episode']for r in contract['episodes']],return_home=return_home,
            recorded_time_scale=1.,initial_home=home.record(),start_alignment='Each episode returns to its own recorded measured start, using original holding commands.'))
        if resume_from is not None:
            manifest=json.loads((output/'manifest.json').read_text());manifest['resumed_from']=str(resume_from);manifest['preserved_complete_episodes']=[r['episode']for r in results];json_write(output/'manifest.json',manifest)
        log=(output/'control.jsonl').open('w',buffering=1);load_episode();last_read=None;next_cycle=time.monotonic();last_health=0
        print('NO VLA | SPACE: guarded move to recorded start, then SPACE: replay | P: stop episode | Q: hold/quit',flush=True)
        print(f'{len(contract["episodes"])} recorded marker episode(s); keep box/loose objects clear. Each replay uses original 25Hz timing.',flush=True)
        while True:
            now=time.monotonic()
            if now<next_cycle:time.sleep(min(.005,next_cycle-now));continue
            next_cycle+=period
            if now-next_cycle>period:next_cycle=now+period
            read_start=time.monotonic();actual,currents,raw=arm.telemetry();stamp=time.monotonic()
            if stamp-read_start>.15:raise RuntimeError('Motor feedback read exceeded 150ms')
            dt=period if last_read is None else stamp-last_read;last_read=stamp
            if not 0<dt<=.15:
                if stage in ('waiting_start','ready','between') or (stage=='resetting_start' and not plan.prepared):
                    dt=period  # Window creation/review may pause an idle UI, never replay time.
                    gate.invalidate()
                else:raise RuntimeError('Replay control interval exceeded 150ms')
            if np.any(actual<LOWER) or np.any(actual>UPPER):raise RuntimeError('Measured joint limit violation')
            geometry.check(actual)
            if stamp-last_health>=.5:arm.check_health();last_health=stamp
            actual_pose=pose6(forward(actual,tcp))
            if stage!='replay':gate.observe(actual,raw[9],stamp)
            if stage=='ready':
                status=('READY: SPACE replays this episode at recorded timing'if gate.ready else
                    'Waiting: '+', '.join(gate.diagnostics.get('blocking_checks',[])))
            if stage in ('resetting_start','returning_home'):
                if plan.advance_preflight(geometry,stamp):
                    command=plan.sample(stamp);servo.check_candidate(command,actual,dt);check_joint_path(geometry,servo.goal,command,actual)
                    arm.goal(command);servo.accept(command)
                    desired=gate.command_gripper;goal=gripper_goal+int(np.clip(desired-gripper_goal,-max(1,int(gripper_rate*dt)),max(1,int(gripper_rate*dt))))
                    if goal!=gripper_goal:
                        comm,err=arm.packet.write4ByteTxRx(arm.port,9,116,goal&0xffffffff)
                        if comm!=arm.dxl.COMM_SUCCESS or err:raise RuntimeError('Gripper return command failed')
                        gripper_goal=goal
                    status=f'RETURNING: median XYZ {gate.diagnostics.get("filtered_position_error_m",0)*1000:.1f}mm; wait for READY'
                    if plan.finished(stamp):
                        if stage=='returning_home':stage='finished';break
                        stage='ready';status='READY: SPACE replays this episode at recorded timing'
                        print(f'Episode {item["episode"]} READY; SPACE replays',flush=True)
                else:status='CHECKING joint return path; no motion until checked'
            if stage=='replay':
                if origin is None:
                    origin=stamp;active_result['start_monotonic_s']=stamp
                elapsed=stamp-origin;frame=replay_frame(elapsed,period)
                if frame>=len(data['actions']):
                    if governor:stage='settling_end';tail_started=stamp
                    elif complete_replay(stamp):break
                elif frame!=last_frame:
                    skipped+=max(0,frame-last_frame-1);last_frame=frame;action=data['actions'][frame]
                    try:
                        command,diagnostic=pipeline.propose(action,actual,dt)
                        check_joint_path(geometry,servo.goal,command,actual)
                    except (RuntimeError,ValueError)as exc:
                        finish_episode('controller_guard',str(exc),stamp);stage='between';status='GUARD: '+str(exc)+' | SPACE next / Q quit'
                        if index==len(contract['episodes'])-1:break
                    else:
                        arm.goal(command);pipeline.accept(command)
                        desired=gripper_raw(action[6],closed,opened);step=max(1,int(gripper_rate*dt));goal=gripper_goal+int(np.clip(desired-gripper_goal,-step,step))
                        if goal!=gripper_goal:
                            comm,err=arm.packet.write4ByteTxRx(arm.port,9,116,goal&0xffffffff)
                            if comm!=arm.dxl.COMM_SUCCESS or err:raise RuntimeError('Gripper replay command failed')
                            gripper_goal=goal
                        row=dict(event='sample',episode=item['episode'],source_frame=frame,source_time_s=float(data['timestamp'][frame]),
                            source_clock_error_s=elapsed-frame*period,
                            episode_elapsed_s=elapsed,feedback_monotonic_s=stamp,command_monotonic_s=time.monotonic(),
                            actual_pose6=actual_pose.tolist(),actual_joints_rad=actual.tolist(),gripper_raw=int(raw[9]),
                            gripper_goal_raw=gripper_goal,currents_a=currents.tolist(),dt=dt,**diagnostic)
                        episode_log.write(json.dumps(row)+'\n');samples+=1;status=f'REPLAY {frame+1}/{len(data["actions"])} | P stop / Q quit'
            if stage=='settling_end':
                if stamp-tail_started>5:raise RuntimeError('Final governed target did not settle within 5 seconds')
                command,diagnostic=pipeline.propose(data['actions'][-1],actual,dt)
                check_joint_path(geometry,servo.goal,command,actual)
                arm.goal(command);pipeline.accept(command)
                velocity=float(np.max(np.abs(np.rad2deg(pipeline.governor.velocity))))
                acceleration=float(np.max(np.abs(np.rad2deg(pipeline.governor.acceleration))))
                episode_log.write(json.dumps(dict(event='settle_sample',episode=item['episode'],
                    episode_elapsed_s=stamp-origin,actual_pose6=actual_pose.tolist(),dt=dt,**diagnostic))+'\n')
                status=f'SETTLING final target: reference speed {velocity:.2f}deg/s | P stop / Q quit'
                if velocity<.5 and acceleration<5 and np.max(np.abs(command-actual))<np.deg2rad(3):
                    if complete_replay(stamp):break
            log.write(json.dumps(dict(event='feedback',stage=stage,episode=item['episode'],host_monotonic_s=stamp,
                actual_joints_rad=actual.tolist(),actual_pose6=actual_pose.tolist(),gripper_raw=int(raw[9]),
                start_pose_check=gate.diagnostics if stage!='replay' else None,status=status))+'\n')
            json_write(output/'status.json',dict(stage=stage,status=status,episode=item['episode'],episode_number=index+1,
                active=stage=='replay',frames_sent=samples,completed_episodes=len(results),actual_pose6=actual_pose.tolist(),
                start_pose_check=gate.diagnostics if stage!='replay' else None))
            image=np.zeros((260,1000,3),np.uint8)
            text=[f'Recorded expert replay - NO VLA - episode {item["episode"]} ({index+1}/{len(contract["episodes"])})',status,
                f'Measured XYZ m: {actual_pose[:3].round(4)}',
                f'HF target XYZ m: {data["actions"][max(0,last_frame),:3].round(4)}',
                'SPACE moves to recorded start / starts replay; P stops episode; Q holds/quits']
            for i,line in enumerate(text):cv2.putText(image,line[:125],(12,35+i*43),0,.5,(230,230,230),1)
            cv2.imshow('Expert actions: target vs actual (no VLA)',image);key=cv2.waitKey(1)&0xff
            if key==ord('q'):finish_episode('operator_quit','Q',stamp);break
            if key==ord('p'):
                finish_episode('operator_pause','P',stamp);stage='between' if origin is not None else 'waiting_start';plan=None;status='PAUSED: Q quits, SPACE prepares start'
                if index==len(contract['episodes'])-1 and results:break
            if key==ord(' '):
                if stage in ('waiting_start','between'):
                    if stage=='between':index+=1
                    if index>=len(contract['episodes']):break
                    load_episode();gate.invalidate();plan=begin_return(gate,stamp);stage='resetting_start';origin=None;last_frame=-1;samples=0;skipped=0
                elif stage=='ready':
                    if not gate.ready:status='Wait for measured start settling';continue
                    pipeline=ReplayPipeline(actual,servo.goal,rates,tcp,smoothing_ms=smoothing_ms,governor=governor,
                        acceleration_deg_s2=acceleration_deg_s2,jerk_deg_s3=jerk_deg_s3);servo=pipeline.ik
                    origin=None;last_frame=-1;samples=0;skipped=0;stage='replay'
                    active_result=dict(episode=item['episode'],start_monotonic_s=None,recorded_frames=len(data['actions']),
                        measured_start_pose6=actual_pose.tolist(),start_pose_check=gate.diagnostics,source='HF expert Cartesian actions')
                    episode_log=(output/f'episode_{item["episode"]:06d}.jsonl').open('w',buffering=1)
                    print(f'Replaying episode {item["episode"]}',flush=True)
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}';finish_episode('error',error,time.monotonic());raise
    finally:
        if episode_log is not None:episode_log.close()
        if log is not None:log.close()
        if arm is not None:arm.close()
        cv2.destroyAllWindows()
        json_write(output/'summary.json',dict(finished=True,stage=stage,error=error,episodes=results,
            requested_episodes=[r['episode']for r in contract['episodes']],no_vla=True,
            all_requested_completed=len(results)==len(contract['episodes']) and all(r['full_episode_completed']for r in results),
            all_five_completed=len(results)==5 and all(r['full_episode_completed']for r in results),returned_home=stage=='finished'))
        json_write(output/'status.json',dict(stage='stopped',status='STOPPED' if error is None else error,active=False,completed_episodes=len(results)))
        render_plots(batch,output)
