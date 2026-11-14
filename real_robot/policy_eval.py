"""Paused learned-policy trial controller. SPACE starts; Q holds and quits."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time
import uuid

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .aloha_arm import TapeArm
from .collision import GeometryGuard
from .dom_dataset import pose6
from .policy_stream import ActionStream, LiveObjectEstimator, gripper_raw, motion_profile, action_clock_index,action_deadline_elapsed,live_camera
from .velocity_innovation import VelocityInnovation
from third_party.DynamicVLA.proactive_vla.phase_gate import CausalPhaseGate,PhaseGateConfig,gate_from_state
from .policy_ik import PolicyDifferentialIK
from .policy_smoothing import PoseTargetSmoother
from .policy_trajectory import JointTrajectoryGovernor,learned_governor_enabled
from .policy_reset import StartPoseGate,StartPoseReturn,check_joint_path
from .grasp_phase import supports_visible_holding
from .tracking_config import load_config
from .object_estimation import add_object_estimation_arguments,resolve_model
from .vx300s_kinematics import forward, LOWER, UPPER, JOINT_NAMES, MOTOR_IDS
from .wrist_capture import EpisodeWrist

PROJECT=Path(__file__).resolve().parents[1]


def save_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(PROJECT/'real_robot/config_policy_eval.yaml'))
    p.add_argument('--perception',type=Path,required=True);p.add_argument('--exchange',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--residual-phase-gate-mode',choices=['constant','causal_rules'],default='constant')
    p.add_argument('--residual-phase-gate-config-json',default='{}')
    p.add_argument('--rtc',choices=['off','guided'],default='off')
    p.add_argument('--velocity-innovation',choices=['off','on'],default='off')
    p.add_argument('--object-inputs',choices=['required','unused'],default='required')
    p.add_argument('--policy-input-profile',choices=['two_frame','single_frame'],default='two_frame')
    p.add_argument('--instruction',default='Pick up the moving orange and white striped ball and place it in the box.')
    p.add_argument('--seconds',type=float,default=600);p.add_argument('--trial-seconds',type=float,default=30)
    p.add_argument('--motion-profile',choices=['hardware','smooth','auto','pilot','standard'],default='hardware')
    p.add_argument('--joint-governor',choices=['auto','on','off'],default='auto',help='Acceleration/jerk governor; off is preview-only for learned policies')
    p.add_argument('--joint-acceleration-deg-s2',type=float,help='Joint reference acceleration bound; offline validation is required before changing it')
    p.add_argument('--joint-jerk-deg-s3',type=float,help='Joint reference jerk bound; offline validation is required before changing it')
    p.add_argument('--max-trials',type=int,default=0)
    add_object_estimation_arguments(p)
    p.add_argument('--pose-smoothing-ms',type=float,default=80.,help='Short quintic EE-target blend; 0 disables')
    p.add_argument('--hold-target-check',action='store_true',help='Control diagnostic: hold the existing commanded EE pose while the VLA pipeline runs')
    p.add_argument('--start-pose',type=Path,help='Reuse a previous session start_pose.json for matched starts across model runs')
    p.add_argument('--start-position-tolerance-mm',type=float,default=10.,help='Settled TCP distance allowed from the fixed start (default 10 mm)')
    p.add_argument('--start-joint-tolerance-deg',type=float,default=2.,help='Maximum measured joint deviation from the fixed start (default 2 degrees)')
    p.add_argument('--ik-method',choices=['bounded_dls','uniform_dls'],default='bounded_dls',help='Bounded weighted DLS, or the historical uniformly scaled DLS baseline')
    p.add_argument('--geometry-profile',choices=['grasp','hover'],default='grasp',help='grasp permits fingertip/table contact; hover preserves the old 20mm geometry margin')
    args=p.parse_args(argv)
    args.object_occlusion_model=resolve_model(args.object_occlusion_model,args.object_motion)
    try:estimator=LiveObjectEstimator(args.object_prediction_max_gap_s,args.object_prediction_max_displacement_m,motion_model=args.object_occlusion_model)
    except ValueError as exc:p.error(str(exc))
    if args.velocity_innovation=='on' and args.object_inputs!='required':p.error('Velocity innovation requires tracked object inputs')
    innovation=VelocityInnovation();source_stamps={}
    phase_gate=CausalPhaseGate(PhaseGateConfig(**json.loads(args.residual_phase_gate_config_json))) if args.residual_phase_gate_mode=='causal_rules' else None
    if phase_gate is not None and args.object_inputs!='required':p.error('Causal gate requires object state inputs')
    config=load_config(args.config,manipulation=True);settings=config['servo']
    if args.pose_smoothing_ms!=0 and not 40<=args.pose_smoothing_ms<=200:p.error('pose-smoothing-ms must be 0 or between 40 and 200')
    if not np.isfinite([args.start_position_tolerance_mm,args.start_joint_tolerance_deg]).all() or min(args.start_position_tolerance_mm,args.start_joint_tolerance_deg)<=0:
        p.error('start tolerances must be finite and positive')
    rigpath=PROJECT/config['perception']['rig'];rig=json.loads(rigpath.read_text())
    from .training_release import validate_capture_rig
    validate_capture_rig(rig)
    digest=hashlib.sha256(rigpath.read_bytes()).hexdigest();rates=motion_profile(args.motion_profile)
    try:governor_enabled=learned_governor_enabled(args.motion_profile,args.joint_governor,args.execute)
    except ValueError as exc:p.error(str(exc))
    for value in [args.joint_acceleration_deg_s2,args.joint_jerk_deg_s3]:
        if value is not None and (not np.isfinite(value) or value<=0):p.error('Acceleration/jerk bounds must be finite and positive')
    rates['joint_accel_deg_s2']=(args.joint_acceleration_deg_s2 or rates.get('joint_accel_deg_s2',120.)) if governor_enabled else None
    rates['joint_jerk_deg_s3']=(args.joint_jerk_deg_s3 or rates.get('joint_jerk_deg_s3',600.)) if governor_enabled else None
    if rates['tcp_m_s'] is None and args.motion_profile!='hardware':rates['tcp_m_s']=settings['max_tcp_speed_mps']
    tcp=np.asarray(rig.get('validated_tcp_offset_m',[.16,0,0]),float)
    calibration=json.loads(Path('/home/ailab/lerobot/.cache/calibration/aloha_default/right_follower.json').read_text())
    if calibration['calib_mode'][-1]!='LINEAR':raise ValueError('Expected calibrated linear gripper')
    closed,opened=calibration['start_pos'][-1],calibration['end_pos'][-1]
    args.output.mkdir(parents=True,exist_ok=False);args.exchange.mkdir(parents=True,exist_ok=True)
    save_json(args.output/'manifest.json',dict(schema='krvla_policy_eval_v1',instruction=args.instruction,
        mode='execute' if args.execute else 'preview',rig_sha256=digest,control_hz=25,
        action_clock='nearest_nominal_25hz_sample_shared_with_expert_replay',
        observation_offsets=[-2,0],policy_observation_offsets=[0] if args.policy_input_profile=='single_frame' else [-2,0],
        velocity_innovation=dict(enabled=args.velocity_innovation=='on',**innovation.settings),
        rtc=args.rtc,async_inference=True,residual_phase_gate_mode=args.residual_phase_gate_mode,
        residual_phase_gate_config=json.loads(args.residual_phase_gate_config_json),
        action_horizon=20,LAAS=True,continuous_inference=True,
        state_semantics='FK of measured follower raw encoders',action_semantics='absolute Cartesian FK of intended follower joint command, then bounded IK',
        supervised_hold_annotation='H requires visible wrist confirmation, measured closure and >=2cm lift',
        tcp_offset_m=tcp.tolist(),gripper_closed_raw=closed,gripper_open_raw=opened,
        motion_profile=args.motion_profile,trajectory_limits=rates,trial_timeout_s=args.trial_seconds,
        max_action_observation_age_steps=None,action_validity='Elapsed-index LAAS; chunk horizon bounds action expiry',
        object_inputs_used_by_policy=args.object_inputs=='required',
        object_gap_prediction_max_s=estimator.max_gap if args.object_inputs=='required' else None,object_occlusion_prediction=estimator.settings if args.object_inputs=='required' else None,
        minimum_table_geometry_clearance_m=0. if args.geometry_profile=='grasp' else .02,
        table_geometry_numeric_allowance_m=.001 if args.geometry_profile=='grasp' else 0.,
        permitted_table_contact_geoms=['vx300s_right/10_left_gripper_finger','vx300s_right/10_right_gripper_finger'] if args.geometry_profile=='grasp' else [],
        geometry_profile=args.geometry_profile,ee_workspace_enforced=False,
        workspace_min_m=None,workspace_max_m=None,
        controller_backend=args.ik_method,ik_method=args.ik_method,joint_trajectory_backend='ruckig' if governor_enabled else None,
        joint_governor=governor_enabled,
        software_velocity_caps=args.motion_profile!='hardware',
        speed_source='VLA pose targets, constrained by the recorded motion profile',
        pose_smoothing_ms=args.pose_smoothing_ms,
        reset_to_session_start=True,start_pose_source=None if args.start_pose is None else str(args.start_pose.resolve()),
        start_position_tolerance_mm=args.start_position_tolerance_mm,start_joint_tolerance_deg=args.start_joint_tolerance_deg,
        control_source='fixed_held_pose_diagnostic' if args.hold_target_check else 'policy',
        study='Supervised closed-loop evaluation with recorded motion ceilings.'))
    arm=wrist=None;active=False;episode=None;trial_count=0;history=deque(maxlen=3);last_env=None
    token=str(uuid.uuid4());stream=ActionStream(token);started=time.monotonic();origin=started;last=started
    held=False;minimum_z=None;last_result=None;status='PAUSED';last_health=0.;previous_image=None
    motion_sent=False;last_index=-1;video=None;error_text=None;last_inference_s=None;smoother=None;diagnostic_action=None;trajectory=None
    start_gate=None;return_plan=None;reset_pending=False;reset_count=0
    def hold_reference():
        servo.hold()
        if smoother is not None:smoother.reset(forward(servo.goal,tcp))
        if trajectory is not None:trajectory.reset(servo.goal)
    def finish(outcome,reason,reset_after=False):
        nonlocal active,episode,held,token,stream,video,reset_pending
        if args.hold_target_check and outcome=='success':outcome='diagnostic_complete'
        if episode is not None:
            episode.update(outcome=outcome,reason=reason,end_monotonic_s=time.monotonic(),holding_confirmed=held)
            save_json(args.output/f'trial-{trial_count:03d}.json',episode)
        active=False;episode=None;held=False;history.clear();token=str(uuid.uuid4());stream=ActionStream(token);estimator.reset();innovation.reset();source_stamps.clear()
        if video is not None:video.release();video=None
        if phase_gate is not None:phase_gate.reset()
        wrist.enabled.clear();hold_reference()
        if start_gate is not None:start_gate.invalidate()
        reset_pending=bool(args.execute and reset_after)
    try:
        arm=TapeArm();arm.check_health();actual,_,raw=arm.telemetry();retained=arm.goals()
        save_json(args.output/'startup_joints.json',dict(host_monotonic_s=time.monotonic(),
            motion_sent=False,joints=[dict(joint=name,motor_id=motor,
                measured_deg=float(np.rad2deg(actual[i])),retained_goal_deg=float(np.rad2deg(retained[i])),
                lower_deg=float(np.rad2deg(LOWER[i])),upper_deg=float(np.rad2deg(UPPER[i])),
                measured_outside=bool(actual[i]<LOWER[i] or actual[i]>UPPER[i]),
                goal_outside=bool(retained[i]<LOWER[i] or retained[i]>UPPER[i]),raw_position=raw.get(motor))
                for i,(name,motor) in enumerate(zip(JOINT_NAMES,MOTOR_IDS))]))
        geometry=GeometryGuard(str(PROJECT/settings['collision_model']),actual,args.geometry_profile)
        geometry.check(actual,.02);geometry.check(retained,.02)
        hardware_rates=np.asarray(arm.joint_velocity_limits_deg_s(),float)
        if rates['joints_deg_s'] is not None:hardware_rates=np.minimum(hardware_rates,rates['joints_deg_s'])
        if args.motion_profile!='hardware' and settings['max_joint_speed_deg_s'] is not None:hardware_rates=np.minimum(hardware_rates,settings['max_joint_speed_deg_s'])
        rates['joints_deg_s']=hardware_rates.tolist()
        servo=PolicyDifferentialIK(actual,retained,tcp,hardware_rates,max_following_error_deg=6.,
            max_tcp_speed_mps=rates['tcp_m_s'],max_angular_speed_deg_s=rates['angular_deg_s'],method=args.ik_method)
        if governor_enabled:
            trajectory=JointTrajectoryGovernor(retained,hardware_rates,rates['joint_accel_deg_s2'],rates['joint_jerk_deg_s3'])
        if args.pose_smoothing_ms:
            smoother=PoseTargetSmoother(args.pose_smoothing_ms/1000);smoother.reset(forward(retained,tcp))
        gripper_rate=arm.read_register(9,44,4)*.229*4096/60
        profile_rate=arm.read_register(9,112,4)*.229*4096/60
        if profile_rate>0:gripper_rate=min(gripper_rate,profile_rate)
        if not np.isfinite(gripper_rate) or gripper_rate<=0:raise RuntimeError('Invalid existing gripper velocity limit')
        if args.motion_profile not in ('auto','hardware'):gripper_rate=min(gripper_rate,250.)
        manifest=json.loads((args.output/'manifest.json').read_text())
        manifest.update(trajectory_limits=rates,gripper_speed_ceiling_ticks_s=gripper_rate)
        save_json(args.output/'manifest.json',manifest)
        gripper_goal=int(arm.read_register(9,116,4))
        if not min(closed,opened)<=gripper_goal<=max(closed,opened):raise RuntimeError('Existing gripper goal outside calibrated range')
        start_joints=actual.copy();start_gripper=int(raw[9]);start_commands=retained.copy();start_gripper_command=gripper_goal
        if args.start_pose is not None:
            saved=json.loads(args.start_pose.read_text())
            if (saved.get('schema')!='krvla_eval_start_pose_v1' or saved.get('robot')!='right_follower'
                    or not np.allclose(saved['tcp_offset_m'],tcp,atol=1e-9,rtol=0)):
                raise ValueError('Start pose does not match this robot/TCP')
            start_joints=np.asarray(saved['joints_rad'],float);start_gripper=int(saved['gripper_raw'])
            if 'command_joints_rad' not in saved or 'command_gripper_raw' not in saved:
                raise ValueError('Saved start pose lacks its original holding commands; capture a new session start or upgrade that reference from its initial-hold log')
            start_commands=np.asarray(saved['command_joints_rad'],float);start_gripper_command=int(saved['command_gripper_raw'])
        if not min(closed,opened)<=start_gripper<=max(closed,opened):raise RuntimeError('Start gripper position outside calibrated range')
        if not min(closed,opened)<=start_gripper_command<=max(closed,opened):raise RuntimeError('Start gripper command outside calibrated range')
        start_gate=StartPoseGate(start_joints,start_gripper,tcp,command_joints=start_commands,command_gripper=start_gripper_command,
            position_tolerance_m=args.start_position_tolerance_mm/1000,joint_tolerance_deg=args.start_joint_tolerance_deg)
        geometry.check(start_joints,.02);geometry.check(start_commands,.02)
        save_json(args.output/'start_pose.json',dict(start_gate.record(),captured_monotonic_s=time.monotonic(),
            source=None if args.start_pose is None else str(args.start_pose.resolve())))
        wrist=EpisodeWrist(args.output/'wrist','218622278247',fps=30);wrist.thread.start()
        print('PAUSED | SPACE starts trial | ENTER success | X failure | P pause/fail | R reselect mask | H confirm visible grasp+lift | Q hold/quit',flush=True)
        print('Execution enabled: '+str(args.execute)+'. Existing holding torque is required; no torque/mode/limit changes.',flush=True)
        print('Measured-state differential IK | 25 Hz pose targets | '+args.motion_profile+' motion ceilings: '+json.dumps(rates),flush=True)
        print('IK method: '+args.ik_method,flush=True)
        print(f'EE target blend: {args.pose_smoothing_ms:g} ms | EE workspace box: disabled | geometry profile: {args.geometry_profile}',flush=True)
        print('Fixed start saved: start_pose.json | automatic guarded return after completed trials | next trial requires measured settling.',flush=True)
        print(f'Start tolerance: {args.start_position_tolerance_mm:g} mm TCP / {args.start_joint_tolerance_deg:g} deg per joint; 0.4s quiet feedback required.',flush=True)
        if trajectory is not None:print(f'Joint trajectories: Ruckig | acceleration <={rates["joint_accel_deg_s2"]:g} deg/s^2 | jerk <={rates["joint_jerk_deg_s3"]:g} deg/s^3',flush=True)
        if args.hold_target_check:print('CONTROL CHECK: fixed current held-pose reference; excluded from task success rates.',flush=True)
        last=time.monotonic()
        with (args.output/'control.jsonl').open('w',buffering=1) as log:
            while time.monotonic()-started<args.seconds and not (args.output/'STOP').exists():
                if args.max_trials>0 and trial_count>=args.max_trials and not active and return_plan is None and not reset_pending:break
                now=time.monotonic();delay=.04-(now-last)
                if delay>0:time.sleep(delay)
                now=time.monotonic();dt=now-last;last=now
                if dt>.15:
                    if active or return_plan is not None:raise RuntimeError('Control timing exceeded 150 ms')
                    hold_reference()
                    dt=.04
                read_start=time.monotonic();actual,currents,raw=arm.telemetry();read_end=time.monotonic()
                if read_end-read_start>.15:raise RuntimeError('Motor feedback read exceeded 150 ms')
                if np.any(actual<LOWER) or np.any(actual>UPPER):raise RuntimeError('Measured joints outside limits')
                pose=forward(actual,tcp);state=pose6(pose);index=action_clock_index(read_end-origin)
                try:
                    geometry.check(actual,.02)
                except RuntimeError as exc:
                    log.write(json.dumps(dict(event='measured_geometry_violation',host_monotonic_s=read_end,
                        actual_q_rad=actual.tolist(),retained_goal_q_rad=servo.goal.tolist(),tcp_pose6=state.tolist(),reason=str(exc)))+'\n')
                    save_json(args.output/'status.json',dict(status='ERROR: '+str(exc),active=active,trial=trial_count,
                        actual_q_rad=actual.tolist(),tcp_pose6=state.tolist(),host_monotonic_s=read_end,motion_sent=motion_sent))
                    raise
                if now-last_health>.5:arm.check_health();last_health=now
                if not active:start_gate.observe(actual,raw[9],read_end)
                wrist_image,wrist_stamp=wrist.snapshot()
                front_image=record=None;reason=None
                try:
                    if args.object_inputs=='required' and ((args.perception/'MASK_PENDING').exists() or (args.perception/'RESEED').exists()):
                        raise ValueError('Object mask selection pending')
                    with np.load(args.perception/'policy_front.npz',allow_pickle=False) as z:
                        front_image=z['bgr'];record=json.loads(str(z['metadata']))
                    front_stamp=live_camera(record,read_end,digest)
                    if wrist_image is None or wrist_stamp is None or not 0<=read_end-wrist_stamp<=.2:
                        raise ValueError('Wrist camera stale')
                    if abs(front_stamp-wrist_stamp)>.1:raise ValueError('Front/wrist skew exceeds 100 ms')
                    if args.object_inputs=='unused':
                        env=None;required=valid=None
                        estimator.diagnostics=dict(source='not_used_by_policy',predicted=False)
                    elif held and last_env is not None:
                        env=last_env.copy();env[6:]=0.;required=False;valid=False
                        estimator.diagnostics=dict(source='operator_confirmed_holding_velocity_neutral',horizon_s=0.)
                    else:
                        env=estimator.read(record,read_end,digest);last_env=env.copy();required=True;valid=True
                    reason=None
                except (OSError,ValueError,KeyError) as exc:reason=str(exc)
                if active and args.velocity_innovation=='on':
                    if reason or estimator.diagnostics.get('source')!='accepted_measurement':innovation.interrupt()
                    else:innovation.observe(front_stamp,env[6:9])
                if not active:
                    status=('PAUSED: '+reason if reason else
                        'READY: fixed start verified; SPACE starts trial' if start_gate.ready else
                        'PAUSED: SPACE returns to fixed start; wait for READY before starting')
                if active and read_end-origin>args.trial_seconds:
                    if args.hold_target_check:
                        finish('diagnostic_complete','constant held-pose duration completed',reset_after=True);status='PAUSED: hold diagnostic complete'
                    else:finish('failure','timeout',reset_after=True);status='PAUSED: timeout'
                if reset_pending:
                    return_plan=StartPoseReturn(servo.goal,start_gate,hardware_rates,rates['tcp_m_s'],rates['angular_deg_s'],
                        geometry,None,None,read_end)
                    reset_pending=False;start_gate.invalidate()
                    log.write(json.dumps(dict(event='reset_started',after_trial=trial_count,host_monotonic_s=read_end,
                        target_joints_rad=start_gate.joints.tolist(),command_joints_rad=start_gate.command_joints.tolist(),duration_s=return_plan.duration))+'\n')
                if return_plan is not None:
                    if not return_plan.advance_preflight(geometry,read_end):
                        status='CHECKING reset path; P pauses, Q stops'
                    else:
                        candidate=return_plan.sample(read_end)
                        servo.check_candidate(candidate,actual,dt);return_plan.check_pose(candidate,geometry)
                        check_joint_path(geometry,servo.goal,candidate,actual)
                        arm.goal(candidate);servo.accept(candidate)
                        requested=start_gate.command_gripper;step=max(1,int(gripper_rate*dt))
                        goal=gripper_goal+int(np.clip(requested-gripper_goal,-step,step))
                        if goal!=gripper_goal:
                            comm,error=arm.packet.write4ByteTxRx(arm.port,9,116,goal&0xffffffff)
                            if comm!=arm.dxl.COMM_SUCCESS or error:raise RuntimeError('Gripper reset command failed')
                            gripper_goal=goal
                        check=start_gate.diagnostics
                        status=(f'RESETTING: XYZ median {1000*check.get("filtered_position_error_m",0):.1f}/{1000*start_gate.position_tolerance_m:g}mm; '
                            +', '.join(check.get('blocking_checks',[]))+' | P/Q stop')
                        log.write(json.dumps(dict(event='reset_step',host_monotonic_s=read_end,
                            candidate_joints_rad=candidate.tolist(),gripper_goal_raw=gripper_goal,start_pose_check=start_gate.diagnostics))+'\n')
                        if return_plan.finished(read_end):
                            return_plan=None;reset_count+=1;hold_reference()
                            log.write(json.dumps(dict(event='reset_completed',after_trial=trial_count,host_monotonic_s=read_end,
                                start_pose_check=start_gate.diagnostics))+'\n')
                            status='READY: fixed start verified; reset marker, then SPACE starts trial'
                if active:
                    minimum_z=min(minimum_z,float(state[2]))
                    if reason:
                        # Hold on missing tracking, but preserve the episode and
                        # future action buffer. LAAS drops elapsed entries and
                        # the source-age limit rejects stale predictions when
                        # tracking resumes. Each gap must not restart inference.
                        history.clear();hold_reference()
                        status='HOLD: '+reason
                    elif index!=last_index:
                        if args.velocity_innovation=='on':
                            source_stamps[index]=front_stamp
                            source_stamps={i:t for i,t in source_stamps.items() if i>=index-256}
                        try:
                            with np.load(args.exchange/'actions.npz',allow_pickle=False) as z:
                                response=json.loads(str(z['metadata']));actions=z['actions']
                            response_key=(response['token'],response['index'])
                            if response_key!=last_result:
                                current=index  # One sampled action index per feedback/control tick.
                                skipped=stream.merge(response['token'],response['index'],actions,current)
                                if response['token']==token:last_inference_s=response['inference_seconds']
                                last_result=response_key
                                log.write(json.dumps(dict(event='chunk',index=current,skipped_prefix=skipped,response=response))+'\n')
                        except FileNotFoundError:pass
                        observation=dict(index=index,front=front_image.copy(),wrist=wrist_image.copy(),state=state.copy(),
                            front_stamp=front_stamp,wrist_stamp=wrist_stamp,state_stamp=read_end)
                        if args.object_inputs=='required':observation.update(environment=env,required=required,valid=valid)
                        history.append(observation)
                        if phase_gate is not None:
                            phase_gate.observe(read_end,state[:3],env[:3],env[6:9],
                                2*(gripper_goal-closed)/(opened-closed)-1,ee_rotation_xyz=state[3:6])
                        selected=None
                        if args.policy_input_profile=='single_frame':selected=[observation,observation]
                        elif len(history)==3 and history[-1]['index']-history[0]['index']==2:selected=[history[0],history[-1]]
                        if selected is not None:
                            meta=dict(token=token,index=index,created_monotonic_s=time.monotonic(),
                                instruction=args.instruction,trial=trial_count,object_quality=estimator.diagnostics,
                                object_inputs_used_by_policy=args.object_inputs=='required',
                                observation_host_stamps_s={name:[row[name+'_stamp'] for row in selected] for name in ['front','wrist','state']})
                            payload_keys=['front','wrist','state']+(['environment','required','valid'] if args.object_inputs=='required' else [])
                            payload={key:np.stack([row[key] for row in selected]) for key in payload_keys}
                            if phase_gate is not None:
                                payload['phase_gate']=gate_from_state(phase_gate.state(read_end),20,.04)
                                meta['phase_gate_mode']='causal_rules';meta['causal_phase']=phase_gate.phase
                            if args.rtc=='guided':
                                payload['rtc_prefix_actions']=stream.rtc_prefix(index)
                                meta['rtc_prefix_length']=len(payload['rtc_prefix_actions'])
                                meta['rtc_source_index']=index
                                meta['rtc_plan_source_index']=stream.last_source
                            with (args.exchange/'observation.tmp').open('wb') as f:np.savez(f,metadata=json.dumps(meta),**payload)
                            (args.exchange/'observation.tmp').replace(args.exchange/'observation.npz')
                        current=index;action=stream.pop(current)
                        if action is None:hold_reference();status='HOLD: waiting for temporally valid action'
                        else:
                            policy_action=action.copy()
                            if args.velocity_innovation=='on' and not args.hold_target_check:
                                action=innovation.apply(action,source_stamps[stream.popped_source],read_end)
                            if args.hold_target_check:action=diagnostic_action.copy()
                            if not np.isfinite(action).all():
                                raise RuntimeError('Non-finite policy action')
                            else:
                                target=np.eye(4);target[:3,3]=action[:3];target[:3,:3]=Rotation.from_euler('xyz',action[3:6]).as_matrix()
                                if smoother is not None:target=smoother.step(target,dt)
                                ik_candidate=servo.step(target,actual,dt)
                                candidate=ik_candidate if trajectory is None else trajectory.propose(ik_candidate,dt)
                                if trajectory is not None:servo.check_candidate(candidate,actual,dt)
                                try:
                                    check_joint_path(geometry,servo.goal,candidate,actual)
                                except RuntimeError as exc:
                                    log.write(json.dumps(dict(event='proposed_geometry_violation',index=current,
                                        candidate_q_rad=candidate.tolist(),actual_q_rad=actual.tolist(),reason=str(exc)))+'\n')
                                    finish('failure',str(exc),reset_after=True);status='PAUSED: '+str(exc)
                                    continue
                                requested=gripper_raw(action[6],closed,opened)
                                gripper_step=int(gripper_rate*dt)
                                goal=gripper_goal+int(np.clip(requested-gripper_goal,-gripper_step,gripper_step))
                                decision_stamp=time.monotonic()
                                if args.execute:
                                    if action_deadline_elapsed(current,decision_stamp-origin):
                                        hold_reference();status='HOLD: command timestep elapsed'
                                    else:
                                        arm.goal(candidate);servo.accept(candidate)
                                        if trajectory is not None:trajectory.accept(candidate)
                                        if goal!=gripper_goal:
                                            comm,error=arm.packet.write4ByteTxRx(arm.port,9,116,goal&0xffffffff)
                                            if comm!=arm.dxl.COMM_SUCCESS or error:raise RuntimeError('Gripper command failed')
                                            gripper_goal=goal
                                        motion_sent=True;status='EXECUTING'
                                        if estimator.diagnostics.get('predicted'):
                                            status+=f' | predicted object {estimator.diagnostics["horizon_s"]:.2f}s'
                                else:status='PREVIEW: predicted command checked, no motion sent'
                                log.write(json.dumps(dict(event='action',index=current,absolute_action=action.tolist(),
                                    command_decision_monotonic_s=decision_stamp,target_nominal_monotonic_s=origin+current*.04,
                                    target_deadline_monotonic_s=origin+(current+1)*.04,
                                    velocity_innovation=innovation.diagnostics if args.velocity_innovation=='on' else None,
                                    policy_absolute_action=policy_action.tolist(),object_quality=estimator.diagnostics,control_source='fixed_held_pose_diagnostic' if args.hold_target_check else 'policy',
                                    candidate_joints_rad=candidate.tolist(),gripper_goal_raw=goal,
                                    ik_candidate_joints_rad=ik_candidate.tolist(),
                                    joint_trajectory=None if trajectory is None else trajectory.diagnostics,
                                    smoothed_pose6=pose6(target).tolist(),smoothing=None if smoother is None else smoother.diagnostics,
                                    ik=servo.diagnostics,gripper_prediction_clipped=bool(abs(action[6])>1),status=status))+'\n')
                last_index=index
                save_json(args.output/'status.json',dict(status=status,active=active,trial=trial_count,
                    host_monotonic_s=time.monotonic(),motion_sent=motion_sent,holding_confirmed=held,
                    actual_q_rad=actual.tolist(),tcp_pose6=state.tolist(),gripper_raw=int(raw[9]),tracking_reason=reason,
                    motion_profile=args.motion_profile,inference_seconds=last_inference_s,object_quality=estimator.diagnostics,
                    resetting=return_plan is not None or reset_pending,start_pose_check=start_gate.diagnostics,
                    queued_actions=len(stream.pending),last_chunk_source_index=stream.last_source))
                log.write(json.dumps(dict(event='feedback',index=index,host_monotonic_s=read_end,
                    tcp_pose6=state.tolist(),raw_gripper=int(raw[9]),active=active,holding_confirmed=held,
                    tracking_reason=reason,controller_status=status,object_quality=estimator.diagnostics,
                    actual_q_rad=actual.tolist(),currents_a=currents.tolist()))+'\n')
                panels=[]
                for image in [front_image,wrist_image]:
                    panels.append(cv2.resize(image,(640,480)) if image is not None else np.zeros((480,640,3),np.uint8))
                view=np.hstack(panels);footer=np.zeros((90,1280,3),np.uint8)
                for j,text in enumerate([status,'SPACE trial | ENTER success | X failure | P pause | R reselect mask | Q hold/quit',
                    'H: confirm object visibly held in wrist view + closed gripper + >=2cm lift']):
                    cv2.putText(footer,text[:140],(8,23+28*j),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
                if active and video is not None:video.write(view)
                cv2.imshow('Learned KRVLA closed-loop pilot',np.vstack([view,footer]));key=cv2.waitKey(1)&255
                if key in [ord('q'),27]:
                    if active:finish('failure','operator takeover/quit')
                    return_plan=None;reset_pending=False;hold_reference()
                    break
                if key==ord('p') and not active:
                    return_plan=None;reset_pending=False;hold_reference();start_gate.invalidate()
                    status='PAUSED: return cancelled; SPACE checks the fixed start again'
                if key==ord('r') and args.object_inputs=='unused':
                    if active:finish('failure','operator paused; object tracker unused')
                    return_plan=None;reset_pending=False;start_gate.invalidate();history.clear();hold_reference()
                    status='PAUSED: DOM has no object mask; SPACE checks the fixed start'
                if key==ord('r') and args.object_inputs=='required':
                    if active:finish('failure','operator paused to reselect object mask')
                    return_plan=None;reset_pending=False;start_gate.invalidate()
                    (args.perception/'MASK_PENDING').touch()
                    (args.perception/'RESEED').touch();history.clear();hold_reference()
                    estimator.reset()
                    status='PAUSED: select the task object in the front-camera prompt window'
                if key==ord(' ') and not active and return_plan is None and not reset_pending:
                    if not start_gate.ready:
                        if args.execute:reset_pending=True;status='Preparing return to the fixed start; wait for READY'
                        else:status='PREVIEW: fixed start not reached; no reset motion is enabled'
                        continue
                    if reason:status='PAUSED: '+reason;continue
                    token=str(uuid.uuid4());stream=ActionStream(token);history.clear();last_result=None;innovation.reset();source_stamps.clear()
                    if phase_gate is not None:phase_gate.reset()
                    origin=time.monotonic();last_index=-1;trial_count+=1;active=True;held=False;minimum_z=float(state[2])
                    hold_reference()
                    episode=dict(schema='krvla_policy_trial_v1',trial=trial_count,token=token,start_monotonic_s=origin,
                        instruction=args.instruction,mode='execute' if args.execute else 'preview',
                        evaluation_type='fixed_held_pose_diagnostic' if args.hold_target_check else 'task_rollout',
                        common_start_verified=True,start_pose_reference=str(args.output/'start_pose.json'),
                        measured_start_q_rad=actual.tolist(),measured_start_tcp_pose6=state.tolist(),
                        measured_start_gripper_raw=int(raw[9]),start_pose_check=start_gate.diagnostics.copy(),completed_resets=reset_count,
                        excluded_from_task_success_rate=bool(args.hold_target_check))
                    if args.hold_target_check:
                        diagnostic_action=np.r_[pose6(forward(servo.goal,tcp)),2*(gripper_goal-closed)/(opened-closed)-1]
                        episode['fixed_target']=diagnostic_action.tolist()
                    video=cv2.VideoWriter(str(args.output/f'trial-{trial_count:03d}-views.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),25.,(1280,480))
                    if not video.isOpened():raise RuntimeError('Cannot initialize trial video recording')
                    wrist.enabled.set();status='TRIAL: waiting for observation history'
                    print('Started trial '+str(trial_count),flush=True)
                if active and key in [13,10,ord('x'),ord('p')]:
                    outcome='success' if key in [13,10] else 'failure'
                    finish(outcome,'operator labelled '+outcome,reset_after=key!=ord('p'));status='PAUSED: '+outcome
                    print(status,flush=True)
                if active and key==ord('h'):
                    closure=(raw[9]-closed)/(opened-closed)
                    commanded_closure=(gripper_goal-closed)/(opened-closed)
                    if supports_visible_holding(closure,commanded_closure,state[2]-minimum_z) and wrist_stamp is not None and time.monotonic()-wrist_stamp<=.2:
                        held=True;history.clear();token=str(uuid.uuid4());stream=ActionStream(token)
                        hold_reference()
                        episode['hold_confirmed_monotonic_s']=time.monotonic();episode['hold_confirmation']='operator wrist visual + measured closed gripper + measured lift'
                        status='Holding confirmed; external-object velocity neutral'
                    else:status='H rejected: closure/lift/fresh wrist conditions not met'
    except BaseException as exc:
        error_text=f'{type(exc).__name__}: {exc}'
        if active and wrist is not None:finish('failure',error_text)
        raise
    finally:
        if active and wrist is not None:finish('failure','controller stopped or error')
        if wrist is not None:wrist.close()
        if arm is not None:arm.close()
        cv2.destroyAllWindows()
        final_status={}
        if (args.output/'status.json').exists():
            final_status=json.loads((args.output/'status.json').read_text())
        final_status.update(status='STOPPED' if error_text is None else 'ERROR: '+error_text,
                            active=False,resetting=False,motion_sent=motion_sent,host_monotonic_s=time.monotonic())
        save_json(args.output/'status.json',final_status)
        save_json(args.output/'summary.json',dict(trials=trial_count,motion_sent=motion_sent,finished=True,error=error_text,
            completed_resets=reset_count,reset_interrupted=return_plan is not None or reset_pending,
            fixed_start_verified=False if start_gate is None else bool(start_gate.ready)))


if __name__=='__main__':main()
