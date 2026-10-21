"""Continuous right-arm following through the shared pose/velocity servo.

Starts from existing healthy holding torque; no torque/mode/gain/limit writes.
Requires a clear tabletop workspace. Checks robot/table/self geometry, not
arbitrary obstacle avoidance. Q/STOP retains the last holding goals.
"""
import argparse,hashlib,json,time
from pathlib import Path
import cv2,numpy as np
from .cartesian_servo import CartesianIKServo,validate_servo
from .aloha_arm import TapeArm
from .collision import GeometryGuard
from .vx300s_kinematics import forward,LOWER,UPPER
from .tracking_config import load_config

def read_target(path,rig_sha256,settings,now):
    try:record=json.loads(Path(path).read_text())
    except (OSError,json.JSONDecodeError):return None,'waiting for perception'
    if (record.get('schema')!='front_side_observation_v1' or not record.get('running')
        or record.get('rig_sha256')!=rig_sha256 or record.get('output_frame')!='right_base_estimate'):
        return None,'perception unavailable or calibration mismatch'
    age=now-(record.get('timing') or {}).get('oldest_host_arrival_monotonic_s',-float('inf'))
    if record.get('stale',True) or not 0<=age<=settings['max_observation_age_s']:return None,'camera stale'
    track=record.get('track');fusion=record.get('fusion') or {}
    if (fusion.get('identity_recent') and not fusion.get('stale',True)
        and fusion.get('output_frame')=='right_base_estimate' and fusion.get('sources')):
        point=np.asarray(fusion.get('position_output_m'),float)
    elif track and track.get('ready') and track.get('accepted') and track.get('frame')=='right_base_estimate':
        point=np.asarray(track.get('position_m'),float)
    else:return None,'waiting for an accepted object position'
    if point.shape!=(3,) or not np.isfinite(point).all():return None,'invalid object position'
    target=point+[0,0,settings['surface_top_offset_m']+settings['hover_height_m']]
    top=fusion.get('visible_top_z_base_m')
    if top is not None and np.isfinite(top):
        # A partially occluded mask can expose only the lower half of the
        # object. Do not lower toward it more than 8mm below the proxy fallback.
        target[2]=max(top,point[2]+settings['surface_top_offset_m']-.008)+settings['hover_height_m']
    velocity=np.zeros(3)
    if (track and track.get('frame')=='right_base_estimate' and track.get('ready')
            and track.get('accepted') and not track.get('stationary_assumed',False)):
        proposed=np.asarray(track.get('velocity_mps',[0,0,0]),float)
        if proposed.shape==(3,) and np.isfinite(proposed).all():velocity=proposed
    target=target+velocity*age
    if np.any(target<settings['workspace_min_m']) or np.any(target>settings['workspace_max_m']):
        return None,'target outside configured TCP workspace'
    return dict(target_m=target,object_proxy_m=point,velocity_report=record.get('velocity_report'),
                target_velocity_m_s=velocity,
                observation_age_s=age,source_timestamp_s=(track or {}).get('timestamp_s')),None

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--observation',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv);config=load_config(args.config);settings=validate_servo(config.get('servo'))
    project=Path(__file__).resolve().parent.parent
    def resolve(p):
        p=Path(p);return p if p.is_absolute() else project/p
    rigpath=resolve(config['perception']['rig']);rig=json.loads(rigpath.read_text());digest=hashlib.sha256(rigpath.read_bytes()).hexdigest()
    if rig['output_frame']!='right_base_estimate' or rig['front_serial']!='234222300865' or rig['side_serial']!='218722271409':
        raise ValueError('Require this setup\'s right-base front/side camera rig')
    root=Path(args.output);root.mkdir(exist_ok=False,parents=True)
    (root/'config.yaml').write_text(Path(args.config).read_text())
    state=dict(running=True,status='starting',error=None,motion_sent=False,base_calibration_validated=False,
        mode='execute' if args.execute else 'preview',expert_episode=False,rig_sha256=digest,
        observation_file=str(Path(args.observation).resolve()),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    arm=None
    def save():
        (root/'status.tmp').write_text(json.dumps(state,indent=2)+'\n');(root/'status.tmp').replace(root/'status.json')
    try:
        arm=TapeArm();arm.check_health();actual,currents,_=arm.telemetry();retained=arm.goals()
        if np.any(actual<LOWER) or np.any(actual>UPPER):raise RuntimeError('Use leader teleoperation to lift the arm into its normal range first')
        if np.max(np.abs(retained-actual))>np.deg2rad(settings['max_following_error_deg']):raise RuntimeError('Retained holding error exceeds stall bound')
        geometry=GeometryGuard(str(resolve(settings['collision_model'])),actual)
        geometry.check(actual,.025);geometry.check(retained,.025)
        # Read existing motor settings; never write velocity/current limits.
        hardware_rates=arm.joint_velocity_limits_deg_s()
        hardware_currents=np.array([arm.read_register(i,38,2)*.00269 for i in range(1,10)])
        servo=CartesianIKServo(retained,settings,hardware_rates,actual=actual)
        # Current readback is diagnostic only; no software torque/current aborts.
        state.update(configured_joint_velocity_limits_deg_s=hardware_rates,current_limits_a=hardware_currents.tolist(),
            experimental_current_guard=False)
        start=last=time.monotonic();last_health=last_ui=0.;previous_tcp=None
        print('Continuous Cartesian tracking '+('EXECUTING' if args.execute else 'PREVIEW ONLY')+'; existing holding goals preserved.',flush=True)
        with (root/'motion.jsonl').open('w',buffering=1) as log:
            while True:
                now=time.monotonic();dt=now-last
                if dt<1/settings['control_hz']:time.sleep(1/settings['control_hz']-dt);continue
                last=now
                if dt>.15:raise RuntimeError('Control timing exceeded 150 ms')
                actual,currents,_=arm.telemetry()
                measured_tcp=forward(actual,[.16,0,0])[:3,3]
                measured_speed=None if previous_tcp is None else float(np.linalg.norm(measured_tcp-previous_tcp)/dt)
                previous_tcp=measured_tcp.copy()
                if measured_speed is not None and measured_speed>settings['max_tcp_speed_mps']:
                    state.update(measured_fk_tcp_speed_mps=measured_speed,host_monotonic_s=now)
                    raise RuntimeError('Measured FK TCP speed exceeds configured ceiling')
                geometry.check(actual,.02)
                if np.any(actual<LOWER) or np.any(actual>UPPER):raise RuntimeError('Measured joints outside model limits')
                if now-last_health>.25:arm.check_health();last_health=now
                if np.max(np.abs(servo.q-actual))>np.deg2rad(settings['max_following_error_deg']):raise RuntimeError('Measured joint following error exceeds stall bound')
                observation,reason=read_target(args.observation,digest,settings,now)
                if observation is None:servo.hold();state['status']='holding: '+reason
                else:
                    state.update(target_m=observation['target_m'].tolist(),object_proxy_m=observation['object_proxy_m'].tolist(),
                        velocity_report=observation['velocity_report'],observation_age_s=observation['observation_age_s'])
                    try:
                        candidate=servo.step(observation['target_m'],actual,dt,
                            target_velocity_m_s=observation['target_velocity_m_s'])
                        geometry.check(candidate,.03)
                    except ValueError as exc:
                        servo.hold();state['status']='holding: '+str(exc)
                    except RuntimeError as exc:
                        if 'following error' in str(exc):raise
                        servo.hold();state['status']='holding: '+str(exc)
                    else:
                        if args.execute:arm.goal(candidate);servo.accept(candidate);state['motion_sent']=True
                        state['status']='following' if args.execute else 'preview; no motor commands'
                state.update(actual_q_rad=actual.tolist(),goal_q_rad=servo.q.tolist(),currents_a=currents.tolist(),
                    task_space_servo=servo.diagnostics,
                    model_tcp_m=measured_tcp.tolist(),measured_fk_tcp_speed_mps=measured_speed,
                    host_monotonic_s=now,elapsed_s=now-start)
                log.write(json.dumps(state)+'\n');save()
                if now-last_ui>.1:
                    last_ui=now;view=cv2.imread(str(Path(args.observation).parent/'live_frame.jpg'))
                    if view is None:view=np.full((240,1280,3),24,np.uint8)
                    footer=np.full((85,view.shape[1],3),24,np.uint8)
                    labels=['RIGHT ARM | '+state['status']+' | Q stops this controller; torque remains ON',
                        f"Cartesian PID {settings['kp']:g}/{settings['ki']:g}/{settings['kd']:g} | TCP cap {settings['max_tcp_speed_mps']:g} m/s | joint rate: "+('motor settings' if settings['max_joint_speed_deg_s'] is None else str(settings['max_joint_speed_deg_s'])+' deg/s'),
                        'Target (m): '+str(state.get('target_m'))+' | TCP (m): '+str([round(x,3) for x in state['model_tcp_m']])]
                    for i,label in enumerate(labels):cv2.putText(footer,label,(8,22+26*i),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,255,255),1)
                    cv2.imshow('Continuous right-arm Cartesian tracker - Q stop',np.vstack([view,footer]))
                if cv2.waitKey(1)&255==ord('q') or (root/'STOP').exists():state['status']='operator stopped / holding';break
                if now-start>=settings['seconds']:state['status']='session complete / holding';break
    except BaseException as exc:
        state.update(status='stopped / holding goals retained',error=f'{type(exc).__name__}: {exc}');print(state['error'],flush=True)
    finally:
        if arm is not None:arm.close()
        cv2.destroyAllWindows();state.update(running=False,hold_behavior='Last motor goals and torque retained')
        save();(root/'summary.json').write_text(json.dumps(state,indent=2)+'\n')
    return 1 if state['error'] else 0

if __name__=='__main__':raise SystemExit(main())
