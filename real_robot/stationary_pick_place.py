"""Prepare, register a fixed box, and run the shared automatic demonstrator."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from .geometry import validate_transform, transform_point
from .grasp_prior import fresh_observation
from .automatic_demonstrator import DemonstratorConfig
from .demonstrator_adapters import prior_from_template
from .pick_geometry import PickGeometry, inside_polygon, clearance_center
from .tracking_config import load_config
from .vx300_demonstrator import TapeDemonstrator, manipulation_motion
from .task_space_servo import PoseVelocityServo
from .vx300s_kinematics import forward
from .wrist_alignment import blue_seed, line_error, tape_feature
from .wrist_feed import read_wrist_observation
from .wrist_target import capture_joints

PROJECT = Path(__file__).resolve().parent.parent


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        temporary.replace(path)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT/path


def latest_observation(template):
    pointer = read_json(resolve(template['perception_root'])/'latest-perception.json')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if pointer['boot_id'] != boot:
        raise ValueError('Start a new perception session on this boot')
    path = Path(pointer['observation_file'])
    record = read_json(path)
    point = fresh_observation(record, template['rig_sha256'], time.monotonic())
    return path, record, point


def rs_deproject(timing):
    # Pure SDK math; does not open a camera or reset a device.
    import pyrealsense2 as rs
    values = timing['intrinsics']
    intr = rs.intrinsics()
    for key in ('width','height','fx','fy','ppx','ppy','coeffs'):
        setattr(intr,key,values[key])
    intr.model = getattr(rs.distortion,timing['distortion_model'].split('.')[-1])
    return lambda pixel,z: np.asarray(rs.rs2_deproject_pixel_to_point(intr,list(map(float,pixel)),float(z)))


def prepare(args):
    session = resolve(args.session)
    manifest = read_json(session/'manifest.json')
    episode_path = session/'episodes'/f'episode_{args.episode:04d}.json'
    episode = read_json(episode_path)
    if episode['outcome'] != 'success':
        raise ValueError('Select an operator-confirmed successful demonstration')
    destination = episode
    if episode['box_id'] != args.box_id:
        destination = read_json(PROJECT/'scratch/real_robot/grasps/blue-tape-box1-02/episodes/episode_0001.json')
        if destination['box_id'] != args.box_id:
            raise ValueError('No matching taught box destination')
    stages = {}
    for name in ('pregrasp','before_close','lifted','above_box','release'):
        source = destination if name in ('above_box','release') else episode
        sample = source['events'][name]['sample']
        stages[name] = {'T_base_tcp': sample['T_base_tcp_candidate'], 'q_rad':sample['q_model_rad']}
    event = episode['events']['pregrasp']
    with np.load(session/event['wrist_rgbd']['file']) as frame:
        _,mask = blue_seed(frame['bgr'])
        reference = tape_feature(frame['bgr'],mask,frame['depth_m'])
    if not reference['axis_reliable']:
        raise ValueError('Selected demonstration lacks a reliable pregrasp wrist axis')
    handeye_path = resolve(args.handeye)
    handeye = read_json(handeye_path)
    T = validate_transform(handeye['T_gripper_wrist_camera'])
    records = []
    for parent in sorted((PROJECT/'scratch/real_robot/grasps').glob('blue-tape-*')):
        for p in sorted((parent/'episodes').glob('episode_*.json')):
            e = read_json(p)
            if e.get('outcome') != 'success': continue
            features = {}
            for stage in ('pregrasp','before_close'):
                v = e['events'][stage]
                with np.load(parent/v['wrist_rgbd']['file']) as frame:
                    try:
                        _,mask = blue_seed(frame['bgr'])
                        features[stage] = tape_feature(frame['bgr'],mask,frame['depth_m'])
                    except ValueError: continue
            if len(features) != 2: continue
            q0 = np.asarray(e['events']['pregrasp']['sample']['q_model_rad'])
            q1 = np.asarray(e['events']['before_close']['sample']['q_model_rad'])
            dq = q1[5]-q0[5]
            records.append({'session':parent.name,'episode_id':e['episode_id'],
                'pregrasp_axis_deg':float(np.rad2deg(features['pregrasp']['axis_rad'])),
                'pregrasp_roll_deg':float(np.rad2deg(q0[5])),
                'pregrasp_axis_reliable':features['pregrasp']['axis_reliable'],
                'roll_change_deg':float(np.rad2deg(dq)),
                'image_axis_change_deg':float(np.rad2deg(line_error(features['before_close']['axis_rad'],features['pregrasp']['axis_rad']))),
                'unclipped_before_close':not features['before_close']['clipped']})
    calibration = manifest['buses']['follower']['calibration']
    grip_index = calibration['motor_names'].index('gripper')
    result = {'schema':'stationary_tape_pick_template_v1','source_episode':str(episode_path),
        'source_episode_sha256':hashlib.sha256(episode_path.read_bytes()).hexdigest(),
        'box_id':args.box_id,'rig_sha256':manifest['rig_sha256'],'rig':manifest['rig'],
        'tcp_offset_m':manifest['tcp_offset_m'],'object_reference_m':np.asarray(episode['T_base_tape_reference'])[:3,3].tolist(),
        'object_reference_pose':validate_transform(episode['T_base_tape_reference']).tolist(),
        'stages':stages,'wrist_reference':reference,'wrist_serial':manifest['wrist_serial'],
        'T_gripper_wrist_camera':T.tolist(),'handeye_validated':False,
        'handeye_source':str(handeye_path),'perception_root':'scratch/real_robot/tracking',
        'collision_model':str(PROJECT/'scratch/real_robot/right-handeye/hover-plan/offline-collision-model.xml'),
        'efficienttam_checkpoint':str(PROJECT/'third_party/EfficientTAM/checkpoints/efficienttam_ti_512x512.pt'),
        'efficienttam_config':'configs/efficienttam/efficienttam_ti_512x512.yaml',
        'gripper':{'open_raw':episode['events']['before_close']['sample']['follower_raw_ticks'][-1],
            'calibrated_open_raw':calibration['end_pos'][grip_index],
            'empty_closed_raw':calibration['start_pos'][grip_index],
            'observed_contact_raw':episode['events']['closed']['sample']['follower_raw_ticks'][-1],
            'existing_current_limit_a':manifest['hardware_readback']['current_limits_a'][-1]},
        'alignment_calibration':{'recorded_pairs':records,
            'method':'Image error servo, followed by a local 3-degree roll probe before every descent.',
            'model_image_angle_per_roll':-float(T[0,2]),
            'note':'Recorded successful views constrain the desired axis, not an absolute wrist angle. Clipped close-up masks cannot calibrate the axis.'},
        'validated_for_hardware':False,'stationary_only':False,
        'limits':{'joint_rate_deg_s':3,'tcp_rate_m_s':.020,
                  'carried_tape_radius_envelope_m':.045}}
    if result['tcp_offset_m'] != [.16,0,0]:
        raise ValueError('This pilot requires the same 160mm TCP as the recorded setup')
    prior=prior_from_template(result)
    result['automatic_demonstrator']={'schema':'automatic_demonstrator_v1',
        'core':'real_robot.automatic_demonstrator.AutomaticDemonstrator',
        'robot_backend':'shared_pose_velocity_servo','config':asdict(DemonstratorConfig()),
        'prior':{'T_object_approach':prior.T_object_approach.tolist(),
                 'T_object_grasp':prior.T_object_grasp.tolist(),'origin_semantics':prior.semantics},
        'object_rotation_source':'collection heading prior plus wrist-camera alignment',
        'object_states_are_policy_inputs':False,
        'note':'DynamicVLA-style task core; VX300 adapter and feedback-based one-attempt transitions.'}
    output = resolve(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    save_json(output,result,exclusive=True)
    print('Template:',output)
    print('Wrist reference axis:',round(np.rad2deg(reference['axis_rad']),2),'degrees; center:',reference['center_px'])
    print('Recorded successful pairs:',len(records),'; local image/roll gain is verified before descent.')
    print('Demonstrator: shared task-space core; VX300 damped least-squares differential IK.')


def register_box(args):
    template = read_json(resolve(args.template))
    path, record, _ = latest_observation(template)
    rgbd = record.get('rgbd') or {}
    if not rgbd.get('files',{}).get('front'):
        raise ValueError('Use perception --record-rgbd for a fresh registration image')
    with np.load(path.parent/rgbd['files']['front']) as frame:
        image,depth = frame['bgr'],frame['depth_m']
    roi = args.roi
    if roi is None:
        roi = cv2.selectROI('Select visible EMPTY BOX INTERIOR FLOOR, then ENTER',image,False,False)
        cv2.destroyAllWindows()
    x,y,w,h = map(int,roi)
    if min(w,h)<10 or x<0 or y<0 or x+w>image.shape[1] or y+h>image.shape[0]:
        raise ValueError('Empty box-interior selection')
    from .efficienttam_tracker import EfficientTAMTracker
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError(f'Box registration already exists: {output}')
    asset_root = output.parent/(output.stem+'-assets')
    if asset_root.exists():
        # Keep diagnostics from a rejected selection, and permit another one.
        asset_root=output.parent/(output.stem+'-assets-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    asset_root.mkdir(parents=True,exist_ok=False)
    tracker = EfficientTAMTracker(template['efficienttam_checkpoint'],asset_root,
                                  template['efficienttam_config'],args.device)
    timing = rgbd['views']['front']['timing']
    mask,info = tracker.seed('box',image,timing['sensor_timestamp_s'],roi)
    if mask is None: raise ValueError('Box mask unavailable; select the visible interior again')
    region = np.zeros_like(mask);region[y:y+h,x:x+w]=255
    mask = cv2.bitwise_and(mask,region)
    cv2.imwrite(str(asset_root/'selected-front.png'),image)
    cv2.imwrite(str(asset_root/'interior-mask.png'),mask)
    np.savez(asset_root/'registration-rgbd.npz',bgr=image,depth_m=depth)
    ys,xs = np.where(mask>0)
    deproject = rs_deproject(timing)
    transform = np.asarray(template['rig']['T_base_front_candidate'])
    points=[]
    for yy,xx in zip(ys[::max(1,len(ys)//1000)],xs[::max(1,len(xs)//1000)]):
        z = depth[yy,xx]
        if np.isfinite(z) and .1<z<1.5:
            points.append(transform_point(transform,deproject([xx,yy],z)))
    points = np.asarray(points)
    if len(points)<80: raise ValueError('Insufficient box-floor depth')
    floor = float(np.percentile(points[:,2],20))
    points = points[np.abs(points[:,2]-floor)<.012]
    if len(points)<60:raise ValueError('Box floor is not a consistent horizontal depth surface')
    polygon = cv2.convexHull(points[:,:2].astype(np.float32)).reshape(-1,2)
    footprint_method='visible_floor_depth'
    if args.rim_corners is not None:
        if args.box_size_mm is None:
            raise ValueError('Rim corners require measured --box-size-mm WIDTH LENGTH')
        corners=np.asarray(args.rim_corners,float).reshape(4,2)
        if (not np.isfinite(corners).all() or np.any(corners<0)
                or np.any(corners[:,0]>=image.shape[1]) or np.any(corners[:,1]>=image.shape[0])
                or not cv2.isContourConvex(corners.astype(np.float32))):
            raise ValueError('Select four inner rim corners in perimeter order')
        opening=[]
        for pixel in corners:
            ray=transform[:3,:3]@deproject(pixel,1.)
            scale=(args.rim_z_base_m-transform[2,3])/ray[2]
            if scale<=0:raise ValueError('Rim plane is behind the camera')
            opening.append((transform[:3,3]+scale*ray)[:2])
        polygon=np.asarray(opening,np.float32)
        edges=np.linalg.norm(np.roll(polygon,-1,axis=0)-polygon,axis=1)
        dimensions=np.sort([(edges[0]+edges[2])/2,(edges[1]+edges[3])/2])
        measured=np.sort(np.asarray(args.box_size_mm,float)/1000.)
        if (not np.isfinite(measured).all() or measured[0]<=.07
                or np.any(dimensions<.75*measured) or np.any(dimensions>1.05*measured)):
            raise ValueError(f'Opening projection {dimensions.tolist()}m disagrees with measured box dimensions')
        footprint_method='inner_rim_rays_at_measured_height; visible_floor_depth_is_partial'
    center,clearance = clearance_center(polygon)
    save_json(asset_root/'geometry-review.json',{'front_roi_xywh':[x,y,w,h],
        'interior_polygon_xy_m':polygon.tolist(),'release_xy_m':center.tolist(),
        'release_clearance_m':clearance,'floor_z_base_m':floor,'front_timing':timing,
        'footprint_method':footprint_method,'rim_corners_px':args.rim_corners,'box_size_mm':args.box_size_mm})
    if not inside_polygon(center,polygon,.035):
        raise ValueError(f'Selected interior provides {clearance*1000:.1f}mm release-point clearance; 35mm required. Select the full usable interior')
    if not floor+.01<args.rim_z_base_m<.20:
        raise ValueError('Measured rim Z must exceed the visible floor and be below 0.20m')
    result={'schema':'fixed_box_registration_v1','box_id':template['box_id'],
        'rig_sha256':template['rig_sha256'],'interior_polygon_xy_m':polygon.tolist(),
        'release_xy_m':center.tolist(),'floor_z_base_m':floor,'rim_z_base_m':args.rim_z_base_m,
        'release_clearance_m':clearance,
        'footprint_method':footprint_method,'rim_corners_px':args.rim_corners,'box_size_mm':args.box_size_mm,
        'source_observation':str(path),'source_rgbd':rgbd['files']['front'],'front_roi_xywh':[x,y,w,h],
        'assets_directory':str(asset_root),
        'efficienttam':info,'fixed_box':True,'registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'note':'Confirmed fixed box interior; '+footprint_method+'. Keep box/cameras/base fixed. Calibration remains provisional.'}
    save_json(output,result,exclusive=True)
    print('Registered:',output,'center:',center,'floor Z:',floor)


def run(args):
    if not np.isfinite(args.speed_scale) or args.speed_scale <= 0:
        raise ValueError('Speed scale must be finite and positive')
    template=read_json(resolve(args.template)); box=read_json(resolve(args.box_registration))
    config_path=resolve(args.config)
    config=load_config(config_path)
    rates=manipulation_motion(template,config.get('manipulation'),args.speed_scale)
    # No software torque/current aborts in this executor. Current is recorded
    # and used for gripper contact sensing; existing motor firmware handles limits.
    if template.get('schema')!='stationary_tape_pick_template_v1' or box.get('schema')!='fixed_box_registration_v1':
        raise ValueError('Invalid template or box registration schema')
    if box['box_id']!=template['box_id'] or box['rig_sha256']!=template['rig_sha256']:
        raise ValueError('Box ID or rig does not match template')
    if box.get('simulation_only') or not box.get('fixed_box'):
        raise ValueError('Register the actual fixed box before running; simulation geometry is not executable')
    polygon=np.asarray(box['interior_polygon_xy_m'],float)
    if (polygon.ndim!=2 or polygon.shape[1]!=2 or len(polygon)<3 or not np.isfinite(polygon).all()
            or not .01<box['rim_z_base_m']<.20
            or not inside_polygon(box['release_xy_m'],polygon,.035)):
        raise ValueError('Invalid box interior or rim clearance')
    if hashlib.sha256(Path(template['source_episode']).read_bytes()).hexdigest()!=template['source_episode_sha256']:
        raise ValueError('Source demonstration changed; prepare the template again')
    if not Path('/dev/ttyDXL_follower_right').exists():
        raise RuntimeError('Right follower serial device is not exposed in this runtime; run on the robot workstation')
    observation_file, observation, point=latest_observation(template)
    if args.execute and not all((observation.get('rgbd') or {}).get('files', {}).get(role) for role in ('front','side')):
        raise ValueError('Automatic demonstration collection requires perception --record-rgbd')
    output=resolve(args.output) if args.output else PROJECT/'scratch/real_robot/picks'/datetime.now(timezone.utc).strftime('automatic-%Y%m%dT%H%M%SZ')
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/'template.json',template);save_json(output/'box.json',box)
    (output/'config.yaml').write_text(config_path.read_text())
    save_json(output/'manifest.json',{'schema':'automatic_demonstration_session_v1',
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'demonstrator':'real_robot.automatic_demonstrator.AutomaticDemonstrator',
        'robot_backend':'shared_pose_velocity_servo','stationary_only':False,
        'blended_approach_alignment':True,'curved_lift_transport':True,
        'placement_verification':'front_efficienttam_rgbd_after_release',
        'config_file':str(config_path),'experimental_current_guard':False,
        'speed_scale':args.speed_scale,'motion_rates':rates,
        'fixed_observation_file':str(observation_file),'wrist_rgbd_recording':args.execute,
        'teacher_fields':['teacher','target_tcp_pose','differential_ik','placement_verification'],
        'policy_observation_sources':['front_rgb','side_rgb','wrist_rgb','robot_proprioception'],
        'estimated_object_states_are_policy_inputs':False,
        'time_note':'Native camera timestamps and joint host timestamps are recorded; not hardware-synchronized exposures.'})
    child_env=os.environ.copy()
    # aloha_arm disables CUDA at import. Start the camera-only process
    # first so it retains the actual GPU visibility of the workstation session.
    child=subprocess.Popen([sys.executable,'-u','-m','real_robot.wrist_pick_view',
        '--template',str(resolve(args.template)),'--output',str(output/'wrist'),'--device',args.device,
        *(['--record-rgbd'] if args.execute else [])],
        cwd=PROJECT,env=child_env)
    arm=None;worker=None
    stop=threading.Event();start=threading.Event();done=threading.Event()
    prompt_pause=threading.Event();prompt_paused=threading.Event()
    await_prompt_id=[None]
    state={'running':True,'status':'preview / SPACE starts one attempt','motion_sent':False,
           'success':False,'error':None,'torque_writes':False,'execute':args.execute,
           'demonstrator':'automatic_demonstrator_v1','robot_backend':'shared_pose_velocity_servo',
           'teacher_states_are_policy_inputs':False,'measured_speed_guard_enabled':False,
           'experimental_current_guard':False,
           'speed_scale':args.speed_scale,'motion_rates':rates}
    preview_heartbeat=[time.monotonic()]
    def wrist_record():
        record,diagnostics=read_wrist_observation(output/'wrist/observation.json',time.monotonic())
        state['wrist_feed']=diagnostics
        return record
    def wrist_error(context):
        info=state.get('wrist_feed') or {}
        return RuntimeError(f"Wrist feed {info.get('status','unavailable')} {context}: "
                            f"age={info.get('age_s')}s, processing={info.get('processing_state')}, "
                            f"reason={info.get('reason')}")
    try:
        from .aloha_arm import TapeArm
        arm=TapeArm();arm.check_health()
        actual,currents,raw=arm.telemetry();retained=arm.goals()
        if forward(actual,template['tcp_offset_m'])[2,3]<.10:
            raise ValueError('Prepare an existing held pose with TCP at least 0.10m above table using the leader first')
        hardware_rates=arm.joint_velocity_limits_deg_s()
        def make_servo(actual_q,retained_q):
            return PoseVelocityServo(actual_q,retained_q,template['tcp_offset_m'],
                kp=rates['position_kp'],rotation_kp=rates['rotation_kp'],
                max_following_error_deg=None,
                hardware_joint_rates_deg_s=hardware_rates)
        servo=make_servo(actual,retained)
        geometry=PickGeometry(template['collision_model'],actual,box)
        geometry.check(actual)
        gripper_goal=arm.read_register(9,116,4)
        hardware_currents=np.array([arm.read_register(i,38,2)*.00269 for i in range(1,10)])
        state['hardware_current_limits_a']=hardware_currents.tolist()
        state['hardware_joint_velocity_limits_deg_s']=hardware_rates
        state['initial_actual_q_rad']=actual.tolist()
        state['initial_retained_goal_q_rad']=retained.tolist()
        print('Software torque/current aborts: disabled',
              '| firmware current limits unchanged:', hardware_currents.tolist(),flush=True)
        print(f'Motion speed scale: {args.speed_scale:g}x | rates: {rates}',flush=True)
        def control():
            nonlocal gripper_goal
            machine=None;last=time.monotonic();health=last
            previous_feedback=None;last_stage=None;tcp_velocity=np.zeros(3)
            joint_history=[]
            try:
                with (output/'motion.jsonl').open('x',buffering=1) as log:
                    while not stop.is_set():
                        now=time.monotonic();dt=now-last
                        if dt<1/30:stop.wait(1/30-dt);continue
                        last=now
                        control_dt=min(dt,.10)
                        q,currents,raw=arm.telemetry()
                        feedback_stamp=time.monotonic()
                        joint_history.append((feedback_stamp,q.copy()))
                        joint_history[:]=[(t,j) for t,j in joint_history if feedback_stamp-t<=2.]
                        pose=forward(q,template['tcp_offset_m'])
                        state.update(actual_q_rad=q.tolist(),currents_a=currents.tolist(),
                            raw_gripper=raw[9],tcp_m=pose[:3,3].tolist(),host_monotonic_s=feedback_stamp)
                        # Diagnostic only: speed is not an abort condition.
                        if previous_feedback is not None:
                            stamp,position=previous_feedback
                            measured_velocity=(pose[:3,3]-position)/(feedback_stamp-stamp)
                            elapsed=feedback_stamp-stamp
                            tcp_velocity+=elapsed/(.12+elapsed)*(measured_velocity-tcp_velocity)
                            state['measured_tcp_speed_m_s']=float(np.linalg.norm(measured_velocity))
                        state['filtered_tcp_velocity_m_s']=tcp_velocity.tolist()
                        previous_feedback=(feedback_stamp,pose[:3,3].copy())
                        state['measured_speed_guard_enabled']=False
                        if now-health>.25:arm.check_health();health=time.monotonic()
                        if now-preview_heartbeat[0]>.5 and not prompt_pause.is_set():
                            raise RuntimeError('Operator preview heartbeat lost')
                        if child.poll() is not None and (machine is None or machine.needs_wrist_camera):
                            raise RuntimeError('Wrist perception process exited')
                        wrist=wrist_record()
                        fixed=None;fixed_point=None
                        try:
                            fixed=read_json(observation_file)
                            fixed_point=fresh_observation(fixed,template['rig_sha256'],time.monotonic())
                        except (OSError,ValueError):pass
                        hint={'active':False,'host_monotonic_s':feedback_stamp}
                        if fixed_point is not None and (machine is None or machine.needs_object):
                            track=fixed.get('track') or {}
                            velocity=np.asarray(track.get('velocity_mps',[0,0,0]),float)
                            if (velocity.shape!=(3,) or not np.isfinite(velocity).all()
                                    or not track.get('accepted') or not track.get('ready')
                                    or track.get('frame')!='right_base_estimate' or track.get('stationary_assumed')):
                                velocity=np.zeros(3)
                            joint_velocity=np.zeros(6)
                            if len(joint_history)>1:
                                old_stamp,old_q=joint_history[-2]
                                joint_velocity=(q-old_q)/(feedback_stamp-old_stamp)
                            hint.update(active=True,q_rad=q.tolist(),joint_velocity_rad_s=joint_velocity.tolist(),
                                object_position_m=fixed_point.tolist(),object_velocity_m_s=velocity.tolist(),
                                object_stamp_s=fixed['timing']['oldest_host_arrival_monotonic_s'])
                        if (output/'wrist').exists():save_json(output/'wrist/target_hint.json',hint)
                        requested=await_prompt_id[0]
                        acknowledged=(wrist or {}).get('tracker') or {}
                        if requested is not None and acknowledged.get('prompt_id')==requested:
                            await_prompt_id[0]=None
                        if prompt_pause.is_set() or await_prompt_id[0] is not None:
                            servo.hold()
                            if machine is not None:
                                machine.core.last_now=time.monotonic()
                                machine.core.stable_since=None
                            prompt_paused.set()
                            state['status']='holding / mark the tape' if prompt_pause.is_set() else 'holding / applying wrist prompt'
                            log.write(json.dumps(dict(state),allow_nan=False)+'\n')
                            save_json(output/'status.json',state)
                            continue
                        prompt_paused.clear()
                        state['ready_to_start']=bool(machine is None and wrist is not None and fixed_point is not None)
                        if machine is None:
                            state['status']='ready / SPACE starts pursuit and pick' if state['ready_to_start'] else 'preview / waiting for fixed and wrist feeds'
                        if start.is_set() and machine is None and state['ready_to_start']:
                            machine=TapeDemonstrator(template,box,fixed_point,q,time.monotonic(),motion=rates)
                            state['demonstrator_config']=asdict(machine.core.config)
                            servo=make_servo(q,arm.goals())
                        if machine is not None:
                            stamp=((wrist or {}).get('timing') or {}).get('host_arrival_monotonic_s')
                            capture_q=None if stamp is None else capture_joints(joint_history,stamp)
                            action=machine.update(time.monotonic(),q,raw[9],currents[-1],wrist,
                                object_record=fixed if fixed_point is not None else None,
                                front_record=fixed,
                                tcp_velocity_m_s=tcp_velocity,wrist_capture_q=capture_q)
                            state['target_estimate_source']=machine.object_estimator.source
                            state['placement_verification']=machine.placement_evidence
                            if action is None or (wrist is None and machine.needs_wrist_holding_feedback):
                                servo.hold()
                                geometry.check(q,**machine.flags())
                                state.update(status=machine.stage+' / waiting for fresh target feedback',
                                    gripper_target_raw=gripper_goal)
                            else:
                                flags=machine.flags()
                                geometry.check(q,**flags)
                                previous_goal=servo.goal.copy()
                                profile_name=machine.motion_profile(pose)
                                profile=rates[profile_name]
                                joint_rates=np.full(6,profile['joint_rate_deg_s'])
                                tcp_rate=profile['tcp_rate_m_s']
                                relative_rate=None
                                if machine.needs_object:
                                    target_speed=float(np.linalg.norm(action.target_velocity_m_s))
                                    relative_rate=tcp_rate
                                    tcp_rate=min(config.get('servo',{}).get('max_tcp_speed_mps',.5),tcp_rate+target_speed)
                                    # Match object translation using the same
                                    # motor readbacks as continuous tracking;
                                    # retain slower relative contact corrections.
                                    blend=min(1.,target_speed/.03)
                                    joint_rates+=(np.asarray(hardware_rates)-joint_rates)*blend
                                if machine.stage in ('APPROACH_ABOVE_OBJECT','ALIGN_GRASP','APPROACH_OBJECT','GRASP_OBJECT'):
                                    joint_rates[-1]=rates['alignment_wrist_roll_rate_deg_s']
                                command=servo.step(action.tcp_pose,q,control_dt,
                                    tcp_rate_m_s=tcp_rate,joint_rate_deg_s=joint_rates,
                                    angular_rate_deg_s=profile['angular_rate_deg_s'],
                                    acceleration_m_s2=rates['acceleration_m_s2'],
                                    target_velocity_m_s=action.target_velocity_m_s,
                                    relative_tcp_rate_m_s=relative_rate)
                                # Check both actual-to-command and goal-to-command
                                # segments, including retained holding compensation.
                                # Drop containment applies to the measured TCP,
                                # not to compensated joint setpoints along a path.
                                # Keep robot/table/box collision checks on both.
                                path_flags={**flags,'over_box':False}
                                geometry.path(q,command,**path_flags)
                                geometry.path(previous_goal,command,**path_flags)
                                if time.monotonic()-health>.25:arm.check_health();health=time.monotonic()
                                # If the consumed state expires during geometry
                                # calculation, retain the previous goals and
                                # consume the next frame instead of aborting.
                                if machine.needs_object and machine.object_state is not None:
                                    consumed_age=time.monotonic()-machine.object_state.stamp_s
                                    if not 0<=consumed_age<=.25:
                                        servo.hold()
                                        state['status']=machine.stage+' / waiting for next target frame'
                                        save_json(output/'status.json',state)
                                        continue
                                if time.monotonic()-preview_heartbeat[0]>.5:raise RuntimeError('Preview heartbeat lost before command')
                                if stop.is_set():break
                                gripper_step=max(1,int(rates['gripper_rate_raw_s']*control_dt))
                                goal=gripper_goal+int(np.clip(machine.gripper_target-gripper_goal,-gripper_step,gripper_step))
                                if not template['gripper']['empty_closed_raw']<=goal<=template['gripper']['calibrated_open_raw']:
                                    raise RuntimeError('Gripper target outside taught calibration bounds')
                                arm.goal(command)
                                servo.accept(command)
                                if goal!=gripper_goal:
                                    comm,error=arm.packet.write4ByteTxRx(arm.port,9,116,int(goal)&0xffffffff)
                                    if comm!=arm.dxl.COMM_SUCCESS or error:raise RuntimeError('Gripper goal communication failed')
                                    gripper_goal=goal
                                state.update(status=machine.stage,motion_sent=True,roll_gain=machine.roll_gain,
                                    wrist_alignment_phase=machine.aligner.phase,gripper_target_raw=gripper_goal,
                                    goal_q_rad=command.tolist(),motion_profile=profile_name,
                                    target_tcp_pose=action.tcp_pose.tolist(),differential_ik=servo.diagnostics,
                                    teacher={'object_state':machine.teacher_state,'gripper_action':action.gripper,
                                             'target_velocity_m_s':action.target_velocity_m_s.tolist(),
                                             'optional_object_feedback_rejection':machine.object_feedback_reason,
                                             'alignment_reference_stamp_s':action.metadata.get('alignment_reference_stamp_s'),
                                             'wrist_only_alignment':machine.object_estimator.source.startswith('wrist'),
                                             'descent_fraction':action.metadata.get('descent_fraction'),
                                             'path_progress_m':action.metadata.get('path_progress_m'),
                                             'policy_input':False})
                            if machine.stage!=last_stage:print('Pick stage:',machine.stage,flush=True)
                            last_stage=machine.stage
                            if machine.done:
                                state['success']=True;done.set()
                        state['wrist_feature']=None if wrist is None else wrist.get('feature')
                        state['observation_refs']={'front_side_session':str(observation_file.parent),
                            'front_side_rgbd':None if fixed is None else fixed.get('rgbd'),
                            'wrist_rgbd':None if wrist is None else wrist.get('rgbd'),
                            'wrist_timing':None if wrist is None else wrist.get('timing')}
                        log.write(json.dumps(dict(state),allow_nan=False)+'\n')
                        save_json(output/'status.json',state)
                        if done.is_set():break
            except BaseException as exc:
                state.update(error=f'{type(exc).__name__}: {exc}',status='stopped')
                print(state['error'],flush=True)
            finally:done.set()
        worker=threading.Thread(target=control,name='automatic-pick-control',daemon=True);worker.start()
        print('Pick output:',output,'\nStarts in preview. SPACE starts one attempt with --execute; Q stops new goals.',flush=True)
        while not done.is_set():
            preview_heartbeat[0]=time.monotonic()
            live_path=output/'wrist/live.jpg'
            view=cv2.imread(str(live_path)) if live_path.exists() else None
            if view is None:view=np.zeros((480,640,3),np.uint8)
            reference=np.rint(template['wrist_reference']['center_px']).astype(int)
            cv2.drawMarker(view,tuple(reference),(0,255,255),cv2.MARKER_CROSS,20,2)
            footer=np.full((90,640,3),24,np.uint8)
            labels=[state['status'],'SPACE: start | R: mark wrist tape | Q: retain hold',
                    'Moving/stationary tape | '+box['box_id']+' | existing torque must be ON']
            if state['status']=='VERIFY_PLACE':
                evidence=state.get('placement_verification') or {}
                labels[2]=f"FRONT {evidence.get('verified_frames',0)}/{evidence.get('required_frames',5)}: {evidence.get('reason','waiting for placement')}"
            for i,label in enumerate(labels):cv2.putText(footer,label,(8,22+26*i),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,255,255),1)
            cv2.imshow('Automatic demonstrator - wrist alignment',np.vstack([view,footer]))
            key=cv2.waitKey(10)&255
            if key==ord('q') or (output/'STOP').exists():stop.set();break
            if key==32:
                if args.execute:start.set()
                else:print('Preview only. Restart with --execute to permit commands.',flush=True)
            if key==ord('r'):
                prompt_pause.set()
                try:
                    if not prompt_paused.wait(timeout=1.):
                        print('Wrist selection unavailable: control has not acknowledged the hold.',flush=True)
                        continue
                    snapshot=view.copy();frame_file=None;selection_stamp=None
                    try:
                        record=read_json(output/'wrist/observation.json')
                        candidate_file=(record.get('rgbd') or {}).get('file')
                        if candidate_file:
                            with np.load(output/'wrist'/candidate_file) as frame:
                                snapshot=frame['bgr'].copy()
                            frame_file=candidate_file
                            selection_stamp=record['timing']['sensor_timestamp_s']
                    except (OSError,ValueError,KeyError):pass
                    title='Hold tape still; draw tape box, ENTER confirms, ESC cancels'
                    box_roi=cv2.selectROI(title,snapshot,fromCenter=False,showCrosshair=True)
                    cv2.destroyWindow(title)
                    if box_roi[2]>0 and box_roi[3]>0:
                        prompt_id=time.monotonic_ns()
                        await_prompt_id[0]=prompt_id
                        save_json(output/'wrist/prompt_request.json',{
                            'request_id':prompt_id,'box_xywh':list(map(int,box_roi)),
                            'frame_file':frame_file,'sensor_timestamp_s':selection_stamp})
                finally:
                    preview_heartbeat[0]=time.monotonic()
                    prompt_pause.clear()
            if args.execute and (output/'START').exists():start.set()
    except BaseException as exc:
        state.update(error=f'{type(exc).__name__}: {exc}',status='stopped')
        raise
    finally:
        stop.set()
        if worker is not None:worker.join(timeout=10)
        if arm is not None and (worker is None or not worker.is_alive()):arm.close()
        wrist_dir=output/'wrist'
        if wrist_dir.exists():(wrist_dir/'STOP').touch()
        try:child.wait(timeout=15)
        except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=5)
        cv2.destroyAllWindows()
        state['running']=False
        save_json(output/'summary.json',state);save_json(output/'status.json',state)
    return int(bool(state['error']))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare',help='Offline: derive a single grasp template and inspect wrist correlations')
    p.add_argument('--session',default='scratch/real_robot/grasps/blue-tape-box2-02')
    p.add_argument('--episode',type=int,default=0)
    p.add_argument('--box-id',choices=('box_1','box_2'),default='box_2')
    p.add_argument('--handeye',default='scratch/real_robot/right-handeye/session-20261019-03/candidate-calibration.json')
    p.add_argument('--output',required=True)
    p=sub.add_parser('register-box',help='Camera only: manually select a fixed box interior using EfficientTAM')
    p.add_argument('--template',required=True);p.add_argument('--output',required=True)
    p.add_argument('--roi',type=int,nargs=4,help='Explicit interior-floor image rectangle X Y W H; otherwise select in GUI')
    p.add_argument('--rim-corners',type=float,nargs=8,help='Four visible inner-rim corners in perimeter order; ray-plane footprint for an occluded floor')
    p.add_argument('--box-size-mm',type=float,nargs=2,help='Measured box WIDTH LENGTH for checking a rim-corner footprint')
    p.add_argument('--rim-z-base-m',type=float,required=True,help='Measured rim Z above the modeled table/base plane, meters')
    p.add_argument('--device',default='cuda')
    p=sub.add_parser('run',help='Continuous pursuit, wrist alignment and curved pick/place; SPACE starts one attempt')
    p.add_argument('--template',required=True);p.add_argument('--box-registration',required=True)
    p.add_argument('--config',default='real_robot/config.yaml',help='Tracking configuration to validate and snapshot')
    p.add_argument('--speed-scale',type=float,default=1.5,
                   help='Multiply phase joint/TCP, angular and gripper motion rates (default: 1.5)')
    p.add_argument('--output');p.add_argument('--execute',action='store_true');p.add_argument('--device',default='cuda')
    args=parser.parse_args(argv)
    try:
        result={'prepare':prepare,'register-box':register_box,'run':run}[args.command](args)
    except (ValueError,RuntimeError,OSError,KeyError) as exc:
        parser.exit(2,str(exc)+'\n')
    return result or 0


if __name__=='__main__':
    raise SystemExit(main())
