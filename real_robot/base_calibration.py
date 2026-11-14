"""Supervised chessboard/ChArUco TCP contacts and held-out camera-to-base checks.

Capture starts paused. Only --execute plus an operator T press enables the
existing clutched leader controller. Fit/report operations never open hardware.
"""
import os
# Set before importing NumPy/OpenCV; later environment changes do not resize
# an already initialized BLAS pool competing with serial feedback.
for _thread_setting in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_thread_setting]='1'

import argparse
from collections import deque
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import queue
import time

import cv2
import numpy as np

from .geometry import fit_rigid_transform, transform_point, validate_transform
from .multiview import CameraPair, charuco_detector, estimate_board, pose_difference
from .calibration_board import Chessboard, ChessboardDetector, contact_layout

FIT_IDS = (0, 4, 20, 24)
CHECK_IDS = (6, 18)
TARGETS = [(placement, corner, split) for placement in range(2)
           for split, ids in [('fit', FIT_IDS), ('check', CHECK_IDS)] for corner in ids]
WINDOW='Base calibration - supervised point contacts'
PITCH_TOLERANCE_DEG=2.


class MultiScaleCharucoDetector:
    """Decode small printed markers at 3x, retaining native pixel coordinates."""
    def __init__(self, detector, required_ids=FIT_IDS+CHECK_IDS):
        self.detector=detector
        self.required_ids=set(required_ids)
        self.last_scale=1

    def detectBoard(self, image):
        native=self.detector.detectBoard(image)
        def score(result):
            ids=result[1]
            found=set() if ids is None else set(ids.ravel().tolist())
            return (self.required_ids<=found,len(found),0 if result[3] is None else len(result[3]))
        self.last_scale=1
        if score(native)[0] and score(native)[1]>=12:
            return native
        enlarged=cv2.resize(image,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
        scaled=self.detector.detectBoard(enlarged)
        if score(scaled)<=score(native):return native
        corners,ids,markers,marker_ids=scaled
        # Resize maps destination pixel centres back as (x+.5)/scale-.5.
        corners=None if corners is None else ((corners+.5)/3-.5).astype(np.float32)
        markers=None if markers is None else [((p+.5)/3-.5).astype(np.float32) for p in markers]
        self.last_scale=3
        return corners,ids,markers,marker_ids


def board_readiness(history, detection, frame_age_s, expected_corners=25, required_ids=FIT_IDS+CHECK_IDS, board_type='charuco'):
    """Separate detection/freshness failures from the stationary-pose window."""
    count=int(detection.get('corner_count',0))
    status=dict(ready=False,corners=count,markers=int(detection.get('marker_count',0)),
                detector_reason=detection.get('reason'),valid_frames=len(history),
                stable_span_s=0. if len(history)<2 else history[-1][0]-history[0][0],
                frame_age_s=float(frame_age_s))
    marker_text=f', {status["markers"]} markers' if board_type=='charuco' else ''
    if not 0<=frame_age_s<=.20:
        status['message']=f'Waiting for fresh camera image (age {frame_age_s*1000:.0f} ms)'
    elif not detection.get('valid'):
        reason=detection.get('reason','not_detected')
        status['message']=(f'Board detection: {count}/{expected_corners} corners{marker_text}; '
                           f'{reason}. Show full printed face, clear arm/hands, check lighting.')
    else:
        missing=sorted(set(required_ids)-set(detection.get('ids',[])))
        status['missing_corner_ids']=missing
        if missing:
            status['message']=f'Board seen ({count}/{expected_corners} corners), missing IDs {missing}; keep whole board visible'
        elif len(history)<8 or status['stable_span_s']<.6:
            status['message']=(f'Board seen: {count}/{expected_corners}; building steady history '
                               f'{len(history)}/8 frames, {status["stable_span_s"]:.2f}/0.60 s')
        else:
            changes=[pose_difference(d['T_camera_board'],detection['T_camera_board']) for _,d in history]
            distance=max(v[0] for v in changes);angle=max(v[1] for v in changes)
            status.update(pose_variation_mm=distance*1000,pose_variation_deg=angle)
            if distance>.002 or angle>1:
                status['message']=f'Board pose estimate varies {distance*1000:.1f} mm / {angle:.1f} deg; need <=2 mm / 1 deg'
            else:
                status.update(ready=True,message=f'BOARD READY: {count}/{expected_corners} corners. Press B to lock reference.')
    return status


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path=Path(path)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def summarize(errors):
    return dict(count=len(errors),mean_m=float(np.mean(errors)),
                rms_m=float(np.sqrt(np.mean(np.asarray(errors)**2))),max_m=float(np.max(errors)))


def tool_pitch_deg(row):
    axis=validate_transform(row['T_base_tcp_candidate'])[:3,0]
    return float(np.degrees(np.arctan2(-axis[2],np.linalg.norm(axis[:2]))))


def check_contact_pitch(row, requested_pitch, tolerance_deg=PITCH_TOLERANCE_DEG):
    actual=tool_pitch_deg(row)
    if abs(actual-requested_pitch)>tolerance_deg:
        raise ValueError(f'Actual pitch {actual:.1f} deg; need {requested_pitch} +/- {tolerance_deg:g} deg before S')
    return actual


def board_depth_agreement(reference):
    gaps={key:float(np.linalg.norm(np.asarray(point['camera_depth_m'])-point['camera_pnp_m']))
          for key,point in reference['points'].items()}
    return dict(corner_gaps_m=gaps,max_gap_m=max(gaps.values()),limit_m=.010,
                passed=max(gaps.values())<=.010,
                note='A >10 mm camera-space disagreement cannot allow both RGB-board and depth contact errors <=5 mm. Check print dimensions and depth geometry.')


def fit_contacts(session, max_error_m=.005, refine_tcp=True):
    root=Path(session)
    manifest=json.loads((root/'manifest.json').read_text())
    samples=json.loads((root/'contacts.json').read_text())
    if manifest.get('schema')!='krvla_base_contacts_v1' or not manifest.get('mounts_unchanged'):
        raise ValueError('Require a base-contact session with fixed mounts')
    if not np.isfinite(max_error_m) or not 0 < max_error_m <= .01:
        raise ValueError('Maximum error must be positive and at most 10 mm')
    targets=contact_layout(manifest.get('board_squares',[6,6]))['targets']
    if manifest.get('targets') is not None and [tuple(t) for t in manifest['targets']]!=targets:
        raise ValueError('Recorded target schedule does not match the board geometry')
    if len(samples)!=len(targets) or [(s['placement'],s['corner_id'],s['split']) for s in samples]!=targets:
        raise ValueError('Need all 12 prescribed contacts, including four reserved checks')
    if not all(s.get('contact_operator_confirmed') is True for s in samples):
        raise ValueError('Every sample requires operator confirmation of physical TCP contact')
    if manifest.get('board_type')=='chessboard' and not all('board_verified_after_contact_monotonic_s' in s for s in samples):
        raise ValueError('Chessboard contacts require a board-visibility check after each recorded contact')
    camera=np.array([s['camera_pnp_m'] for s in samples],float)
    depth=np.array([s['camera_depth_m'] for s in samples],float)
    base=np.array([s['base_tcp_m'] for s in samples],float)
    if camera.shape!=(12,3) or depth.shape!=(12,3) or base.shape!=(12,3) or not np.isfinite([camera,depth,base]).all():
        raise ValueError('Need finite XYZ camera, RGB-D and physically contacted TCP points')
    fit=np.array([s['split']=='fit' for s in samples])
    axes=np.array([validate_transform(s['robot']['T_base_tcp_candidate'])[:3,0] for s in samples])
    angles=np.rad2deg(np.arccos(np.clip(axes@axes.T,-1,1)))
    if angles.max()<30 or angles[~fit][:,~fit].max()<20:
        raise ValueError('Use at least 30 degrees of tool pitch variation overall and 20 degrees among reserved contacts')
    if np.min(np.ptp(base[:,:2],axis=0)) < .10 or np.ptp(base[:,2]) < .03:
        raise ValueError('Contacts must span >=100 mm in base X/Y and >=30 mm in height; use two supported board heights')
    for i in np.flatnonzero(~fit):
        if np.min(np.linalg.norm(base[fit]-base[i],axis=1)) < .02:
            raise ValueError('Reserved check locations must be at least 20 mm from fit contacts')
    transform,_=fit_rigid_transform(camera[fit],base[fit])
    initial_offset=np.asarray(manifest['tcp_offset_m'],float)
    offset=initial_offset.copy()
    if refine_tcp:
        # Fit only the eight fit contacts. Changing tool pitch makes TCP length
        # observable separately from camera translation; reserve checks stay out.
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        def residual(params):
            rotation=Rotation.from_rotvec(params[:3]).as_matrix()
            return (camera[fit]@rotation.T+params[3:6]-base[fit]-axes[fit]*params[6]).ravel()
        initial=np.r_[Rotation.from_matrix(transform[:3,:3]).as_rotvec(),transform[:3,3],0.]
        result=least_squares(residual,initial,bounds=(np.r_[[-np.inf]*6,-.02],np.r_[[np.inf]*6,.02]),
                             xtol=1e-12,ftol=1e-12,gtol=1e-12)
        # Normalize angular and linear parameter units before rank/conditioning.
        scaled=result.jac@np.diag([1,1,1,.25,.25,.25,.25])
        singular=np.linalg.svd(scaled,compute_uv=False)
        if not result.success or singular[-1]/singular[0]<1e-3 or abs(result.x[6])>=.0199:
            raise ValueError('TCP/camera fit is ill-conditioned or hits its 20 mm correction bound; collect more varied contacts')
        transform[:3,:3]=Rotation.from_rotvec(result.x[:3]).as_matrix()
        transform[:3,3]=result.x[3:6]
        base=base+axes*result.x[6]
        offset[0]+=result.x[6]
    old=validate_transform(manifest['rig']['T_base_front_candidate'])
    residual=np.linalg.norm(camera@transform[:3,:3].T+transform[:3,3]-base,axis=1)
    depth_residual=np.linalg.norm(depth@transform[:3,:3].T+transform[:3,3]-base,axis=1)
    old_residual=np.linalg.norm(camera@old[:3,:3].T+old[:3,3]-base,axis=1)
    passed=bool(residual.max() <= max_error_m and depth_residual.max() <= max_error_m)
    return dict(schema='krvla_base_contact_validation_v1',passed=passed,
                validation_scope='Physical TCP contact alignment in sampled workspace, conditional on measured TCP and robot FK',
                absolute_external_metrology=False,front_serial=manifest['rig']['front_serial'],
                source_rig_sha256=manifest['rig_sha256'],contacts_sha256=digest(samples),
                front_profile=manifest.get('captured_front_profile'),
                depth_corrections=manifest['rig'].get('depth_corrections',{}),
                board_type=manifest.get('board_type','charuco'),board_squares=manifest.get('board_squares',[6,6]),
                square_mm=manifest.get('square_mm',50.),
                tcp_offset_m=offset.tolist(),initial_tcp_offset_m=initial_offset.tolist(),
                tcp_refined_from_fit_contacts=bool(refine_tcp),tool_axis_span_deg=float(angles.max()),
                max_error_m=max_error_m,
                T_base_front=transform.tolist(),fit=summarize(residual[fit]),check=summarize(residual[~fit]),
                depth_all=summarize(depth_residual),old_transform_check=summarize(old_residual[~fit]),
                workspace_min_m=base.min(0).tolist(),workspace_max_m=base.max(0).tolist(),
                residuals_m=residual.tolist(),depth_residuals_m=depth_residual.tolist(),
                note='No outlier removal or refit on check contacts. Failed checks require diagnosis and a fresh check set.',
                created_utc=datetime.now(timezone.utc).isoformat())


def validated_rig(old, report):
    if not report.get('passed') or report.get('schema')!='krvla_base_contact_validation_v1':
        raise ValueError('Contact and depth checks must pass before producing a validated rig')
    if report['front_serial']!=old['front_serial']:
        raise ValueError('Camera serial mismatch')
    rig=copy.deepcopy(old)
    transform=validate_transform(report['T_base_front'])
    rig.update(T_base_front_candidate=transform.tolist(),
               T_base_side_candidate=(transform@validate_transform(old['T_front_side'])).tolist(),
               base_calibration_validated=True,calibration_validated=True,
               validated_tcp_offset_m=report['tcp_offset_m'],
               operator_confirmed_front_and_base_unchanged=True,
               base_validation=report,base_validation_sha256=digest(report),motion_enabled=False)
    if report.get('front_profile'):
        rig.setdefault('camera_profiles',{})['front']=report['front_profile']
    return rig


def check_evidence(rig):
    """Accept validated rigs only when their embedded proof matches the transform."""
    evidence=rig.get('base_validation',{})
    if (evidence.get('schema')!='krvla_base_contact_validation_v1' or evidence.get('passed') is not True
            or evidence.get('front_serial')!=rig.get('front_serial')
            or rig.get('base_validation_sha256')!=digest(evidence)
            or evidence.get('check',{}).get('count',0)<4
            or evidence.get('fit',{}).get('count',0)<8):
        raise ValueError('Validated base rig lacks matching held-out contact evidence')
    threshold=evidence.get('max_error_m',0)
    if not np.isfinite(threshold) or not 0<threshold<=.01:
        raise ValueError('Invalid base-validation error threshold')
    for key in ('fit','check','depth_all'):
        error=evidence.get(key,{}).get('max_m',float('inf'))
        if not np.isfinite(error) or not 0<=error<=threshold:
            raise ValueError('Base-validation residuals do not pass')
    if (not np.allclose(validate_transform(evidence['T_base_front']),validate_transform(rig['T_base_front_candidate']))
            or rig.get('validated_tcp_offset_m')!=evidence.get('tcp_offset_m')):
        raise ValueError('Validated transform/TCP differs from its contact evidence')
    if rig.get('depth_corrections',{}) != evidence.get('depth_corrections',{}):
        raise ValueError('Depth correction differs from the validated contact evidence')
    return evidence


def board_reference(frame, detection, board, required_ids=FIT_IDS+CHECK_IDS):
    if not detection.get('valid') or not set(required_ids)<=set(detection.get('ids',[])):
        raise ValueError(f'Need board corners {list(required_ids)} visible, with >=12 detected corners')
    pose=np.array(detection['T_camera_board'])
    points={}
    for corner in required_ids:
        pixel=detection['pixels'][detection['ids'].index(corner)]
        x,y=np.rint(pixel).astype(int)
        patch=frame['depth'][max(0,y-2):y+3,max(0,x-2):x+3]
        good=patch[np.isfinite(patch)&(patch>0)]
        if len(good)<max(10,patch.size//2):
            raise ValueError(f'Not enough board-surface depth at corner {corner}; keep arm/hands clear')
        points[str(corner)]=dict(pixel=pixel,
            camera_pnp_m=transform_point(pose,board.getChessboardCorners()[corner]).tolist(),
            camera_depth_m=frame['deproject'](pixel,float(np.median(good))))
    return dict(detection=detection,points=points)


def warm_camera(cameras, board, detector, timeout_s=30.):
    """Finish camera, detector and Qt initialization before serial deadlines."""
    deadline=time.monotonic()+timeout_s
    while time.monotonic()<deadline:
        pair=cameras.poll()
        if pair is not None:
            frame=pair[0]
            if 0<=time.monotonic()-frame['timing']['host_arrival_monotonic_s']<=.20:
                detection=estimate_board(frame,board,detector)
                # First imshow/waitKey can initialize Qt/plugins and hold the
                # GIL long enough to trip an already-running robot thread.
                cv2.imshow(WINDOW,frame['bgr'])
                if cv2.waitKey(1)&255 in (ord('q'),27):
                    raise ValueError('Calibration canceled during camera startup')
                return frame,detection
        time.sleep(.01)
    raise RuntimeError('No fresh front-camera frame within 30 seconds; robot control was not started')


def warm_feedback(io, stable_reads=8, timeout_s=10.):
    """Read-only startup qualification; no goals, torque changes or retries in motion.

    The running controller retains its 150 ms read / 200 ms cycle guards.
    Slow startup samples are recorded and reset the consecutive-fast counter.
    """
    deadline=time.monotonic()+timeout_s
    consecutive=0;timings=[]
    while time.monotonic()<deadline:
        started=time.monotonic()
        health=io.health()
        after_health=time.monotonic()
        row=io.read()
        ended=time.monotonic()
        if np.any(health['hardware_errors']):raise RuntimeError('Follower reports a hardware fault')
        if max(row['temperatures_c'])>=55:raise RuntimeError('Follower temperature guard')
        timings.append(dict(total_ms=(ended-started)*1000,health_ms=(after_health-started)*1000,
                            state_ms=(ended-after_health)*1000))
        if ended-started<=.15 and ended<=deadline:
            consecutive+=1
            if consecutive>=stable_reads:
                return dict(stable_reads=consecutive,samples=timings)
        else:consecutive=0
        time.sleep(1/30)
    raise RuntimeError(f'Motor feedback did not stabilize below 150 ms before enabling control; '
                       f'{len(timings)} read-only samples, peak {max((t["total_ms"] for t in timings),default=0):.1f} ms')


def pending_contact_verified(pending, frame, detection, reference, now):
    """Verify visibility after a chessboard contact without changing its TCP sample."""
    if now-pending['recorded_monotonic_s']>30:
        raise RuntimeError('Chessboard contact not verified within 30 s; clear the full board after S and start a new session')
    stamp=frame['timing']['host_arrival_monotonic_s']
    if stamp<=pending['recorded_monotonic_s'] or not 0<=now-stamp<=.20 or not detection.get('valid'):
        return False
    distance,angle=pose_difference(reference['detection']['T_camera_board'],detection['T_camera_board'])
    if distance>.003 or angle>1:
        raise RuntimeError('Board moved during contact verification; keep it fixed and start a new session')
    pending['sample']['board_verified_after_contact_monotonic_s']=stamp
    pending['sample']['board_check_T_camera_board']=detection['T_camera_board']
    return True


def capture(args):
    from .grasp_capture import connect_buses, RobotIO
    from .collection_control import CollectionControl
    from .recording_lease import LeaseOwner
    from .multiview_track import load_rig
    from .tracking_config import load_config
    cv2.setNumThreads(1)
    project=Path(__file__).resolve().parents[1]
    config=load_config(args.config)
    rig_path=Path(config['perception']['rig'])
    if not rig_path.is_absolute():rig_path=project/rig_path
    rig,_,_,rig_hash=load_rig(rig_path)
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    board_type=getattr(args,'board_type','charuco')
    squares=getattr(args,'board_squares',None) or [6,6]
    layout=contact_layout(squares);targets=layout['targets']
    required_ids=layout['fit_ids']+layout['check_ids']
    if board_type=='chessboard':
        board=Chessboard(squares,args.square_mm/1000)
        detector=ChessboardDetector(board)
    else:
        board,native_detector=charuco_detector(args.square_mm/1000,args.marker_mm/1000,squares)
        detector=MultiScaleCharucoDetector(native_detector,required_ids)
    manifest=dict(schema='krvla_base_contacts_v1',rig=rig,rig_sha256=rig_hash,
                  config_source=str(args.config.resolve()),
                  config_yaml=args.config.read_text(),execute=args.execute,
                  mounts_unchanged=True,tcp_offset_m=args.tcp_offset_m,
                  board_type=board_type,board_squares=list(squares),square_mm=args.square_mm,
                  pitch_tolerance_deg=PITCH_TOLERANCE_DEG,
                  marker_mm=args.marker_mm if board_type=='charuco' else None,
                  operator_reference='Earlier reported ~220 mm bend-to-tip and corrected ~160 mm rotation-centre-to-tip; approximate',
                  targets=targets,created_utc=datetime.now(timezone.utc).isoformat())
    control=cameras=None;buses={};lease=None;samples=[];references={};history=deque()
    error=None
    startup=dict(stage='initializing',opencv_threads=cv2.getNumThreads(),robot_started=False)
    write(root/'startup.json',startup)
    write(root/'manifest.json',manifest);write(root/'contacts.json',samples)
    try:
        lease=LeaseOwner(project/'scratch/real_robot/active-collection.json')
        # Camera/Qt setup and read-only bus warmup finish while no controller
        # exists, so startup delays cannot be mistaken for active feedback loss.
        startup['stage']='camera';write(root/'startup.json',startup)
        print('Initializing front camera and calibration window before robot control...',flush=True)
        cameras=CameraPair(front=rig['front_serial'],side=rig['side_serial'],front_only=True,fps=15,depth_corrections=rig.get('depth_corrections')).start()
        latest=warm_camera(cameras,board,detector)
        startup['stage']='robot_read_only';write(root/'startup.json',startup)
        buses,provenance=connect_buses(args.calibration_dir)
        io=RobotIO(buses['leader'],buses['follower'])
        startup['feedback']=warm_feedback(io)
        control=CollectionControl(io,args.execute,args.tcp_offset_m,max_relative_target=5.)
        manifest['buses']=provenance;write(root/'manifest.json',manifest);write(root/'contacts.json',samples)
        control.start()
        startup.update(stage='ready',robot_started=True);write(root/'startup.json',startup)
        image=latest[0]['bgr'].copy();reference=None;pending_contact=None
        message='Clear board, hold still; B locks board reference. T clutches leader; S saves contact; Q quits.'
        print(message,flush=True)
        print('Starts PAUSED. Hold leader and support any unpowered follower before T. T may enable holding torque at the current pose.',flush=True)
        print('Keep the gripper closed at the same fingertip-midpoint definition. Use tool pitches near 55 and 90 degrees as shown.',flush=True)
        if board_type=='chessboard':
            print(f'Plain {squares[0]}x{squares[1]} squares: expect {layout["corner_count"]} inner corners. '
                  'After S, lift the gripper clear so the full board can be verified; the contact TCP is already recorded.',flush=True)
        started=time.monotonic();last_status_write=-float('inf');last_status_print=None;last_print_time=started
        while time.monotonic()-started<args.seconds and len(samples)<len(targets):
            control.pulse()
            while True:
                try:control.pending.get_nowait()
                except queue.Empty:break
            pair=cameras.poll()
            if pair:
                frame=pair[0];detection=estimate_board(frame,board,detector);latest=(frame,detection)
                if detection.get('valid'):
                    history.append((time.monotonic(),detection))
                    while history and time.monotonic()-history[0][0]>1.:history.popleft()
                else:history.clear()
            if latest:
                frame,detection=latest;image=frame['bgr'].copy()
                for corner,pixel in zip(detection.get('ids',[]),detection.get('pixels',[])):
                    cv2.putText(image,str(corner),tuple(np.rint(pixel).astype(int)),0,.4,(0,255,255),1)
            readiness=board_readiness(history,latest[1],time.monotonic()-latest[0]['timing']['host_arrival_monotonic_s'],
                                      layout['corner_count'],required_ids,board_type)
            readiness['detection_scale']=detector.last_scale
            now=time.monotonic()
            if now-last_status_write>=1.:
                write(root/'board_status.json',readiness)
                cv2.imwrite(str(root/'preview.jpg'),image)
                last_status_write=now
            if reference is None and readiness['message']!=last_status_print and now-last_print_time>=2.:
                print(readiness['message'],flush=True)
                last_status_print=readiness['message'];last_print_time=now
            placement,corner,split=targets[len(samples)]
            requested_pitch=55 if corner in layout['low_pitch_ids'] else 90
            try:actual_pitch=tool_pitch_deg(control.snapshot())
            except ValueError:actual_pitch=None
            pitch_ok=actual_pitch is not None and abs(actual_pitch-requested_pitch)<=PITCH_TOLERANCE_DEG
            pitch_line=(f'Actual pitch: {actual_pitch:.1f} deg | target {requested_pitch} +/- {PITCH_TOLERANCE_DEG:g} | '
                        +('PITCH OK' if pitch_ok else 'ADJUST PITCH')) if actual_pitch is not None else 'Waiting for fresh pitch feedback'
            if reference is not None:
                pixel=reference['points'][str(corner)]['pixel']
                cv2.circle(image,tuple(np.rint(pixel).astype(int)),14,(0,0,255),2)
            footer=np.zeros((175,640,3),np.uint8)
            for i,line in enumerate([f'{len(samples)}/12 | height {placement+1}/2 | corner {corner} | {split.upper()} | pitch ~{requested_pitch} deg',
                    pitch_line,
                    f'{"FOLLOWING" if control.enabled else "PAUSED"} | T clutch | B board reference | S contact | Q finish',
                    readiness['message'][:95],message[:90]]):
                color=((0,255,0) if pitch_ok else (0,128,255)) if i==1 else (0,255,255)
                cv2.putText(footer,line,(5,24+30*i),0,.40,color,1)
            cv2.imshow(WINDOW,np.vstack([image,footer]))
            key=cv2.waitKey(10)&255
            if key==ord('q'):break
            commit_sample=None
            try:
                if key==ord('t'):control.toggle()
                elif key==ord('b'):
                    if placement in references:raise ValueError('Board reference already locked for these six contacts')
                    if not readiness['ready']:raise ValueError(readiness['message'])
                    candidate=board_reference(frame,detection,board,required_ids)
                    agreement=board_depth_agreement(candidate)
                    write(root/f'board_{placement}_depth_check.json',agreement)
                    if not agreement['passed']:
                        cv2.imwrite(str(root/f'board_{placement}_rejected.jpg'),frame['bgr'])
                        np.savez(root/f'board_{placement}_rejected.npz',bgr=frame['bgr'],depth_m=frame['depth'])
                        raise ValueError(f'Board/depth mismatch {agreement["max_gap_m"]*1000:.1f} mm (>10 mm). Check printed grid size/depth before recording contacts.')
                    reference=candidate;reference['depth_agreement']=agreement;references[placement]=reference
                    if board_type=='chessboard':detector.lock_reference(detection['pixels'])
                    if placement==0:
                        manifest['captured_front_profile']={k:frame['timing'][k] for k in
                            ('camera_serial','pyrealsense2_version','intrinsics','distortion_model')}
                        write(root/'manifest.json',manifest)
                    write(root/f'board_{placement}.json',reference)
                    np.savez(root/f'board_{placement}.npz',bgr=frame['bgr'],depth_m=frame['depth'])
                    message=f'Board locked. Use leader to touch corner {corner} with fingertip midpoint; hold, S.'
                elif key==ord('s'):
                    if pending_contact is not None:raise ValueError('Contact recorded. Lift gripper clear to verify the board before the next point.')
                    if reference is None:raise ValueError('Clear arm/hands and press B to lock board first')
                    if not control.stationary():raise ValueError('Hold leader, gripper and arm still for one second')
                    if latest is None or not 0<=time.monotonic()-frame['timing']['host_arrival_monotonic_s']<=.20:
                        raise ValueError('Need fresh camera image for the contact record')
                    if board_type=='charuco':
                        if not detection.get('valid'):raise ValueError('Need fresh visible board geometry around the gripper')
                        distance,angle=pose_difference(reference['detection']['T_camera_board'],detection['T_camera_board'])
                        if distance>.003 or angle>1:raise ValueError('Board moved after reference; start a fresh session')
                    row=control.snapshot()
                    actual_contact_pitch=check_contact_pitch(row,requested_pitch)
                    with control.lock:
                        positions=np.array([np.array(r['T_base_tcp_candidate'])[:3,3] for r in control.history
                                            if row['host_read_end_monotonic_s']-r['host_read_end_monotonic_s']<=.5])
                    if len(positions)<8 or np.linalg.norm(np.ptp(positions,axis=0))>.003:
                        raise ValueError('TCP still moves by >3 mm; hold the contact still before S')
                    sample=dict(placement=placement,corner_id=corner,split=split,
                                **reference['points'][str(corner)],
                                base_tcp_m=positions.mean(0).tolist(),
                                actual_pitch_deg=actual_contact_pitch,requested_pitch_deg=requested_pitch,
                                robot=row,contact_operator_confirmed=True)
                    cv2.imwrite(str(root/f'contact_{len(samples):02d}.jpg'),image)
                    if board_type=='chessboard':
                        pending_contact=dict(sample=sample,recorded_monotonic_s=time.monotonic())
                        write(root/'pending_contact.json',pending_contact)
                        message='Contact RECORDED. Lift gripper clear; waiting to verify full chessboard.'
                        print(message,flush=True)
                    else:commit_sample=sample
            except ValueError as exc:
                message=str(exc);print(message,flush=True)
            if pending_contact is not None and pending_contact_verified(pending_contact,frame,detection,reference,time.monotonic()):
                commit_sample=pending_contact['sample'];pending_contact=None
                (root/'pending_contact.json').unlink(missing_ok=True)
            if commit_sample is not None:
                samples.append(commit_sample);write(root/'contacts.json',samples)
                print(f'Saved {len(samples)}/12: height {placement+1}, corner {corner}, {split}',flush=True)
                if len(samples)==6:
                    if control.enabled:control.toggle()
                    reference=None;history.clear()
                    if board_type=='chessboard':detector.clear_reference()
                    message='Clear arm. Raise/support board by ~50 mm, keep same region; hold still, B.'
                elif len(samples)<12:
                    message=f'Next: touch corner {targets[len(samples)][1]}, hold still, then S.'
        print(f'{len(samples)}/12 contacts retained in {root}',flush=True)
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write(root/'capture_summary.json',dict(contacts=len(samples),complete=len(samples)==len(targets),error=error,
              startup_stage=startup['stage'],feedback_timing=getattr(control,'feedback_timing',None)))
        # Stop all writes before releasing either serial port.
        if control is not None:control.close()
        for bus in buses.values():
            if bus.is_connected:bus.disconnect()
        if cameras is not None:cameras.close()
        if lease is not None:lease.close()
        cv2.destroyAllWindows()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    cap=sub.add_parser('capture')
    cap.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'))
    cap.add_argument('--output',type=Path,required=True)
    cap.add_argument('--tcp-offset-m',type=float,nargs=3,required=True)
    cap.add_argument('--board-type',choices=['charuco','chessboard'],default='charuco')
    cap.add_argument('--board-squares',type=int,nargs=2,metavar=('COLUMNS','ROWS'),help='Count squares, not internal intersections')
    cap.add_argument('--square-mm',type=float,default=50.)
    cap.add_argument('--marker-mm',type=float,default=37.5)
    cap.add_argument('--seconds',type=float,default=1800.)
    cap.add_argument('--execute',action='store_true')
    cap.add_argument('--mounts-unchanged',action='store_true',required=True)
    cap.add_argument('--calibration-dir',default='/home/ailab/lerobot/.cache/calibration/aloha_default')
    fit=sub.add_parser('fit')
    fit.add_argument('--session',type=Path,required=True)
    fit.add_argument('--max-error-mm',type=float,default=5.)
    fit.add_argument('--fixed-tcp',action='store_true',help='Use an independently established TCP without refining its axial length')
    args=parser.parse_args(argv)
    if args.mode=='capture':
        args.board_squares=args.board_squares or ([10,10] if args.board_type=='chessboard' else [6,6])
        try:contact_layout(args.board_squares)
        except ValueError as exc:parser.error(str(exc))
        if (not np.isfinite([*args.tcp_offset_m,args.square_mm,args.marker_mm,args.seconds]).all()
                or not .05<=np.linalg.norm(args.tcp_offset_m)<=.25
                or not args.square_mm>0 or (args.board_type=='charuco' and not 0<args.marker_mm<args.square_mm)
                or not 0<args.seconds<=3600):
            parser.error('Invalid TCP, board dimensions or duration')
        capture(args)
    else:
        report=fit_contacts(args.session,args.max_error_mm/1000,refine_tcp=not args.fixed_tcp)
        output=args.session/'base_validation.json'
        if output.exists():raise ValueError('This capture was already evaluated; retain it and collect a fresh check set')
        write(output,report)
        if report['passed']:
            manifest=json.loads((args.session/'manifest.json').read_text())
            write(args.session/'rig-validated.json',validated_rig(manifest['rig'],report))
            if manifest.get('config_source'):
                import yaml
                config=yaml.safe_load(manifest.get('config_yaml') or Path(manifest['config_source']).read_text())
                config['perception']['rig']=str((args.session/'rig-validated.json').resolve())
                (args.session/'config-validated.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
