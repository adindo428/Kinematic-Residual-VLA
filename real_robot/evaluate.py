"""Launch a paused learned-policy trial with continuous inference."""
import argparse
import hashlib
from datetime import datetime,timezone
import json
import math
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time
from .recording_lease import LeaseOwner
from .tracking_config import load_config
from .object_estimation import add_object_estimation_arguments,resolve_model
from .policy_stream import live_object,live_camera,LiveObjectEstimator
from .camera_transport import validate_transport,resolve_front_color_format
from .policy_trajectory import learned_governor_enabled
from .family_live import FAMILIES,select_family_checkpoint,family_object_inputs_required

PROJECT=Path(__file__).resolve().parents[1]
DEFAULT_RUN=PROJECT/'scratch/real_robot/finetuning-runs/action-v5-fullsampler-20261110'
DEFAULT_FAMILY_CAMPAIGN=PROJECT/'scratch/real_robot/family-finetuning-v6/campaign.json'


def read_run_contract(run):
    run=Path(run)
    path=run/'launch_contract.json'
    if not path.is_file() and run.name=='dom-runs':path=run.parent/'dom-contract.json'
    return json.loads(path.read_text())


def checkpoint_selection(value):
    if value not in ('best','last') and not re.fullmatch(r'epoch-[0-9]{3,}',value):
        raise argparse.ArgumentTypeError('Use best, last, or an explicit completed epoch such as epoch-010')
    return value


def prepare_checkpoint(run, family, selection):
    run=Path(run).resolve();training=run/family
    if selection=='best':
        source=Path(json.loads((training/'best_checkpoint.json').read_text())['path']).resolve()
    elif re.fullmatch(r'epoch-[0-9]{3,}',selection):
        source=(training/selection).resolve()
    elif selection=='last':
        completed=[]
        for candidate in training.glob('epoch-*'):
            provenance=candidate/'provenance.json'
            if provenance.is_file():
                record=json.loads(provenance.read_text())
                if record.get('completed_epoch'):completed.append((record['epoch'],candidate))
        if not completed:raise ValueError('No completed training epochs available')
        source=max(completed,key=lambda item:item[0])[1].resolve()
    else:raise ValueError('Unknown checkpoint selection')
    if source.parent!=training.resolve():raise ValueError('Checkpoint is outside the selected model training directory')
    record=json.loads((source/'provenance.json').read_text())
    if not record.get('completed_epoch'):raise ValueError('Evaluation requires a completed-epoch checkpoint')
    if selection.startswith('epoch-') and record['epoch']!=int(selection[6:]):raise ValueError('Explicit checkpoint epoch mismatch')
    config=json.loads((source/'config.json').read_text())
    contract=read_run_contract(run)
    config['delta_timestamps']={'observation':contract['observation_history'],'action':list(range(config['chunk_size']))}
    config['use_checkpoint_normalization']=True
    destination=run/'evaluation'/(family if selection=='best' else family+'-'+selection)
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    weights=destination/'model.safetensors'
    if weights.is_symlink() and weights.resolve()!=source/'model.safetensors':weights.unlink()
    if not weights.exists():weights.symlink_to(source/'model.safetensors')
    if weights.resolve()!=source/'model.safetensors':raise ValueError('Evaluation weight path does not match selected checkpoint')
    (destination/'selection.json').write_text(json.dumps(dict(selection=selection,source=str(source),epoch=record['epoch'],
        validation_loss=record['validation_loss']),indent=2)+'\n')
    return destination,record



def policy_selection(run,family,selection,family_campaign):
    if family in FAMILIES:
        checkpoint,selected,contract=select_family_checkpoint(family_campaign,family,selection)
        contract=contract['training_recipe']
    else:
        checkpoint,selected=prepare_checkpoint(run,family,selection)
        contract=read_run_contract(run)
    spec=Path(contract['local_verified_dataset'])/'meta/image_preprocessing.json'
    return checkpoint,selected,spec



def policy_object_inputs_required(model,checkpoint,velocity_conditioning=None):
    if model in FAMILIES:return family_object_inputs_required(model,velocity_conditioning)
    if model!='dom':return True
    config=json.loads((Path(checkpoint)/'config.json').read_text())
    return bool(config.get('condition_on_object_velocity') or config.get('enable_residual_flow'))


def worker_command(model,checkpoint,campaign,exchange,spec,expected_sha256=None,rtc='off',rtc_execution_horizon=10,rtc_max_guidance_weight=10.):
    if model in FAMILIES:
        command=[str(PROJECT/'workstation/family-python.sh'),'-u',str(PROJECT/'workstation/real_family_policy_worker.py'),
                 '--campaign',str(Path(campaign).resolve()),'--family',model]
        if expected_sha256:command+=['--expected-sha256',expected_sha256]
    else:
        command=[str(PROJECT/'workstation/finetune-python.sh'),'-u',str(PROJECT/'workstation/real_policy_worker.py')]
    if rtc!='off':
        if not model.startswith('smolvla_'):raise ValueError('Guided RTC requires the native SmolVLA family')
        command+=['--rtc',rtc,'--rtc-execution-horizon',str(rtc_execution_horizon),'--rtc-max-guidance-weight',str(rtc_max_guidance_weight)]
    return command+['--checkpoint',str(checkpoint.resolve()),'--exchange',str(exchange),'--image-spec',str(spec.resolve())]


def weight_digest(checkpoint):
    path=checkpoint if checkpoint.is_file() else checkpoint/'model.safetensors'
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def stop(child):
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGINT)
        try:child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=3)


def perception_ready(root,rig_sha,object_tracking_required=True):
    try:
        if (root/'MASK_PENDING').exists() or (root/'RESEED').exists():return False
        if not (root/'policy_front.npz').exists():return False
        record=json.loads((root/'observation.json').read_text())
        if object_tracking_required:live_object(record,time.monotonic(),rig_sha)
        else:live_camera(record,time.monotonic(),rig_sha)
        return True
    except (OSError,ValueError,KeyError,TypeError):return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['dom','krv100','domv',*FAMILIES],default='krv100')
    p.add_argument('--checkpoint',type=checkpoint_selection,default='best',help='best, last, or an explicit completed epoch such as epoch-010')
    p.add_argument('--run',type=Path,default=DEFAULT_RUN)
    p.add_argument('--family-campaign',type=Path,default=DEFAULT_FAMILY_CAMPAIGN)
    p.add_argument('--expected-checkpoint-sha256',help='Refuse a checkpoint different from the frozen experiment selection')
    p.add_argument('--config',type=Path,default=PROJECT/'real_robot/config_policy_eval.yaml')
    p.add_argument('--task',choices=['striped_ball','marker','orange'],default='striped_ball')
    p.add_argument('--auto-mask',action='store_true',help='Automatically propose the live target from its saved appearance profile')
    p.add_argument('--new-mask',action='store_true',help='Select a fresh front-camera task-object mask during startup')
    p.add_argument('--output',type=Path);p.add_argument('--seconds',type=float,default=600)
    p.add_argument('--trial-seconds',type=float,default=30)
    p.add_argument('--rtc',choices=['off','guided'],default='off')
    p.add_argument('--rtc-execution-horizon',type=int,default=10)
    p.add_argument('--rtc-max-guidance-weight',type=float,default=10.)
    p.add_argument('--velocity-innovation',choices=['off','on'],default=None,help='Paper execution-time correction; use on for matched DOMV/KRV trials')
    add_object_estimation_arguments(p)
    p.add_argument('--max-trials',type=int,default=0,help='Exit after this many trials; 0 is unlimited')
    p.add_argument('--pose-smoothing-ms',type=float,default=80.,help='Short quintic EE-target blend; 0 disables')
    p.add_argument('--hold-target-check',action='store_true',help='Hold the existing commanded EE pose under the same asynchronous workload; diagnostic only')
    p.add_argument('--record-inputs',action='store_true',help='Save lossless model inputs for subsequent checkpoint comparisons')
    p.add_argument('--start-pose',type=Path,help='Reuse controller/start_pose.json from another session for a matched starting pose')
    p.add_argument('--start-position-tolerance-mm',type=float,default=10.,help='Settled TCP start tolerance (default 10 mm)')
    p.add_argument('--start-joint-tolerance-deg',type=float,default=2.,help='Measured per-joint start tolerance (default 2 degrees)')
    p.add_argument('--ik-method',choices=['bounded_dls','uniform_dls'],default='bounded_dls')
    p.add_argument('--geometry-profile',choices=['grasp','hover'],default='grasp',help='grasp allows fingertip/table contact while retaining other geometry checks')
    p.add_argument('--joint-governor',choices=['auto','on','off'],default='auto',help='Acceleration/jerk governor; off is preview-only for learned policies')
    p.add_argument('--joint-acceleration-deg-s2',type=float)
    p.add_argument('--joint-jerk-deg-s3',type=float)
    p.add_argument('--camera-fps',type=int,choices=[30],default=30,help='VLA visual capture stays at 30FPS; policy actions remain 25Hz')
    p.add_argument('--front-color-format',choices=['auto','bgr8','yuyv'],default='auto',help='auto selects packed YUYV on USB 2 and BGR8 on USB 3; perception always receives BGR')
    p.add_argument('--motion-profile',choices=['hardware','smooth','auto','pilot','standard'],default='hardware',
        help='hardware: motor read-back velocity limits with no extra EE/angular/joint caps; smooth/auto/pilot/standard retain historical software caps')
    p.add_argument('--execute',action='store_true',help='SPACE in the evaluation window may begin robot motion; starts paused')
    args=p.parse_args()
    if args.rtc!='off' and not args.model.startswith('smolvla_'):p.error('Guided RTC requires native SmolVLA')
    if not 1<=args.rtc_execution_horizon<=20 or not math.isfinite(args.rtc_max_guidance_weight) or args.rtc_max_guidance_weight<=0:p.error('Invalid RTC settings')
    try:object_estimator=LiveObjectEstimator(args.object_prediction_max_gap_s,args.object_prediction_max_displacement_m,motion_model=resolve_model(args.object_occlusion_model,args.object_motion))
    except ValueError as exc:p.error(str(exc))
    try:learned_governor_enabled(args.motion_profile,args.joint_governor,args.execute)
    except ValueError as exc:p.error(str(exc))
    for value in [args.joint_acceleration_deg_s2,args.joint_jerk_deg_s3]:
        if value is not None and (not math.isfinite(value) or value<=0):p.error('Acceleration/jerk bounds must be finite and positive')
    if not all(math.isfinite(v) and v>0 for v in [args.start_position_tolerance_mm,args.start_joint_tolerance_deg]):
        p.error('start tolerances must be finite and positive')
    config=load_config(args.config,manipulation=True)
    if args.max_trials<0:p.error('max-trials must be nonnegative')
    tasks=json.loads((PROJECT/'real_robot/collection_plan.json').read_text())['tasks']
    task=next(task for task in tasks if task['id']==args.task)
    checkpoint,selected,spec=policy_selection(args.run,args.model,args.checkpoint,args.family_campaign)
    native_contract=None
    if args.model not in FAMILIES:
        from .native_policy_contract import verify_native_contract,training_instruction
        native_contract=verify_native_contract(checkpoint,args.model,spec.parent.parent,read_run_contract(args.run))
        task=dict(task,instruction=training_instruction(spec.parent.parent,args.task,mode=read_run_contract(args.run).get('instruction_mode','generated')))
    if args.model in FAMILIES:
        family_recipe=json.loads(Path(args.family_campaign).read_text())['training_recipe']
        if family_recipe.get('instruction_mode')=='recorded':
            from .native_policy_contract import training_instruction
            task=dict(task,instruction=training_instruction(family_recipe['local_verified_dataset'],args.task,mode='recorded'))
    if args.velocity_innovation is None:
        args.velocity_innovation=(read_run_contract(args.run).get('execution_policy',{}).get('velocity_innovation',{}).get(args.model,'off') if args.model not in FAMILIES else 'off')
    gate_settings=json.loads((checkpoint/'config.json').read_text()) if args.model not in FAMILIES else {}
    args.object_occlusion_model=resolve_model(args.object_occlusion_model,args.object_motion,spec.parent.parent,args.task)
    object_estimator=LiveObjectEstimator(args.object_prediction_max_gap_s,args.object_prediction_max_displacement_m,motion_model=args.object_occlusion_model)
    object_tracking_required=policy_object_inputs_required(args.model,checkpoint,selected.get('velocity_conditioning'))
    if args.velocity_innovation=='on' and not object_tracking_required:p.error('Velocity innovation requires tracked object inputs')
    checksum=weight_digest(checkpoint)
    if args.expected_checkpoint_sha256 and checksum!=args.expected_checkpoint_sha256:p.error('Frozen experiment checkpoint changed')
    from .camera import enumerate_cameras
    rig=json.loads((PROJECT/config['perception']['rig']).read_text())
    if native_contract is not None:
        from .native_policy_contract import verify_live_geometry
        native_contract['geometry']=verify_live_geometry(spec.parent.parent,rig,read_run_contract(args.run))
    try:
        transport=validate_transport(enumerate_cameras(),dict(front=rig['front_serial'],wrist='218622278247'),
                                     fps=args.camera_fps,color_format=args.front_color_format)
        front_color_format=resolve_front_color_format(transport['front'],args.front_color_format)
    except ValueError as exc:p.error(str(exc))
    profile=PROJECT/'scratch/real_robot/object_profiles'/args.task/'on_ramp'
    if object_tracking_required and not (profile/'front-prompt.json').exists():args.new_mask=True
    rig_sha=hashlib.sha256((PROJECT/config['perception']['rig']).read_bytes()).hexdigest()
    root=args.output or PROJECT/'scratch/real_robot/evaluations'/(args.task+'-'+args.model+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    root=root.resolve();root.mkdir(parents=True,exist_ok=False)
    if native_contract is not None:(root/'native_contract.json').write_text(json.dumps(native_contract,indent=2)+'\n')
    (root/'camera_transport.json').write_text(json.dumps(transport,indent=2)+'\n')
    print(f'Front capture: {front_color_format}, 30FPS, USB {transport["front"]["usb_type_descriptor"]}; policy control 25Hz; side stream disabled.',flush=True)
    (root/'exchange').mkdir();children=[];streams=[];lease=None
    try:
        lease=LeaseOwner(PROJECT/'scratch/real_robot/active-collection.json')
        env=dict(os.environ,DISPLAY=os.environ.get('DISPLAY',':1'))
        commands=[('worker',worker_command(args.model,checkpoint,args.family_campaign,root/'exchange',spec,checksum,rtc=args.rtc,rtc_execution_horizon=args.rtc_execution_horizon,rtc_max_guidance_weight=args.rtc_max_guidance_weight)),
            ('perception',[sys.executable,'-u','-m','real_robot.multiview_track','--rig',str(PROJECT/config['perception']['rig']),
            '--output',str(root/'perception'),'--mask-tracker','efficienttam','--object-profile',str(profile),
            '--collection-mode','--position-source','front','--front-mask-only','--front-camera-only','--policy-feed','--save-every','0',
            '--preview','--interactive-reseed',
            '--camera-fps',str(args.camera_fps),'--front-color-format',front_color_format,
            '--seconds',str(args.seconds+120),'--observation-max-age','.2'])]
        if not object_tracking_required:
            commands[1]=('perception',[sys.executable,'-u','-m','real_robot.policy_camera','--rig',str(PROJECT/config['perception']['rig']),
                '--output',str(root/'perception'),'--seconds',str(args.seconds+120),'--camera-fps',str(args.camera_fps),
                '--front-color-format',front_color_format,'--preview'])
            print(f"{'DOM' if args.model=='dom' else args.model} uses RGB + robot state only; object tracker and mask selection are disabled.",flush=True)
        for name,command in commands:
            if name=='worker' and args.record_inputs:command+=['--record-inputs']
            if name=='perception' and args.auto_mask and object_tracking_required:command+=['--auto-mask']
            if name=='perception' and args.new_mask and object_tracking_required:command+=['--new-mask']
            log=(root/(name+'.log')).open('w');streams.append(log)
            child=subprocess.Popen(command,cwd=PROJECT,env=dict(env,HF_HUB_OFFLINE='1') if name=='worker' else env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(child)
        deadline=time.monotonic()+100;last_prompt=None
        while not ((root/'exchange/worker.json').exists() and perception_ready(root/'perception',rig_sha,object_tracking_required)):
            for name,child in zip(['worker','perception'],children):
                if child.poll() is not None:raise RuntimeError(f'{name} exited ({child.returncode}); see {root/(name+".log")}')
            prompt_path=root/'perception/prompt_status.json'
            if prompt_path.exists():
                prompt=json.loads(prompt_path.read_text())
                if prompt!=last_prompt:
                    print('Mask setup: '+prompt['state']+' | '+str(prompt.get('reason') or 'draw a box around the task object and press ENTER'),flush=True)
                    last_prompt=prompt
                if prompt['state']=='awaiting_selection':deadline=time.monotonic()+100
            if time.monotonic()>deadline:raise RuntimeError('Camera or inference startup timeout; inspect session logs')
            time.sleep(.1)
        (root/'perception/EVAL_STARTED').touch()
        controller=[sys.executable,'-u','-m','real_robot.policy_eval','--config',str(args.config.resolve()),
            '--instruction',task['instruction'],
            '--velocity-innovation',args.velocity_innovation,
            '--rtc',args.rtc,
            '--residual-phase-gate-mode',gate_settings.get('residual_phase_gate_mode','constant'),
            '--residual-phase-gate-config-json',json.dumps(gate_settings.get('residual_phase_gate_config',{})),
            '--object-inputs','required' if object_tracking_required else 'unused',
            '--policy-input-profile','single_frame' if args.model in FAMILIES else 'two_frame',
            '--ik-method',args.ik_method,
            '--geometry-profile',args.geometry_profile,
            '--object-occlusion-model',args.object_occlusion_model,
            '--object-prediction-max-gap-s',str(args.object_prediction_max_gap_s),
            '--object-prediction-max-displacement-m',str(args.object_prediction_max_displacement_m),
            '--joint-governor',args.joint_governor,
            '--perception',str(root/'perception'),'--exchange',str(root/'exchange'),'--output',str(root/'controller'),
            '--seconds',str(args.seconds),'--trial-seconds',str(args.trial_seconds),
            '--motion-profile',args.motion_profile,'--max-trials',str(args.max_trials),
            '--pose-smoothing-ms',str(args.pose_smoothing_ms),
            '--start-position-tolerance-mm',str(args.start_position_tolerance_mm),
            '--start-joint-tolerance-deg',str(args.start_joint_tolerance_deg)]
        if args.execute:controller+=['--execute']
        if args.joint_acceleration_deg_s2 is not None:controller+=['--joint-acceleration-deg-s2',str(args.joint_acceleration_deg_s2)]
        if args.joint_jerk_deg_s3 is not None:controller+=['--joint-jerk-deg-s3',str(args.joint_jerk_deg_s3)]
        if args.hold_target_check:controller+=['--hold-target-check']
        if args.start_pose is not None:controller+=['--start-pose',str(args.start_pose.resolve())]
        log=(root/'controller.log').open('w');streams.append(log)
        child=subprocess.Popen(controller,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child)
        record=dict(output=str(root),model=args.model,task=args.task,instruction=task['instruction'],checkpoint=str(checkpoint.resolve()),
            checkpoint_selection=args.checkpoint,checkpoint_epoch=selected['epoch'],checkpoint_sha256=checksum,
            selected_weights=selected.get('weights','raw'),policy_observation_offsets=[0] if args.model in FAMILIES else [-2,0],
            object_inputs_used_by_policy=object_tracking_required,object_tracking_enabled=object_tracking_required,
            native_contract=native_contract,
            velocity_conditioning=native_contract['velocity_conditioning'] if native_contract else selected.get('velocity_conditioning'),object_occlusion_prediction=object_estimator.settings if object_tracking_required else None,
            front_camera_fps=args.camera_fps,front_color_format=front_color_format,
            usb2_compatibility=str(transport['front']['usb_type_descriptor']).split('.')[0]=='2',side_stream_enabled=False,
            hold_target_check=args.hold_target_check,
            rtc=args.rtc,rtc_execution_horizon=args.rtc_execution_horizon,rtc_max_guidance_weight=args.rtc_max_guidance_weight,async_inference=True,
            record_inputs=args.record_inputs,velocity_innovation=args.velocity_innovation,automatic_mask=bool(args.auto_mask and object_tracking_required),automatic_mask_requested=args.auto_mask,
            reset_to_session_start=True,start_pose_source=None if args.start_pose is None else str(args.start_pose.resolve()),
            start_position_tolerance_mm=args.start_position_tolerance_mm,start_joint_tolerance_deg=args.start_joint_tolerance_deg,
            ik_method=args.ik_method,
            ee_workspace_enforced=False,geometry_profile=args.geometry_profile,
            software_velocity_caps=args.motion_profile!='hardware',
            joint_governor=args.joint_governor,
            joint_acceleration_deg_s2=args.joint_acceleration_deg_s2,
            joint_jerk_deg_s3=args.joint_jerk_deg_s3,
            motion_profile=args.motion_profile,pose_smoothing_ms=args.pose_smoothing_ms,
            controller_pid=child.pid,execute_enabled=args.execute,starts_paused=True)
        (root/'session.json').write_text(json.dumps(record,indent=2)+'\n')
        pointer=PROJECT/'scratch/real_robot/evaluations/latest.json'
        pointer.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record,indent=2),flush=True)
        while child.poll() is None:
            for name,dependency in zip(['worker','perception'],children[:2]):
                if dependency.poll() is not None:
                    (root/'controller/STOP').touch()
                    raise RuntimeError(f'{name} stopped; ending evaluation')
            time.sleep(.1)
        if child.returncode:
            detail=''
            status_path=root/'controller/status.json'
            if status_path.exists():detail=json.loads(status_path.read_text()).get('status','')
            raise RuntimeError(f'Controller exited {child.returncode}: {detail}; see {root/"controller.log"}')
    finally:
        if (root/'controller').exists():(root/'controller/STOP').touch()
        (root/'exchange/STOP').touch()
        for child in reversed(children):stop(child)
        for stream in streams:stream.close()
        if lease is not None:lease.close()
        if any((root/'controller').glob('trial-*.json')):
            try:
                with (root/'plots.log').open('w') as plot_log:
                    plotted=subprocess.run([str(PROJECT/'workstation/finetune-python.sh'),
                        str(PROJECT/'workstation/plot_policy_eval.py'),'--session',str(root)],cwd=PROJECT,
                        env=dict(os.environ,MPLCONFIGDIR='/tmp/krvla-mpl'),stdout=plot_log,stderr=subprocess.STDOUT,timeout=60)
                print('EE plots: '+str(root/'plots') if plotted.returncode==0 else 'Plotting failed; see '+str(root/'plots.log'),flush=True)
            except (OSError,subprocess.TimeoutExpired) as exc:print('Plotting did not complete: '+str(exc),flush=True)


if __name__=='__main__':main()
