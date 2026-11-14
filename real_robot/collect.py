"""One-command manual collection: saved masks, perception, clutched leader and DOM export."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .object_profiles import find_tape_reference, has_prompt
from .recording_lease import LeaseOwner
from .tracking_config import load_config
from .relative_target import parse_relative_target

PROJECT=Path(__file__).resolve().parent.parent
FRONT_MASK_TASKS={'striped_ball','marker','blue_bottle','orange'}


def resolve(path):
    path=Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT/path).resolve()


def prepare_upload(args):
    """Resolve the published base and inherit its pixel transform before recording."""
    from .hub_upload import authenticated_repo,verify_upload_access,verify_append_base,find_published_base
    api,args.hub_repo=authenticated_repo(args.hub_repo)
    info=verify_upload_access(api,args.hub_repo,args.public)
    if args.append_to is None:
        args.append_to=find_published_base(api,args.hub_repo,info)
        if args.append_to is not None:
            print(f'Existing shared dataset base: {args.append_to}',flush=True)
    if args.append_to is None:return None
    from .dom_dataset import validate_dataset
    print('Checking append-base metadata and video headers; full frame validation runs before upload.',flush=True)
    validate_dataset(args.append_to,decode_videos=False)
    parent=verify_append_base(args.append_to,args.hub_repo)
    previous=args.append_to/'meta/image_preprocessing.json'
    if args.image_preprocessing is None and previous.exists():args.image_preprocessing=previous
    return parent


def wait_for_perception(child, observation, timeout=600, required_views=('front','side')):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if child.poll() is not None:
            raise RuntimeError('Perception exited before startup completed; see perception.log')
        try:
            row=json.loads(Path(observation).read_text())
            if (row.get('running') and row.get('fixed_cameras_ready',True) and time.monotonic()-row.get('host_published_monotonic_s',0)<2
                    and set((row.get('rgbd') or {}).get('views',{}))==set(required_views)):
                return
        except (OSError,ValueError):
            pass
        time.sleep(.1)
    raise RuntimeError('Perception startup timed out; see perception.log')


def stop_child(child):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid,signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=12)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL)
            child.wait(timeout=5)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',default='blue_tape',choices=['blue_tape','striped_ball','marker','blue_bottle','tennis_ball','orange','ping_pong_ball','small_container'])
    parser.add_argument('--viewer', choices=['opencv','rerun'], default='opencv')
    parser.add_argument('--keyboard-source', choices=['terminal','global'], default='terminal')
    parser.add_argument('--start-delay',type=float,default=1.)
    parser.add_argument('--no-audio',action='store_true')
    parser.add_argument('--default-success', action='store_true')
    parser.add_argument('--static-object',action='store_true',help='Lock object position per attempt; zero approach velocity; automatic offline lift/release labels')
    parser.add_argument('--robot-max-relative-target', '--robot.max_relative_target', type=parse_relative_target, default=5.,help='Positive scalar or JSON list of nine limits in right-arm motor order; gripper last')
    parser.add_argument('--hub-repo', help='Upload the validated export; a bare name uses the authenticated account')
    parser.add_argument('--stream-upload',action='store_true',help='Process/upload completed episode batches in a separate background process')
    parser.add_argument('--upload-batch-episodes',type=int,default=5)
    parser.add_argument('--append-to',type=Path,help='Explicit previous export; with --hub-repo the current published base is found automatically when omitted')
    parser.add_argument('--image-preprocessing',type=Path,help='Saved crop/384px KRVLA padding settings for export; raw capture remains 640x480')
    parser.add_argument('--public', action='store_true', help='Create a public Hugging Face dataset')
    parser.add_argument('--split',default='train',choices=['train','validation','test'])
    parser.add_argument('--config',default=str(PROJECT/'real_robot/config_collection.yaml'))
    parser.add_argument('--output',type=Path,help='New session folder; automatically named by default')
    parser.add_argument('--episodes',type=int,default=20,help='Stop after this many operator-success attempts; 0 is unlimited')
    parser.add_argument('--seconds',type=float,default=1500)
    parser.add_argument('--new-mask',action='store_true',help='Select and save fresh masks for this object')
    parser.add_argument('--no-export',action='store_true',help='Keep raw session without automatic DOM export')
    parser.add_argument('--perception-only',action='store_true',help='Save/reuse masks and preview, without opening robot serial ports')
    args=parser.parse_args(argv)
    if args.stream_upload and (not args.hub_repo or args.no_export or args.perception_only):
        parser.error('--stream-upload requires --hub-repo and normal recording/export')
    if args.upload_batch_episodes<1:parser.error('--upload-batch-episodes must be positive')
    if args.append_to:
        if args.no_export or args.perception_only:parser.error('--append-to requires recording and export')
        args.append_to=resolve(args.append_to)
        if not args.hub_repo:
            from .dom_dataset import validate_dataset
            validate_dataset(args.append_to)
            previous_preprocessing=args.append_to/'meta/image_preprocessing.json'
            if args.image_preprocessing is None and previous_preprocessing.exists():
                args.image_preprocessing=previous_preprocessing
    if args.hub_repo and (args.no_export or args.perception_only):
        parser.error('--hub-repo requires a recording with export enabled')
    if args.viewer == 'rerun' and (args.new_mask or args.perception_only):
        parser.error('Prepare masks separately before Rerun collection; use the mask preparation command')
    if args.public and not args.hub_repo: parser.error('--public requires --hub-repo')
    import math
    if not math.isfinite(args.start_delay) or not 0 <= args.start_delay <= 10:
        parser.error('start-delay must be between 0 and 10 seconds')
    if args.viewer == 'rerun':
        import importlib.util
        if importlib.util.find_spec('rerun') is None: parser.error('Install rerun-sdk to use --viewer rerun')
    plan=json.loads((PROJECT/'real_robot/collection_plan.json').read_text())
    task=next((task for task in plan['tasks'] if task['id']==args.task),None)
    if task is None:parser.error('Task is missing from collection_plan.json: '+args.task)
    print(f'New collection task: {args.task}\nInstruction: {task["instruction"]}',flush=True)
    if task.get('success_criterion'):print('Success criterion: '+task['success_criterion'],flush=True)
    print('Relative-target limits: '+json.dumps(args.robot_max_relative_target),flush=True)
    append_parent=None
    if args.hub_repo:
        try:append_parent=prepare_upload(args)
        except (ValueError,PermissionError) as exc:parser.error(str(exc))
        print(f'Upload destination: https://huggingface.co/datasets/{args.hub_repo} (public={args.public})', flush=True)
        print('New-video upload permission confirmed before camera/robot startup.',flush=True)
    if args.episodes<0 or not 0<args.seconds<=1700:
        parser.error('episodes must be nonnegative; seconds must be in (0,1700]')
    config=load_config(args.config)
    rig=json.loads(resolve(config['perception']['rig']).read_text())
    if args.hub_repo == 'ailab-Clemson/KRVLA-dataset':
        from .training_release import validate_capture_rig
        validate_capture_rig(rig)
    from .calibration_acceptance import accepted_transform
    if accepted_transform(rig):
        print('Using the existing camera-to-right-base transform explicitly accepted by the operator; independent validation status is retained.',flush=True)
    elif not rig.get('base_calibration_validated',False):
        print('Object-to-robot-base calibration is UNVALIDATED. Raw collection is retained; geometric KRV training needs a held-out base-frame check.',flush=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    root=resolve(args.output) if args.output else PROJECT/'scratch/real_robot/collections'/f'{"static-" if args.static_object else ""}{args.task}-{args.split}-{stamp}'
    root.mkdir(parents=True,exist_ok=False)
    profile=PROJECT/'scratch/real_robot/object_profiles'/args.task
    if args.task in FRONT_MASK_TASKS:
        profile = profile/'on_ramp'
        if not args.new_mask and not has_prompt(profile,'front'):
            parser.error(f'Prepare the {args.task} mask first with real_robot.ramp_velocity --object {args.task} --mask-check')
    profile.parent.mkdir(parents=True,exist_ok=True)
    section=config['perception']
    lease=root/'recording.json'
    observation=root/'perception/observation.json'
    try:
        lock=LeaseOwner(PROJECT/'scratch/real_robot/active-collection.json')
    except ValueError as exc:
        print(str(exc),flush=True)
        return 2
    child=None
    uploader=None;upload_log=None
    code=1
    try:
        command=[sys.executable,'-u','-m','real_robot.multiview_track',
                 '--rig',str(resolve(section['rig'])),'--output',str(root/'perception'),
                 '--mask-tracker','efficienttam','--object-profile',str(profile),
                 '--collection-mode','--camera-fps','30','--save-every','1','--compress-rgbd',
                 '--recording-lease',str(lease),'--seconds',str(args.seconds if args.perception_only else 1800),
                 '--max-skew',str(section['max_skew_s']),
                 '--observation-max-age',str(section.get('observation_max_age_s',.20))]
        if args.viewer == 'opencv': command += ['--preview']
        if args.task in FRONT_MASK_TASKS: command += ['--position-source','front','--front-mask-only']
        for role in ('front','side'):
            if role+'_depth_range_m' in section:
                command += ['--'+role+'-depth-range',*map(str,section[role+'_depth_range_m'])]
        if args.new_mask:
            command += ['--new-mask']
        elif args.task=='blue_tape' and not all(has_prompt(profile,r) for r in ('front','side')):
            previous=find_tape_reference(PROJECT)
            if previous is not None:
                command += ['--seed-session',str(previous)]
                print(f'Importing your saved tape reference from {previous}',flush=True)
        print(f'Session: {root}\nObject mask library: {profile}',flush=True)
        if not args.new_mask and (all(has_prompt(profile,r) for r in (('front',) if args.task in FRONT_MASK_TASKS else ('front','side'))) or '--seed-session' in command):
            print('Using saved object masks; no selection needed.',flush=True)
        else:
            print('First use: select the object once in each fixed view; masks will be saved.',flush=True)
        with (root/'perception.log').open('w') as log:
            env=dict(os.environ)
            env.setdefault('DISPLAY',':1')
            child=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            wait_for_perception(child,observation)
            if args.perception_only:
                print('Camera-only preview; Q in perception closes it. Robot ports are unopened.',flush=True)
                code=child.wait()
            else:
                if args.stream_upload:
                    from .stream_upload import save
                    settings=dict(repo=args.hub_repo,public=args.public,batch_episodes=args.upload_batch_episodes,
                        image_preprocessing=None if args.image_preprocessing is None else str(resolve(args.image_preprocessing)))
                    save(root/'stream-upload/config.json',settings)
                    upload_log=(root/'stream-upload/worker.log').open('a')
                    upload_env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                    uploader=subprocess.Popen([sys.executable,'-u','-m','real_robot.stream_upload','--session',str(root)],
                        cwd=PROJECT,env=upload_env,stdout=upload_log,stderr=subprocess.STDOUT,start_new_session=True)
                    print(f'Background processing/upload every {args.upload_batch_episodes} successes. Status: {root/"stream-upload/status.json"}',flush=True)
                from .dynamic_capture import main as capture
                os.environ.setdefault('DISPLAY',':1')
                print('SPACE: start. ENTER: stop/save success. X: stop/save failure. Q: finish/export/upload. T: pause.' if args.default_success else 'N: record and clutch automatically. V: success. X: failed. Q: finish/export. T: pause.',flush=True)
                print('Lift/release labels are derived automatically at export from measured gripper/TCP motion in successful episodes; L/R are optional. X if nudged before grasp.' if args.static_object else 'L: mark lift; R: mark release. Markers help wrist-video review of held-object intervals.',flush=True)
                code=capture(['--task',args.task,'--split',args.split,'--config',str(resolve(args.config)),
                              '--output',str(root/'recording'),'--observation',str(observation),
                              '--recording-lease',str(lease),'--camera-fps','30','--clutched','--execute',
                              '--heading-deg','180' if args.task=='blue_tape' else '0',
                              '--tcp-offset-m',*map(str,rig.get('validated_tcp_offset_m',[.16,0,0])),
                              '--max-successes',str(args.episodes),'--seconds',str(args.seconds),
                              '--start-delay',str(args.start_delay),*(['--no-audio'] if args.no_audio else []),
                              '--viewer',args.viewer,'--keyboard-source',args.keyboard_source,'--robot-max-relative-target',str(args.robot_max_relative_target),
                              *(['--default-success'] if args.default_success else []),
                              *(['--static-object'] if args.static_object else [])])
        stop_child(child)
        if code!=0 and (root/'perception/summary.json').exists():
            detail=json.loads((root/'perception/summary.json').read_text()).get('error')
            if detail:
                print('Perception error: '+str(detail),flush=True)
        if not args.stream_upload and not args.no_export and not args.perception_only and (root/'recording/manifest.json').exists():
            successful=[p for p in (root/'recording/episodes').glob('*.json')
                        if json.loads(p.read_text()).get('outcome')=='success']
            if not successful:
                print('Raw session saved; no successful attempts to export.',flush=True)
                return code
            from .dom_dataset import export_sessions
            try:
                report=export_sessions([root/'recording'],root/'dom',image_preprocessing=args.image_preprocessing)
                print(f"DOM export: {root/'dom'} | {report['episodes']} accepted episodes",flush=True)
                from .dataset_quality import process as audit_quality
                audit_quality(root/'dom',root/'quality-audit',repair=True)
                print(f"Tracking/phase audit: {root/'quality-audit/quality_audit.json'}",flush=True)
                upload_root=root/'dom'
                if args.append_to and not args.hub_repo:
                    from .merge_datasets import merge_datasets
                    upload_root=root/'dom-combined'
                    combined=merge_datasets([args.append_to,root/'dom'],upload_root)
                    print(f"Combined export: {combined['episodes']} episodes, {combined['frames']} frames | {upload_root}",flush=True)
                if args.hub_repo:
                    from .hub_upload import upload_dataset
                    try:
                        uploaded=upload_dataset(upload_root,args.hub_repo,args.public,append=True,expected_parent=append_parent)
                        print(f"Uploaded: {uploaded['url']} | commit {uploaded['commit']}",flush=True)
                        if args.hub_repo == 'ailab-Clemson/KRVLA-dataset':
                            from .training_release import sync as sync_training
                            try:
                                sync_training(args.hub_repo,uploaded['dataset_path'],uploaded['commit'])
                            except Exception as exc:
                                print(f'Raw upload succeeded, but training branch refresh failed: {type(exc).__name__}: {exc}',flush=True)
                                print(f'Retry: {sys.executable} -m real_robot.training_release --repo {args.hub_repo}',flush=True)
                                code=1
                    except Exception as exc:
                        print(f'Export saved locally, but upload failed: {type(exc).__name__}: {exc}',flush=True)
                        print(f'Retry: {sys.executable} -m real_robot.hub_upload --dataset {upload_root} --repo {args.hub_repo}' + (' --public' if args.public else '') + ' --append',flush=True)
                        code=1
            except ValueError as exc:
                print(f'Raw data saved. Export needs review: {exc}\nSee {root}/dom.partial/failure.json',flush=True)
                code=1
    except (Exception,KeyboardInterrupt) as exc:
        print(f'Collection stopped: {type(exc).__name__}: {exc}\nSession/log: {root}',flush=True)
        if (root/'perception.log').exists():
            print('\n'.join((root/'perception.log').read_text(errors='replace').splitlines()[-12:]),flush=True)
        code=1
    finally:
        stop_child(child)
        if uploader is not None:
            (root/'stream-upload/FINISH').touch()
            print('Recording stopped. Finishing queued processing/uploads; raw files remain local.',flush=True)
            try:
                result=uploader.wait()
                status_path=root/'stream-upload/status.json'
                status=json.loads(status_path.read_text()) if status_path.exists() else {}
                if result or status.get('state')!='completed':
                    code=1
                    print('Background processing/upload needs retry: '+str(status.get('error') or 'see worker.log'),flush=True)
                    print(f'{sys.executable} -m real_robot.stream_upload --session {root}',flush=True)
                else:
                    print(f'Uploaded {len(status.get("uploaded_episode_ids",[]))} episodes from this session.',flush=True)
                    if args.hub_repo=='ailab-Clemson/KRVLA-dataset' and status.get('last_receipt'):
                        from .training_release import sync as sync_training
                        receipt=status['last_receipt']
                        sync_training(args.hub_repo,receipt['dataset_path'],receipt['commit'])
            except (Exception,KeyboardInterrupt) as exc:
                code=1
                print(f'Upload queue retained: {type(exc).__name__}: {exc}. Resume with {sys.executable} -m real_robot.stream_upload --session {root}',flush=True)
            finally:
                if upload_log is not None:upload_log.close()
        lock.close()
        try:
            from .experiment_tracker import refresh
            refresh()
            print('Experiment ledger and paper tables refreshed.',flush=True)
        except Exception as exc:
            print(f'Tracker refresh needs review: {exc}. Raw session is preserved.',flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
