"""Managed EfficientTAM startup for one pick/place session; no robot I/O."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .grasp_prior import fresh_observation
from .tracking_config import load_config

PROJECT = Path(__file__).resolve().parent.parent


def resolve(path):
    value = Path(path).expanduser()
    return value if value.is_absolute() else PROJECT/value


def complete_prompts(root):
    root = Path(root)
    return all((root/(role+'-prompt.json')).is_file()
               and (root/('prompt_'+role)/'00000.jpg').is_file() for role in ('front','side'))


def save_prompts(source, destination):
    """Share collection's reference-image/mask layout; metadata is published last."""
    for role in ('front','side'):
        folder = destination/('prompt_'+role)
        folder.mkdir(parents=True,exist_ok=True)
        for name in ('00000.jpg','reference.png','mask.png'):
            src = source/('prompt_'+role)/name
            if src.exists():
                temporary = folder/(name+'.tmp')
                shutil.copyfile(src,temporary)
                temporary.replace(folder/name)
            elif (folder/name).exists():
                (folder/name).unlink()
        meta = destination/(role+'-prompt.json')
        temporary = meta.with_suffix('.tmp')
        shutil.copyfile(source/meta.name,temporary)
        temporary.replace(meta)


def stop_owned(child):
    if child is None or child.poll() is not None:return
    for sig,seconds in ((signal.SIGINT,12),(signal.SIGTERM,5),(signal.SIGKILL,5)):
        try:os.killpg(child.pid,sig)
        except ProcessLookupError:return
        try:
            child.wait(timeout=seconds)
            return
        except subprocess.TimeoutExpired:pass


def run_managed(args, run_pick):
    config = load_config(resolve(args.config))
    template = json.loads(resolve(args.template).read_text())
    section = config['perception']
    rig = resolve(section['rig'])
    if hashlib.sha256(rig.read_bytes()).hexdigest()!=template['rig_sha256']:
        raise ValueError('Pick template and perception configuration use different camera rigs')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    session = PROJECT/'scratch/real_robot/pick-place/sessions'/stamp
    session.mkdir(parents=True,exist_ok=False)
    perception = session/'perception'
    profile = resolve(args.mask_profile)
    seed = None
    if not args.new_mask:
        if complete_prompts(profile):seed = profile
        elif section.get('seed_session') and complete_prompts(resolve(section['seed_session'])):
            seed = resolve(section['seed_session'])
        else:
            # Import only the reference associated with this tape demonstration.
            manifest = resolve(template['source_episode']).parent.parent/'manifest.json'
            if manifest.exists():
                previous = json.loads(manifest.read_text()).get('observation_file')
                if previous and complete_prompts(resolve(previous).parent):seed = resolve(previous).parent
    command = [sys.executable,'-u','-m','real_robot.multiview_track',
        '--rig',str(rig),'--output',str(perception),'--mask-tracker','efficienttam',
        '--save-every','1','--seconds','1800','--preview','--device',args.device,
        '--checkpoint',str(resolve(template['efficienttam_checkpoint'])),
        '--model-config',template['efficienttam_config'],
        '--max-skew',str(section.get('max_skew_s',.035)),
        '--observation-max-age',str(section.get('observation_max_age_s',.20))]
    for role in ('front','side'):
        if role+'_depth_range_m' in section:
            command += ['--'+role+'-depth-range',*map(str,section[role+'_depth_range_m'])]
    if seed is not None:command += ['--seed-session',str(seed)]
    print('Pick/perception session:',session,flush=True)
    print('Reusing saved tape masks: '+str(seed) if seed else
          'Select the tape in FRONT and SIDE once; prompts will be saved for later runs.',flush=True)
    print('Mask library:',profile,flush=True)
    child = None
    try:
        with (session/'perception.log').open('w') as log:
            env = dict(os.environ);env.setdefault('DISPLAY',':1')
            child = subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
            observation = perception/'observation.json'
            deadline = time.monotonic()+600
            while True:
                if child.poll() is not None:
                    raise RuntimeError('Perception exited; see '+str(session/'perception.log'))
                try:
                    row = json.loads(observation.read_text())
                    fresh_observation(row,template['rig_sha256'],time.monotonic())
                    if (not args.execute or all((row.get('rgbd') or {}).get('files',{}).get(r)
                                               for r in ('front','side'))):break
                except (OSError,ValueError):pass
                if time.monotonic()>deadline:
                    raise RuntimeError('Waiting for tape perception timed out; see '+str(session/'perception.log'))
                time.sleep(.1)
            save_prompts(perception,profile)
            args.observation = str(observation)
            args.managed_perception_process = child
            if not args.output:args.output = str(session/'pick')
            (session/'session.json').write_text(json.dumps({
                'schema':'managed_pick_perception_v1','observation_file':str(observation),
                'pick_output':str(resolve(args.output)),'mask_profile':str(profile),
                'seed_source':None if seed is None else str(seed)},indent=2)+'\n')
            print('Masks ready. SPACE starts one attempt in the manipulation window.',flush=True)
            return run_pick(args)
    finally:
        stop_owned(child)
