"""Prepare or launch best/last completed static-orange SmolVLA evaluation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from real_robot.family_live import select_family_checkpoint

DEFAULT_CAMPAIGN=PROJECT/'scratch/real_robot/finetuning-runs/static-orange-smolvla-pretrained-20261113/campaign.json'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def build_command(campaign,selection,execute=False,max_trials=1,start_pose=None,object_motion='auto',trial_seconds=30.,mask_mode='auto',rtc='guided',rtc_execution_horizon=10,rtc_max_guidance_weight=10.):
    if not math.isfinite(trial_seconds) or trial_seconds<=0:raise ValueError('trial-seconds must be finite and positive')
    if max_trials<0:raise ValueError('max-trials must be nonnegative; 0 is unlimited')
    if mask_mode not in ('auto','manual','reuse'):raise ValueError('Unknown mask mode')
    if rtc not in ('off','guided') or not 1<=rtc_execution_horizon<=20 or not math.isfinite(rtc_max_guidance_weight) or rtc_max_guidance_weight<=0:raise ValueError('Invalid RTC settings')
    checkpoint,record,settings=select_family_checkpoint(campaign,'smolvla_baseline',selection)
    recipe=settings['training_recipe']
    if recipe.get('instruction_mode')!='recorded':raise ValueError('Expected the unchanged recorded training prompt')
    if recipe.get('prompt')!='Pick up the rolling orange and place it in the box.':raise ValueError('Unexpected training prompt')
    command=[str(PROJECT/'.venv/bin/python'),'-u','-m','real_robot.evaluate',
        '--family-campaign',str(Path(campaign).resolve()),'--model','smolvla_baseline',
        '--checkpoint',selection,'--expected-checkpoint-sha256',digest(checkpoint),
        '--task','orange','--record-inputs','--max-trials',str(max_trials),'--trial-seconds',str(trial_seconds),
        '--velocity-innovation','off','--object-motion',object_motion,
        '--rtc',rtc,'--rtc-execution-horizon',str(rtc_execution_horizon),'--rtc-max-guidance-weight',str(rtc_max_guidance_weight)]
    if record['object_inputs_used_by_policy']:
        if mask_mode=='auto':command+=['--auto-mask']
        elif mask_mode=='manual':command+=['--new-mask']
    if start_pose is not None:command+=['--start-pose',str(Path(start_pose).resolve())]
    if execute:command+=['--execute']
    return command,record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('selection',choices=['best','last'])
    p.add_argument('--campaign',type=Path,default=DEFAULT_CAMPAIGN)
    p.add_argument('--execute',action='store_true',help='Enable the usual supervised SPACE-at-READY trial')
    p.add_argument('--object-motion',choices=['auto','static','moving'],default='auto')
    p.add_argument('--trial-seconds',type=float,default=30.,help='Per-trial timeout in seconds')
    p.add_argument('--mask-mode',choices=['auto','manual','reuse'],default='auto')
    p.add_argument('--max-trials',type=int,default=1,help='Maximum trials; 0 is unlimited');p.add_argument('--start-pose',type=Path)
    p.add_argument('--rtc',choices=['off','guided'],default='guided',help='Guided real-time chunking; off for native-sampler comparison')
    p.add_argument('--rtc-execution-horizon',type=int,default=10)
    p.add_argument('--rtc-max-guidance-weight',type=float,default=10.)
    p.add_argument('--check',action='store_true',help='Read selection and print the command; no cameras, GPU or robot access')
    a=p.parse_args()
    if a.max_trials<0:p.error('max-trials must be nonnegative')
    try:command,record=build_command(a.campaign,a.selection,a.execute,a.max_trials,a.start_pose,a.object_motion,a.trial_seconds,a.mask_mode,a.rtc,a.rtc_execution_horizon,a.rtc_max_guidance_weight)
    except (ValueError,OSError,KeyError) as exc:
        print('Evaluation not ready: '+str(exc),file=sys.stderr);return 2
    print(json.dumps(dict(selection=a.selection,epoch=record['epoch'],weights=record.get('weights','raw'),object_tracking_required=record['object_inputs_used_by_policy'],checkpoint=record['source']),indent=2),flush=True)
    print(shlex.join(command),flush=True)
    if not a.check:os.execv(command[0],command)
    return 0


if __name__=='__main__':sys.exit(main())
