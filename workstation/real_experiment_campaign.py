"""Freeze trained checkpoints, preflight saved inputs, and launch supervised trials."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from real_robot.family_live import select_family_checkpoint
from real_robot.object_estimation import resolve_model,DEFAULT_MAX_GAP_S,DEFAULT_MAX_DISPLACEMENT_M

DOM_RUN=PROJECT/'scratch/real_robot/finetuning-runs/action-v5-fullsampler-20261110'
FAMILY_CAMPAIGN=PROJECT/'scratch/real_robot/family-finetuning-v6/campaign.json'
MODELS=['dom','domv','krv100','smolvla_baseline','smolvla_kr']


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def schedule(models,tasks,conditions,seed=12345):
    rng=random.Random(seed);rows=[]
    for phase,n in [('pilot',3),('scored',20)]:
        for task in tasks:
            for condition in (conditions[:1] if phase=='pilot' else conditions):
                order=list(models);rng.shuffle(order)
                rows.extend(dict(phase=phase,model=model,task=task,condition=condition,trials=n) for model in order)
    return rows



def trained_task_specifications(campaign_path=FAMILY_CAMPAIGN,collection_path=PROJECT/'real_robot/collection_plan.json'):
    """Objects with quality/history-eligible anchors in the actual training split."""
    recipe=read(campaign_path)['training_recipe'];dataset=Path(recipe['local_verified_dataset'])
    training=set(read(dataset/'meta/splits.json')['train'])
    quality=read(dataset/'meta/quality_policy.json')['usable_frames']
    catalog=read(collection_path)['tasks'];by_instruction={x['instruction'].strip().lower():x for x in catalog}
    result={}
    for line in (dataset/'meta/episodes.jsonl').read_text().splitlines():
        episode=json.loads(line);ep=episode['episode_index']
        if ep not in training:continue
        usable=set(quality[str(ep)])
        anchors=sum(max(0,i-2) in usable for i in usable)
        if not anchors:continue
        if len(episode['tasks'])!=1:raise ValueError('Map multi-task episodes explicitly before scheduling experiments')
        task=episode['tasks'][0];task=json.loads(task) if isinstance(task,str) else task
        instruction=task['instruction'].strip().lower()
        if instruction not in by_instruction:raise ValueError('Unmapped trained instruction: '+instruction)
        metadata=by_instruction[instruction];identity=metadata['id']
        if identity not in result:
            result[identity]=dict(instruction=metadata['instruction'],task_type=task['task'],training_episodes=0,training_anchors=0,
                success_criterion=metadata.get('success_criterion','Operator confirms the object was picked up and placed in the box.'))
        result[identity]['training_episodes']+=1;result[identity]['training_anchors']+=anchors
    return result


def prepare(args):
    if (args.root/'plan.json').exists():raise ValueError('Plan exists; do not overwrite experiment records')
    collection=read(PROJECT/'real_robot/collection_plan.json')
    specifications=trained_task_specifications()
    tasks=list(specifications) if args.tasks=='all' else args.tasks.split(',')
    if set(tasks)-set(specifications):raise ValueError('Requested object was not trained on eligible real-data anchors')
    conditions=[x['id'] for x in collection['evaluation']['conditions']]
    settings=dict(motion_profile='hardware',joint_governor='on',pose_smoothing_ms=80,ik_method='bounded_dls',
        geometry_profile='grasp',trial_seconds=30,start_position_tolerance_mm=10,start_joint_tolerance_deg=2,
        record_inputs=True,action_hz=25,action_horizon=20,object_prediction_max_gap_s=DEFAULT_MAX_GAP_S,object_prediction_max_displacement_m=DEFAULT_MAX_DISPLACEMENT_M,object_occlusion_model='auto')
    write(args.root/'plan.json',dict(schema='real_model_comparison_v1',state='waiting_for_smolvla_kr',models=MODELS,tasks=tasks,
        checkpoint_selection='best_sampled_action_score',dom_run=str(DOM_RUN),family_campaign=str(FAMILY_CAMPAIGN),
        settings=settings,conditions=collection['evaluation']['conditions'],task_specifications={task:specifications[task] for task in tasks},
        success_definition='Operator-confirmed task completion: pick and place in box; note failed grasp, drop, miss, timeout and intervention separately.',
        schedule=schedule(MODELS,tasks,conditions),pilot_scored_separately=True,
        velocity_protocol='DOM native; DOMV/KRV saved config; current v6 SmolVLA uses velocity only for analytical KR. Do not add v7 velocity prompts to these checkpoints.',
        real_task_metrics=['success_rate','successful_completion_time_s','holding_confirmed','failure_reason','timeout','takeover'],
        offline_action_mse_is_not_rollout_success=True,created_utc=datetime.now(timezone.utc).isoformat()))
    print('Prepared',args.root/'plan.json')


def freeze(args):
    from real_robot.evaluate import prepare_checkpoint
    plan=read(args.root/'plan.json');records={}
    for model in MODELS:
        if model.startswith('smolvla'):
            checkpoint,selected,campaign=select_family_checkpoint(FAMILY_CAMPAIGN,model)
            spec=Path(campaign['training_recipe']['local_verified_dataset'])/'meta/image_preprocessing.json'
            weight=checkpoint;config=None
            neural=selected['velocity_conditioning'];kr=selected['analytical_kr'];offsets=[0]
        else:
            if read(DOM_RUN/model/'status.json').get('state')!='completed':raise ValueError('Training incomplete: '+model)
            checkpoint,selected=prepare_checkpoint(DOM_RUN,model,'best');weight=checkpoint/'model.safetensors'
            spec=Path(read(DOM_RUN/'launch_contract.json')['local_verified_dataset'])/'meta/image_preprocessing.json'
            config=checkpoint/'config.json';cfg=read(config)
            neural=bool(cfg['condition_on_object_velocity']);kr=bool(cfg['enable_residual_flow']);offsets=[-2,0]
        records[model]=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(weight),epoch=selected['epoch'],
            selected_weights=selected.get('weights','raw'),selection_score=selected.get('selection_score'),
            image_spec=str(spec),image_spec_sha256=sha(spec),config_sha256=None if config is None else sha(config),
            neural_velocity_conditioning=neural,analytical_kr=kr,policy_observation_offsets=offsets)
    plan.update(state='preflight_pending',selected=records);write(args.root/'plan.json',plan)
    return plan


def preflight(args,plan):
    folder=args.root/'preflight';folder.mkdir(exist_ok=True)
    for model,record in plan['selected'].items():
        output=folder/(model+'.json');log=folder/(model+'.log')
        if model.startswith('smolvla'):
            command=[str(PROJECT/'workstation/family-python.sh'),str(PROJECT/'workstation/real_family_policy_worker.py'),
                '--campaign',str(FAMILY_CAMPAIGN),'--family',model,'--checkpoint',record['checkpoint'],
                '--expected-sha256',record['checkpoint_sha256'],'--image-spec',record['image_spec'],
                '--saved-input',str(args.saved_input.resolve()),'--preflight-output',str(output)]
        else:
            command=[str(PROJECT/'workstation/finetune-python.sh'),str(PROJECT/'workstation/preflight_selected_policy.py'),
                '--checkpoint',record['checkpoint'],'--image-spec',record['image_spec'],
                '--saved-input',str(args.saved_input.resolve()),'--output',str(output)]
        write(args.root/'status.json',dict(state='offline_preflight',model=model,hardware_accessed=False))
        with log.open('w') as stream:result=subprocess.run(command,cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode or not output.is_file() or not read(output).get('passed'):
            write(args.root/'status.json',dict(state='preflight_failed',model=model,log=str(log),returncode=result.returncode));return False
    plan['state']='ready_for_supervised_pilot';write(args.root/'plan.json',plan)
    write(args.root/'status.json',dict(state=plan['state'],hardware_accessed=False,physical_trials_started=False,
        next_step='Operator launches one matched pilot session; SPACE controls each trial.'))
    return True


def watch(args):
    while True:
        status=read(FAMILY_CAMPAIGN.parent/'runs/smolvla_kr/status.json')
        if status.get('state')=='completed':break
        if status.get('state')=='failed':
            write(args.root/'status.json',dict(state='training_failed',details=status));return
        write(args.root/'status.json',dict(state='waiting_for_smolvla_kr',training=status,hardware_accessed=False))
        time.sleep(15)
    if args.training_pid:
        while Path(f'/proc/{args.training_pid}/cmdline').exists():
            command=Path(f'/proc/{args.training_pid}/cmdline').read_bytes()
            if b'train_family_finetune.py' not in command or b'smolvla_kr' not in command:break
            time.sleep(5)
    plan=freeze(args)
    if not args.saved_input:raise ValueError('--saved-input required for offline preflight')
    preflight(args,plan)


def launch(args):
    plan=read(args.root/'plan.json')
    if plan['state']!='ready_for_supervised_pilot':raise ValueError('Wait for completed training and all saved-input preflights')
    if args.model not in plan['selected'] or args.task not in plan['tasks']:raise ValueError('Model/task outside frozen plan')
    conditions={row['id'] for row in plan['conditions']}
    if args.condition not in conditions:raise ValueError('Condition outside plan')
    if args.condition=='static' and args.phase!='pilot':raise ValueError('Static is a diagnostic pilot condition')
    rec=plan['selected'][args.model];checkpoint=Path(rec['checkpoint'])
    weight=checkpoint if checkpoint.is_file() else checkpoint/'model.safetensors'
    if sha(weight)!=rec['checkpoint_sha256'] or sha(rec['image_spec'])!=rec['image_spec_sha256']:raise ValueError('Frozen evaluation inputs changed')
    if rec['config_sha256'] and sha(checkpoint/'config.json')!=rec['config_sha256']:raise ValueError('Frozen model config changed')
    shared=args.root/'shared-start-pose.json'
    if not shared.exists() and args.phase!='pilot':raise ValueError('Capture the shared start during a pilot first')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=(args.root/'sessions'/f'{args.phase}-{args.task}-{args.condition}-{args.model}-{stamp}').resolve()
    output.parent.mkdir(exist_ok=True)
    settings=plan['settings'];trials=3 if args.phase=='pilot' else 20
    override=getattr(args,'max_trials',None)
    if override is not None:
        if args.phase!='pilot' or override<=0:raise ValueError('Positive trial overrides apply only to pilots')
        trials=override
    command=[str(PROJECT/'.venv/bin/python'),'-u','-m','real_robot.evaluate','--model',args.model,
        '--run',str(DOM_RUN),'--family-campaign',str(FAMILY_CAMPAIGN),'--checkpoint','best',
        '--expected-checkpoint-sha256',rec['checkpoint_sha256'],'--task',args.task,'--output',str(output),
        '--motion-profile',settings['motion_profile'],'--joint-governor',settings['joint_governor'],
        '--ik-method',settings['ik_method'],'--geometry-profile',settings['geometry_profile'],
        '--object-occlusion-model',resolve_model(settings.get('object_occlusion_model','auto'),'static' if args.condition=='static' else 'moving'),
        '--object-prediction-max-gap-s',str(settings.get('object_prediction_max_gap_s',DEFAULT_MAX_GAP_S)),
        '--object-prediction-max-displacement-m',str(settings.get('object_prediction_max_displacement_m',DEFAULT_MAX_DISPLACEMENT_M)),
        '--pose-smoothing-ms',str(settings['pose_smoothing_ms']),'--trial-seconds',str(settings['trial_seconds']),
        '--start-position-tolerance-mm',str(settings['start_position_tolerance_mm']),
        '--start-joint-tolerance-deg',str(settings['start_joint_tolerance_deg']),
        '--seconds','600' if args.phase=='pilot' else '1680','--max-trials',str(trials),'--record-inputs']
    if shared.exists():command+=['--start-pose',str(shared.resolve())]
    if args.execute:command+=['--execute']
    if args.new_mask:command+=['--new-mask']
    from real_robot.policy_stream import LiveObjectEstimator
    from real_robot.evaluate import policy_object_inputs_required
    object_inputs_required=policy_object_inputs_required(args.model,checkpoint,rec.get('neural_velocity_conditioning'))
    prediction=(LiveObjectEstimator(settings.get('object_prediction_max_gap_s',DEFAULT_MAX_GAP_S),settings.get('object_prediction_max_displacement_m',DEFAULT_MAX_DISPLACEMENT_M),
        motion_model=resolve_model(settings.get('object_occlusion_model','auto'),'static' if args.condition=='static' else 'moving')).settings if object_inputs_required else dict(model='not_used_by_policy',object_inputs_required=False))
    write(args.root/'session-requests'/(output.name+'.json'),dict(model=args.model,task=args.task,condition=args.condition,
        phase=args.phase,planned_trials=trials,object_inputs_used_by_policy=object_inputs_required,tracking_prediction=prediction,output=str(output),checkpoint=rec,command=command,execute=args.execute))
    write(args.root/'status.json',dict(state='session_starting',model=args.model,task=args.task,condition=args.condition,
        output=str(output),starts_paused=True,requested_trials=trials))
    result=subprocess.run(command,cwd=PROJECT,env=dict(os.environ,DISPLAY=os.environ.get('DISPLAY',':1')))
    captured=output/'controller/start_pose.json'
    if not shared.exists() and captured.is_file():
        reference=read(captured)
        if 'command_joints_rad' not in reference or 'command_gripper_raw' not in reference:raise ValueError('Incomplete start reference')
        shutil.copyfile(captured,shared)
    if result.returncode:raise RuntimeError('Session failed; inspect '+str(output))


def summarize(args):
    groups=defaultdict(list)
    for request_path in (args.root/'session-requests').glob('*.json'):
        request=read(request_path)
        if not request['execute']:continue
        prediction=request.get('tracking_prediction')
        if prediction is None:
            manifest=Path(request['output'])/'controller/manifest.json'
            previous=read(manifest) if manifest.is_file() else {}
            prediction=previous.get('object_occlusion_prediction',dict(model='legacy_coasting_only',max_gap_s=previous.get('object_gap_prediction_max_s',.08)))
        recipe_key=json.dumps(prediction,sort_keys=True)
        for path in (Path(request['output'])/'controller').glob('trial-*.json'):
            trial=read(path)
            if trial.get('excluded_from_task_success_rate'):continue
            groups[(request['phase'],request['model'],request['task'],request['condition'],recipe_key)].append(trial)
    rows=[]
    for key,trials in sorted(groups.items()):
        successes=[x for x in trials if x['outcome']=='success']
        times=[x['end_monotonic_s']-x['start_monotonic_s'] for x in successes]
        rows.append(dict(phase=key[0],model=key[1],task=key[2],requested_condition=key[3],tracking_prediction=json.loads(key[4]),trials=len(trials),successes=len(successes),
            success_rate=len(successes)/len(trials),mean_successful_completion_s=sum(times)/len(times) if times else None,
            holding_confirmed=sum(bool(x.get('holding_confirmed')) for x in trials),
            timeouts=sum(x.get('reason')=='timeout' for x in trials),
            takeovers=sum('takeover' in x.get('reason','').lower() for x in trials),
            failure_reasons=dict(Counter(x.get('reason','unknown') for x in trials if x['outcome']!='success'))))
    write(args.root/'rollout-summary.json',dict(groups=rows,
        interpretation='Pilot/scored separate. Speed condition is operator-requested; verify measured velocity from recorded inputs before claiming speed-stratified results. No expert-target action MSE inferred from autonomous rollout.'))
    print('Wrote',args.root/'rollout-summary.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','watch','launch','summarize'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--tasks',default='all')
    p.add_argument('--saved-input',type=Path);p.add_argument('--training-pid',type=int)
    p.add_argument('--model',choices=MODELS);p.add_argument('--task',choices=['striped_ball','marker','orange'])
    p.add_argument('--phase',choices=['pilot','scored'],default='pilot');p.add_argument('--condition',default='iid_slow')
    p.add_argument('--max-trials',type=int,help='Override pilot trial count; scored blocks remain fixed')
    p.add_argument('--execute',action='store_true');p.add_argument('--new-mask',action='store_true')
    a=p.parse_args();a.root=a.root.resolve()
    try:globals()[a.command](a)
    except BaseException as exc:
        write(a.root/'status.json',dict(state='error',operation=a.command,error=f'{type(exc).__name__}: {exc}',
            hardware_access_possible=a.command=='launch'));raise


if __name__=='__main__':main()
