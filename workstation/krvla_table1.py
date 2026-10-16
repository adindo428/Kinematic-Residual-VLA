#!/usr/bin/env python3
"""Wait for the validated epoch-one checkpoint, then run the paper DOM suite."""
import argparse
import csv
import datetime
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'evaluation/official_dom/scripts'))
import local as official
from prepare import sha256, write_json
from report import summarize, verify_folder


def state(root, phase, **details):
    record={'state':phase,'updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'expected_trials':1800,'results':str(root),**details}
    write_json(root/'suite_status.json',record)
    print(json.dumps(record),flush=True)


def validate_request(request):
    if not request.get('user_authorized') or request['target_epoch']!=1:
        raise ValueError('Authorized epoch-one Table 1 request is required')
    for path, expected in request['files_sha256'].items():
        if sha256(path)!=expected:
            raise ValueError(f'Evaluation code changed after queuing: {path}')
    sys.path.insert(0,request['policy_source'])
    from core.dom_training import load_family
    manifest=load_family(Path(request['family_root']))
    if manifest['fingerprint']!=request['manifest']:
        raise ValueError('Training family changed after evaluation was queued')
    baseline=Path(request['baseline_results'])
    if sha256(baseline/'protocol.json')!=request['baseline_protocol_sha256']:
        raise ValueError('Baseline benchmark protocol changed')
    if sha256(baseline/'summary.json')!=request['baseline_summary_sha256']:
        raise ValueError('Completed baseline results changed')
    return manifest


def dependency_ready(request, manifest):
    run=Path(request['family_root'])/'runs/krv100'
    path=run/'training_complete.json'
    if not path.exists():
        return None
    completed=json.loads(path.read_text())
    if (completed.get('completion_reason')!='user_epoch_limit' or completed.get('epoch')!=1
            or completed.get('cursor')!=0 or completed.get('pending_validation')
            or completed.get('samples')!=manifest['train_frames']
            or completed.get('updates')!=math.ceil(manifest['train_frames']/352)
            or completed.get('fingerprint')!=request['experiment_fingerprint']
            or completed.get('epoch_limit_control_sha256')!=request['training_control_sha256']):
        raise ValueError('Completion record is not the approved, fully trained epoch-one checkpoint')
    coordinator=json.loads((Path(request['family_root']).parent/'training_status.json').read_text())
    if coordinator['phase']!='complete':
        return None  # Wait for both ranks and the successful training coordinator exit.
    final=json.loads((run/'final.json').read_text())
    epoch=json.loads((run/'epoch-001.json').read_text())
    if final!=epoch or final['fingerprint']!=request['experiment_fingerprint']:
        raise ValueError('Epoch-one and final pointers must name the same saved generation')
    checkpoint=run/'generations'/final['generation']
    if str(checkpoint)!=completed['checkpoint'] or final['progress']['samples']!=manifest['train_frames']:
        raise ValueError('Saved checkpoint differs from the completed training record')
    validation=run/'validation-epoch-001.json'
    if not validation.is_file():
        raise ValueError('Epoch-one held-out validation has not finished')
    return checkpoint


def bind(request, root, checkpoint, manifest):
    protocol_path=root/'protocol.json'
    if protocol_path.exists():
        protocol=json.loads(protocol_path.read_text())
        if protocol['trained_checkpoint']!=str(checkpoint) or protocol['checkpoint_weights_sha256']!=sha256(checkpoint/'model.safetensors'):
            raise ValueError('Existing evaluation is bound to another checkpoint')
        return protocol
    stage=root/'stage'
    (stage/'checkpoint').mkdir(parents=True,exist_ok=True)
    simulator_source=Path(request['official_stage'])/'source'
    if (stage/'source').exists():
        if (stage/'source').resolve()!=simulator_source.resolve():
            raise ValueError('Staged simulator source points elsewhere')
    else:
        (stage/'source').symlink_to(simulator_source,target_is_directory=True)
    for name in ('config.json','model.safetensors','provenance.json'):
        target=stage/'checkpoint'/name
        if target.exists():
            if not os.path.samefile(checkpoint/name,target):
                raise ValueError('Staged checkpoint belongs to another generation')
        else:
            os.link(checkpoint/name,target)
    cfg=json.loads((checkpoint/'config.json').read_text())
    if (cfg['completed_epochs']!=1 or cfg['experiment_fingerprint']!=request['experiment_fingerprint']
            or cfg['completion_reason']!='user_epoch_limit' or not cfg['enable_residual_flow']
            or not cfg['condition_on_object_velocity'] or cfg['residual_flow_formulation']!='coordinate'
            or cfg['velocity_input_mean']!=manifest['velocity_mean'] or cfg['velocity_input_std']!=manifest['velocity_std']):
        raise ValueError('Trained checkpoint/configuration differs from the requested KRVLA+Velocity model')
    baseline=Path(request['baseline_results'])
    (root/'test_lists').mkdir(exist_ok=True)
    for original in (baseline/'test_lists').iterdir():
        target=root/'test_lists'/original.name
        if target.exists() and target.read_bytes()!=original.read_bytes():
            raise ValueError('Staged benchmark list differs from the DOM baseline')
        if not target.exists():
            shutil.copy2(original,target)
    protocol=json.loads((baseline/'protocol.json').read_text())
    protocol.update(benchmark='DOM Table 1 closed-loop evaluation of trained KRVLA + Velocity',
        method_label='KRVLA + Velocity · epoch 1',checkpoint_kind='locally_trained',
        checkpoint_repo='Kinematic-Residual-VLA local training',checkpoint_revision=checkpoint.name,
        checkpoint=str(stage/'checkpoint'),trained_checkpoint=str(checkpoint),
        checkpoint_weights_sha256=sha256(checkpoint/'model.safetensors'),
        checkpoint_config_sha256=sha256(checkpoint/'config.json'),
        policy_source=request['policy_source'],policy_source_sha256=manifest['source_sha256'],
        training_manifest=manifest['fingerprint'],completed_training_epochs=1,
        original_scheduler_horizon_epochs=100,epoch_limit_control_sha256=request['training_control_sha256'],
        velocity_source='Current simulator target-object pose and actual linear velocity in the robot frame; no future state',
        additional_policy_inputs=['observation.environment_state: object XYZ, Euler rotation, XYZ velocity'],
        recording='Official trial observer; observation-only extension attaches already computed live object state',
        local_dom_baseline=json.loads((baseline/'summary.json').read_text())['observed'],
        local_dom_baseline_summary_sha256=request['baseline_summary_sha256'],
        evaluation_code_sha256=request['files_sha256'],
        baseline_policy_dependencies=protocol['dependencies'],
        dependencies={name:importlib.metadata.version(name) for name in
                      ('torch','transformers','lerobot','timm','huggingface-hub','torchcodec')})
    write_json(protocol_path,protocol)
    (root/'README.md').write_text('KRVLA + Velocity epoch-one closed-loop Table 1 evaluation\n\n'
        'Isaac Sim 4.5.0 / Isaac Lab 2.2.1; the same 90 released configurations, seeds, '
        '20 repeats each, streaming, action controller and success/time definitions as the completed DOM baseline. '
        'The trained policy additionally receives current simulator object pose/velocity in robot coordinates. '
        'Nine pilot trials gate the full 1800-trial suite. This benchmark is separate from training-split held-out loss validation.\n')
    return protocol


def comparison(root, protocol):
    observed=json.loads((root/'summary.json').read_text())['observed']
    rows=[]
    for key in [*protocol['dimension_order'],'Success','Time']:
        paper=protocol['paper_reference'][key];baseline=protocol['local_dom_baseline'][key]
        rows.append({'Metric':key,'Paper_DOM':paper,'Local_DOM_baseline':baseline,
                     'KRVLA_Velocity_epoch1':observed[key],'Difference_vs_local_DOM':observed[key]-baseline})
    with (root/'table1_baseline_comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Table 1: KRVLA + Velocity after one epoch','',
           '| Metric | Paper DOM | Local DOM baseline | KRVLA + Velocity, epoch 1 | Change vs local DOM |',
           '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['Metric']} | {row['Paper_DOM']:.2f} | {row['Local_DOM_baseline']:.2f} | "
                     f"{row['KRVLA_Velocity_epoch1']:.2f} | {row['Difference_vs_local_DOM']:+.2f} |")
    lines+=['','Success columns are percentages; success changes are percentage points. Time is mean simulated seconds across successes and failures.',
            'Each local method has 1800 verified manipulation trials under the same published test configurations and simulator stack.',
            'KRVLA + Velocity additionally uses current simulator object state. See protocol.json for checkpoint, input and runtime provenance.','']
    (root/'table1_baseline_comparison.md').write_text('\n'.join(lines))


def evaluate(request, root, manifest):
    manifest=validate_request(request)
    checkpoint=dependency_ready(request,manifest)
    if checkpoint is None:
        raise RuntimeError('Successful epoch-one training dependency is not ready')
    protocol=bind(request,root,checkpoint,manifest)
    args=SimpleNamespace(stage=root/'stage',root=root,assets=Path(request['assets']),
        isaaclab=Path(request['isaaclab']),policy_python=Path(request['policy_python']),
        policy_source=Path(request['policy_source']),policy_probe=REPO/'workstation/krvla_table1_probe.py',
        port=27000,sim_gpu=0,policy_gpu=1,kit_args=request['kit_args'],dry_run=False,resume_verified=True)
    original=official.commands
    def commands(*positional,**keywords):
        server,client=original(*positional,**keywords)
        server[2]=str(REPO/'workstation/krvla_table1_server.py')
        client[2]=str(REPO/'workstation/inference_krvla.py')
        client.extend(['--source',request['policy_source']])
        client[client.index('-a')+1]='KRVLA_Velocity_epoch001'
        return server,client
    official.commands=commands
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f'Evaluation interrupted by signal {signum}')
    import signal
    signal.signal(signal.SIGTERM,interrupted)
    pilot=root/'pilot_verified.json'
    if not pilot.exists():
        state(root,'pilot',checkpoint=str(checkpoint))
        official.run_one(args,'pilot')
    else:
        verify_folder(root/'pilot',root,root/'test_lists/pilot.txt',1)
    for index,dimension in enumerate(protocol['dimension_order']):
        validate_request(request)
        if sha256(root/'stage/checkpoint/config.json')!=protocol['checkpoint_config_sha256']:
            raise ValueError('Bound checkpoint configuration changed during evaluation')
        state(root,'running',dimension=dimension,checkpoint=str(checkpoint))
        official.run_one(args,'full',index)
    summarize(root)
    comparison(root,protocol)
    state(root,'complete',verified_trials=1800,checkpoint=str(checkpoint),exit_code=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args()
    request=json.loads(args.request.read_text());root=Path(request['results'])
    ownership=(root/'.dependency.lock').open('a')
    try:
        fcntl.flock(ownership,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError('Another process already owns this queued evaluation') from error
    try:
        manifest=validate_request(request)
        if (root/'suite_status.json').exists() and json.loads((root/'suite_status.json').read_text())['state']=='complete':
            summarize(root);return
        state(root,'waiting_epoch_one',training_run=request['family_root'],target_epoch=1)
        while dependency_ready(request,manifest) is None:
            # This is an independent background service; no GPU allocations yet.
            time.sleep(30)
        evaluate(request,root,manifest)
    except BaseException as error:
        state(root,'failed',error=str(error),exit_code=1)
        raise


if __name__=='__main__':
    main()
