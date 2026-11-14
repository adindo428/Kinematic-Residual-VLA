"""Prepare pinned DOM/DOMV/KRV recipes and a CPU collection/inference audit.

Does not launch training, open robot devices or overwrite the source dataset.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from real_robot.grasp_prior import follower_pose
from real_robot.vx300s_kinematics import forward,joints_from_raw_ticks


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def prepare(output,dataset_override=None):
    output.mkdir(parents=True,exist_ok=True)
    parent=PROJECT/'scratch/real_robot/finetuning-runs/20261109T014917Z'
    contract=json.loads((parent/'launch_contract.json').read_text())
    dataset=Path(dataset_override or contract['local_verified_dataset']).resolve()
    contract['local_verified_dataset']=str(dataset)
    contract['dataset_quality_policy_sha256']=sha(dataset/'meta/quality_policy.json')
    if dataset_override is not None:
        contract['dataset_source']='local grasp-annotated derivative; parent Hugging Face revision retained for provenance, not a claim this derivative is uploaded'
        contract['grasp_annotation_report']=str(dataset/'grasp_annotation_report.json')
    original=PROJECT/'scratch/real_robot/static-marker-eval/checkpoints/original-dom'
    contract.update(families=['dom','domv','krv100'],recipe_version='real_robust_v4_grasp_common_dom' if dataset_override else 'real_robust_v3_common_dom',parent_run=str(parent),
        status='prepared; no training launched',training_started=False,learning_rate=3e-6,weight_decay=.01,
        scheduler=dict(type='warmup_plateau',warmup_steps=200,factor=.5,patience=2,min_lr=3e-7),
        max_epochs=30,early_stopping_patience=7,
        augmentation=dict(photometric=dict(enabled=True,brightness=[.9,1.1],contrast=[.9,1.1],saturation=[.95,1.05],
            shared_across_history_and_cameras=True),
            proprioception=dict(enabled=False,sample_probability=.3,translation_std_m=.002,rotation_std_deg=.5,clip_std=3.)),
        controller_contract=dict(kinematics='real_robot.vx300s_kinematics',tcp_offset_m=[.16,0,0],fps=25,
            observation_offsets=[-2,0],action_horizon=20,euler_delta='wrapped_xyz_v1',
            state='measured joint FK',action='commanded joint FK; continuous gripper',
            collection_actuation='leader/follower joint position',inference_actuation='bounded_dls plus Ruckig',
            actuation_equivalence='not established physically; requires expert-command replay before interpreting model comparisons',
            matched_start_pose=True),
        launch_checks=['shared FK/export/inference audit','strict checkpoint load and training smoke',
            'fixed controller build/profile and recovered home verified physically; controller faults logged separately'],
        methods=dict(dom=dict(condition_on_object_velocity=False,enable_residual_flow=False),
            domv=dict(condition_on_object_velocity=True,enable_residual_flow=False),
            krv100=dict(condition_on_object_velocity=True,enable_residual_flow=True,residual_phase_gate_mode='constant',
                residual_flow_formulation='coordinate',residual_lambda_type='constant',residual_drift_scale=1.,
                residual_drift_mode='trajectory',residual_action_dt=.04,residual_axis_scale_xyz=[1.,1.,1.],
                residual_normalize_drift=True,residual_center_drift=True,residual_estimate_drift_mean=False)),
        comparison_note='All three methods start from identical original released DOM parameters, with the same real-data split, anchors, statistics and optimization. No simulation-specific weights or optimizer states are used.')
    source=json.loads((original/'source.json').read_text());dom_id='original-dom-'+source['revision'][:12]
    contract['checkpoint']={family:dom_id for family in contract['families']}
    contract['checkpoint_preferences']=[dict(checkpoint_id=dom_id,path=str(original),
        sha256={name:sha(original/name) for name in ['config.json','model.safetensors']},source=source)]
    contract['checkpoint_repo']=source.get('repo_id','hzxie/dynamic-vla-DOM')
    contract['checkpoint_revision']=source['revision']
    # Preserve historical launch details in parent_run, not as new verification claims.
    for key in ['initial_memory_smoke','workstation_setup','checkpoint_upload_status','continuation_policy']:
        contract.pop(key,None)
    contract['checkpoint_architecture_verification']='See per-family disposable real-data smoke reports; no full training run launched.'
    contract['checkpoint_selection_reason']='User requests identical original released DOM initialization for DOM, DOMV and KRV100. Simulation-specific checkpoints are excluded.'
    contract.pop('completed_epochs_only',None)
    contract['completed_epoch_policy']={family:'original released DOM weights; author epoch not specified' for family in contract['families']}
    contract['checkpoint_smoke_reports']=[str(output/(family+'-smoke.json')) for family in contract['families']]
    contract['profiling_note']='Batch 16 is inherited from the earlier FastViT profile; the new augmentation recipe has only been smoke-tested at batch 1, not throughput-profiled.'
    contract['vision_backbone']['applies_to']=contract['families']
    contract['deployment_requirements']['applies_to']=contract['families']
    contract['controller_contract']['matched_start_pose']='required for comparison; physical reset remains unresolved'
    contract['audit_reports']={name:str(output/(name+'.json')) for name in
        ['pipeline_audit','training_motion_audit','phase_coverage_audit']}
    run_prefix='real-v4-grasp-common-dom-' if dataset_override else 'real-v3-common-dom-'
    contract['wandb']['run_names']={family:run_prefix+family+'-photo-lr3e6' for family in contract['families']}
    names=set(contract.get('source_hashes',{}))|{'workstation/train_real_finetune.py','workstation/smoke_real_finetune.py',
        'real_robot/training_augmentation.py','real_robot/vx300s_kinematics.py','real_robot/grasp_prior.py','real_robot/aloha_arm.py',
        'real_robot/policy_ik.py','real_robot/policy_reset.py','real_robot/policy_eval.py','workstation/real_policy_worker.py',
        'workstation/prepare_real_robustness.py','workstation/smoke_robust_finetune.py'}
    names.update({'real_robot/grasp_annotations.py','real_robot/grasp_phase.py'})
    contract['source_hashes']={name:sha(PROJECT/name) for name in sorted(names)}
    (output/'photometric.json').write_text(json.dumps(contract,indent=2)+'\n')
    noise=copy.deepcopy(contract);noise['augmentation']['proprioception']['enabled']=True
    noise['wandb']['run_names']={family:run_prefix+family+'-photo-proprio-lr3e6' for family in contract['families']}
    (output/'photometric_plus_proprio.json').write_text(json.dumps(noise,indent=2)+'\n')
    # Compare saved raw collection telemetry against the canonical inference FK.
    recording=PROJECT/'scratch/real_robot/collections/marker-train-20261108T233827.476765Z/recording'
    manifest=json.loads((recording/'manifest.json').read_text());errors=[];count=0
    for line in (recording/'robot.jsonl').open():
        row=json.loads(line)
        if 'follower_raw_ticks' not in row or 'T_base_tcp_candidate' not in row:continue
        if count%37==0:
            raw=row['follower_raw_ticks'];q,t=follower_pose(raw,manifest['tcp_offset_m'])
            deployed=forward(joints_from_raw_ticks(raw),manifest['tcp_offset_m'])
            recorded=np.asarray(row['T_base_tcp_candidate'])
            errors.append([float(np.max(abs(t-deployed))),float(np.max(abs(recorded-deployed)))])
        count+=1
    measured=np.array(errors)
    report=dict(hardware_accessed=False,dataset=str(dataset),recorded_robot_samples=count,checked_samples=len(errors),
        maximum_collection_vs_inference_fk_error=float(measured[:,0].max()),
        maximum_recorded_vs_current_fk_error=float(measured[:,1].max()),
        tcp_offset_m=manifest['tcp_offset_m'],pose_encoder='shared real_robot.dom_dataset.pose6',
        data_fps=25,live_camera_fps=30,history_steps=[-2,0],horizon_steps=20,
        preprocessing_sha256=sha(dataset/'meta/image_preprocessing.json'),
        tracking_quality_policy_sha256=sha(dataset/'meta/quality_policy.json'),
        image_preprocessing='export and live worker use real_robot.image_preprocessing.transform, BGR->RGB once; prepared videos are not transformed twice',
        normalization='clean training split statistics, serialized into each new checkpoint and reused at inference',
        intentional_difference='Collection sends leader/follower joint goals; inference maps Cartesian command-FK targets through bounded IK and jerk limits.',
        unresolved='Physical expert-command replay and stable reset/home verification are not replaced by numerical FK agreement.')
    (output/'pipeline_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(recipes=[str(output/'photometric.json'),str(output/'photometric_plus_proprio.json')],audit=report),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,help='Use a new verified local derivative while preserving the original split and checkpoint provenance')
    args=parser.parse_args();prepare(args.output.resolve(),args.dataset)
