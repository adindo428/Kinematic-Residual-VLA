"""One disposable real-data optimizer step; no saved weights or robot access."""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from smoke_real_finetune import build_training_components,PROJECT
sys.path.insert(0,str(PROJECT))
from real_robot.training_augmentation import augment_training_batch,model_inputs
from utils.real_robot_contract import absolute_pose


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--family',choices=['dom','domv','krv100'],default='dom');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();contract=json.loads(args.contract.read_text());family=args.family
    preference=next(p for p in contract['checkpoint_preferences'] if p['checkpoint_id']==contract['checkpoint'][family])
    model,cfg,data,item=build_training_components(SimpleNamespace(bundle=PROJECT/'scratch/real_robot/finetuning-checkpoints',
        dataset=Path(contract['local_verified_dataset']),revision=contract['revision'],alias=preference.get('alias'),
        checkpoint=preference.get('path'),checkpoint_id=preference['checkpoint_id'],expected_sha256=preference.get('sha256'),family=family,
        method_overrides=contract.get('methods',{}).get(family)))
    import hashlib
    # Fingerprint every learned parameter before the disposable optimization.
    digest=hashlib.sha256()
    for name,value in model.named_parameters():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    initial_parameter_sha256=digest.hexdigest()
    sample=data[0];batch={k:(v.unsqueeze(0).cuda() if torch.is_tensor(v) else [v]) for k,v in sample.items()}
    model.eval().cuda();torch.manual_seed(42)
    invariant=None
    if family=='dom':
        noise=torch.randn((1,cfg.chunk_size,cfg.max_action_dim),device='cuda')
        altered=dict(batch);altered['observation.environment_state']=batch['observation.environment_state']*100+17
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            a=model._get_action_chunk(model._prepare_batch(batch),noise=noise)
            b=model._get_action_chunk(model._prepare_batch(altered),noise=noise)
        invariant=float((a-b).abs().max());assert invariant==0.
    # Exercise both optional augmentations, even for the photo-only first recipe.
    recipe=copy.deepcopy(contract['augmentation']);recipe['proprioception'].update(enabled=True,sample_probability=1.)
    augmented,metrics=augment_training_batch(batch,recipe)
    before=absolute_pose(batch['action'],batch['observation.state'][:,-1:,:],'wrapped_xyz_v1')
    after=absolute_pose(augmented['action'],augmented['observation.state'][:,-1:,:],'wrapped_xyz_v1')
    label_error=float((before[...,:3]-after[...,:3]).abs().max());assert label_error<1e-6
    inputs=model_inputs(augmented,cfg);model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-6,weight_decay=.01)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=model(inputs)
    assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    report=dict(family=family,checkpoint=item['id'],strict_load=True,condition_on_object_velocity=cfg.condition_on_object_velocity,
        initial_parameter_sha256=initial_parameter_sha256,residual_flow_formulation=cfg.residual_flow_formulation,
        dataset_path=str(Path(contract['local_verified_dataset']).resolve()),
        quality_policy_sha256=hashlib.sha256((Path(contract['local_verified_dataset'])/'meta/quality_policy.json').read_bytes()).hexdigest(),
        enable_residual_flow=cfg.enable_residual_flow,usable_train_anchors=len(data),loss=float(loss.detach()),gradient_norm=float(norm),
        augmentation=metrics,absolute_xyz_label_error_m=label_error,object_conditioning_invariance_error=invariant,
        optimizer_step=True,weights_saved=False,hardware_accessed=False,wandb_run_started=False,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
