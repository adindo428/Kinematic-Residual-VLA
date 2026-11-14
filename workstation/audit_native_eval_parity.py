"""Compare native training-validation and deployed samplers; opens no devices."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from smoke_real_finetune import LeRobotDataset,PROJECT
sys.path.insert(0,str(PROJECT))
from real_policy_worker import load_policy,infer,prepare_live_batch
from real_robot.action_objective import sampled_absolute_actions
from real_robot.evaluate import prepare_checkpoint,read_run_contract
from real_robot.image_preprocessing import load_spec,transform
from real_robot.native_policy_contract import verify_native_contract,training_instruction,verify_live_geometry
from real_robot.training_augmentation import model_inputs
from utils.dom_data import DOMData
from utils.real_robot_contract import absolute_pose


def clone(batch):return {k:v.clone() if torch.is_tensor(v) else v for k,v in batch.items()}
def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--session',type=Path,action='append',default=[])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    contract=read_run_contract(args.run);dataset=Path(contract['local_verified_dataset']);spec=load_spec(dataset/'meta/image_preprocessing.json')
    base=LeRobotDataset(contract['repo_id'],root=dataset,split='validation',revision=contract['revision'],
        euler_delta_convention=contract['euler_delta_convention'],required_features=[
            'observation.images.opst_cam','observation.images.wrist_cam','observation.state','observation.environment_state','action'],
        delta_timestamps={'observation':[-2,0],'action':list(range(20))})
    # Use the very same batched video adapter as training/validation.
    data=DOMData(base,horizon=20,cache_bytes=contract.get('decoder_cache_bytes',512*1024**2)) if contract.get('batched_decode') else base
    metadata={r['episode_index']:r for r in map(json.loads,(dataset/'meta/episodes.jsonl').read_text().splitlines())}
    selected={}
    for slot,ep in enumerate(base.episodes):
        task=json.loads(metadata[ep]['tasks'][0])['objects']
        if task in selected:continue
        start=int(base.episode_data_index['from'][slot]);end=int(base.episode_data_index['to'][slot])
        eligible=np.flatnonzero((np.asarray(base.quality_indices)>=start)&(np.asarray(base.quality_indices)<end))
        if len(eligible):selected[task]=[int(eligible[len(eligible)//3]),int(eligible[2*len(eligible)//3])]
    rig=read(PROJECT/'real_robot/calibration/collection-rig.json')
    report=dict(geometry=verify_live_geometry(dataset,rig,contract),hardware_accessed=False,models={},native_instructions={t:training_instruction(dataset,t) for t in ['orange','marker','striped_ball']},
        limits=['Training-only photometric augmentation is disabled in evaluation, as in held-out validation.',
                'Live raw RGB and lossy training video pixels cannot be bit-identical.',
                'Human teleoperation during collection differs from bounded IK/Ruckig execution.',
                'Live asynchronous inference skips expired actions; training supervises the full chunk.',
                'Velocity innovation is outside the native model sampler and is excluded from parity checks.'])
    sources={r['episode_index']:r for r in map(json.loads,(dataset/'meta/sources.jsonl').read_text().splitlines())}
    pixels=[]
    for family in ['domv','krv100']:
        checkpoint,_=prepare_checkpoint(args.run,family,'best')
        result=verify_native_contract(checkpoint,family,dataset,contract)
        model,cfg=load_policy(checkpoint,PROJECT/'scratch/real_robot/finetuning-checkpoints')
        noise=torch.randn((1,cfg.chunk_size,cfg.max_action_dim),device='cuda',generator=torch.Generator(device='cuda').manual_seed(12345))
        result['samples']=[]
        for task,indices in selected.items():
            for index in indices:
                item=data[index];batch=default_collate([item]);batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
                batch=model_inputs(batch,cfg);original=batch['observation.state'].clone()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    trained=sampled_absolute_actions(model,clone(batch),noise.clone())
                    deployed=model._get_action_chunk(model._prepare_batch(clone(batch)),noise=noise.clone())
                    deployed=absolute_pose(deployed,original[:,-1:,:],cfg.euler_delta_convention)
                torch.testing.assert_close(trained,deployed,atol=2e-5,rtol=2e-5)
                torch.testing.assert_close(batch['observation.state'],original,atol=0,rtol=0)
                ep=int(item['episode_index']);fr=int(item['frame_index'])
                result['samples'].append(dict(kind='heldout_training_loader',task=task,episode=ep,frame=fr,
                    maximum_absolute_decode_difference=float((trained-deployed).abs().max())))
                if family=='domv':
                    quality=[json.loads(l) for l in (dataset/f'quality/episode_{ep:06d}.jsonl').open()][fr]
                    for role,camera in [('front','opst_cam'),('wrist','wrist_cam')]:
                        relative=quality.get('camera_source_files',{}).get(camera)
                        raw=Path(sources[ep]['raw_session'])/relative if relative else None
                        if raw is not None and raw.exists():
                            with np.load(raw,allow_pickle=False) as z:expected=transform(z['bgr'],spec,role)[:,:,::-1].copy()
                            actual=(item['observation.images.'+camera][-1].permute(1,2,0).numpy()*255).round()
                            pixels.append(dict(episode=ep,frame=fr,camera=camera,raw_export_rgb_mae=float(np.abs(actual-expected.astype(float)).mean()),
                                red_blue_swapped_mae=float(np.abs(actual-expected[:,:,::-1].astype(float)).mean())))
        for session in args.session:
            files=sorted((session/'exchange/inference_inputs').glob('*.npz'))
            for path in [files[0],files[len(files)//2]] if files else []:
                with np.load(path,allow_pickle=False) as z:
                    payload={k:z[k] for k in ['front','wrist','state','environment','required','valid']};meta=json.loads(str(z['metadata']))
                batch=prepare_live_batch(cfg,payload,meta,spec);original=batch['observation.state'].clone()
                deployed=infer(model,cfg,payload,meta,spec,noise.clone())
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):trained=sampled_absolute_actions(model,clone(batch),noise.clone())
                expected=trained[0].float().cpu().numpy();np.testing.assert_allclose(deployed,expected,atol=2e-5,rtol=2e-5)
                result['samples'].append(dict(kind='recorded_live_transport',archive=str(path),maximum_absolute_decode_difference=float(np.abs(deployed-expected).max())))
                # The same constant-gate quality mask must neutralize confirmed holding.
                batch['dynamic_object.velocity_required'].fill_(False);batch['dynamic_object.velocity_valid'].fill_(False)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    held=sampled_absolute_actions(model,clone(batch),noise.clone())
                    native=model._get_action_chunk(model._prepare_batch(clone(batch)),noise=noise.clone())
                    native=absolute_pose(native,original[:,-1:,:],cfg.euler_delta_convention)
                torch.testing.assert_close(held,native,atol=2e-5,rtol=2e-5)
                result['samples'].append(dict(kind='confirmed_holding_gate',maximum_absolute_decode_difference=float((held-native).abs().max())))
        report['models'][family]=result
        del model;torch.cuda.empty_cache()
        (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(family,'passed',len(result['samples']),'sampler comparisons',flush=True)
    report.update(passed=True,image_comparisons=pixels)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Saved',args.output/'summary.json',flush=True)


if __name__=='__main__':main()
