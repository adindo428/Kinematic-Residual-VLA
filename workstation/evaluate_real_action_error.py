"""Offline sampled EE action errors on a shared deterministic validation episode subset."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import random
import numpy as np
from scipy.spatial.transform import Rotation
from smoke_real_finetune import build_training_components, LeRobotDataset, PROJECT
import torch
sys.path.insert(0, str(PROJECT))
from real_robot.training_augmentation import model_inputs
from utils.real_robot_contract import absolute_pose


def metrics(rows):
    a=np.concatenate(rows)
    return dict(valid_targets=len(a), position_mse_m2=float(a[:,:3].mean()),
        position_rmse_mm=float(np.sqrt(a[:,:3].mean())*1000),
        mean_position_distance_mm=float(np.sqrt(a[:,:3].sum(1)).mean()*1000),
        orientation_mean_deg=float(a[:,3].mean()),gripper_mae=float(a[:,4].mean()),
        gripper_mse=float(a[:,5].mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--seed',type=int,default=12345)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for this evaluation')
    config=json.loads((a.runs/'dom/run_config.json').read_text())
    root=Path(config['local_verified_dataset'])
    candidates=json.loads((root/'meta/splits.json').read_text())['validation']
    episodes=sorted(np.random.default_rng(a.seed).choice(candidates,a.episodes,replace=False).tolist())
    report=dict(episodes=episodes,split='validation',seed=a.seed,
        dataset=str(root),dataset_revision=config['revision'],
        protocol='All quality/history-eligible anchors; clean inputs; one independent seeded action sample per anchor; padded future steps excluded; frame-weighted aggregates.',models={})
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    for family in ['dom','domv','krv100']:
        c=json.loads((a.runs/family/'run_config.json').read_text())
        best=json.loads((a.runs/family/'best_checkpoint.json').read_text()); checkpoint=Path(best['path'])
        checksums={n:hashlib.sha256((checkpoint/n).read_bytes()).hexdigest() for n in ['config.json','model.safetensors']}
        model,cfg,train,_=build_training_components(SimpleNamespace(
            bundle=PROJECT/'scratch/real_robot/finetuning-checkpoints',checkpoint=checkpoint,
            checkpoint_id=f'{family}/epoch-{best["epoch"]:03d}',expected_sha256=checksums,
            dataset=root,revision=c['revision'],family=family))
        del train
        data=LeRobotDataset(c['repo_id'],root=root,split='validation',episodes=episodes,
            revision=c['revision'],euler_delta_convention=c['euler_delta_convention'],
            required_features=['observation.images.opst_cam','observation.images.wrist_cam',
                'observation.state','observation.environment_state','action'],
            delta_timestamps={'observation':c['observation_history'],'action':list(range(cfg.chunk_size))})
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        loader=torch.utils.data.DataLoader(data,batch_size=a.batch_size,shuffle=False,num_workers=0)
        model.cuda().eval(); all_rows=[]; first=[]; by_episode={}; horizon=[[] for _ in range(cfg.chunk_size)]
        samples=[]; done=0
        with torch.inference_mode():
            for batch in loader:
                ids=batch['episode_index'].tolist(); frames=batch['frame_index'].tolist()
                batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
                state=batch['observation.state'][:,-1:,:].clone()
                truth=absolute_pose(batch['action'],state,cfg.euler_delta_convention)
                # Explicit per-anchor noise makes sampling identical across families and batch sizes.
                noise=torch.stack([torch.randn((cfg.chunk_size,cfg.max_action_dim),
                    generator=torch.Generator(device='cuda').manual_seed(a.seed+ep*100000+fr),device='cuda')
                    for ep,fr in zip(ids,frames)])
                model.reset()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    pred=model.predict_action_chunk(model_inputs(batch,cfg),noise=noise)
                if cfg.use_delta_action:pred=absolute_pose(pred,state,cfg.euler_delta_convention)
                pred=pred.float().cpu().numpy(); truth=truth.float().cpu().numpy()
                valid=(~batch['action_is_pad']).cpu().numpy()
                if not np.isfinite(pred).all():raise RuntimeError('Nonfinite predictions')
                diff=pred-truth
                r1=Rotation.from_euler('xyz',pred[:,:,3:6].reshape(-1,3))
                r2=Rotation.from_euler('xyz',truth[:,:,3:6].reshape(-1,3))
                angles=np.rad2deg((r1.inv()*r2).magnitude()).reshape(pred.shape[:2])
                errors=np.concatenate([diff[:,:,:3]**2,angles[:,:,None],abs(diff[:,:,6:7]),diff[:,:,6:7]**2],axis=2)
                for i,(ep,fr) in enumerate(zip(ids,frames)):
                    row=errors[i,valid[i]]; all_rows.append(row); by_episode.setdefault(ep,[]).append(row)
                    if valid[i,0]:first.append(errors[i,:1])
                    for h in range(cfg.chunk_size):
                        if valid[i,h]:horizon[h].append(errors[i,h:h+1])
                    samples.append(dict(episode=ep,frame=fr,prediction=pred[i].tolist(),target=truth[i].tolist(),valid=valid[i].tolist()))
                done+=len(ids)
                if done==len(ids) or done%80==0:print(f'{family}: {done}/{len(data)} anchors',flush=True)
        result=dict(checkpoint=str(checkpoint),checkpoint_sha256=checksums,anchors=done,
            full_horizon=metrics(all_rows),first_action=metrics(first),
            per_episode={str(ep):metrics(rows) for ep,rows in by_episode.items()},
            per_horizon=[metrics(rows) if rows else None for rows in horizon])
        (a.output/f'{family}.json').write_text(json.dumps(result,indent=2)+'\n')
        with (a.output/f'{family}-predictions.jsonl').open('w') as stream:
            for row in samples:stream.write(json.dumps(row)+'\n')
        report['models'][family]=result
        (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f'{family}: {json.dumps(result["full_horizon"])}',flush=True)
        del model; torch.cuda.empty_cache()

if __name__=='__main__':main()
