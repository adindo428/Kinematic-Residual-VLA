#!/usr/bin/env python3
"""Paired four-model held-out imitation; no grasp-success or artificial offline OOD."""
import argparse,csv,itertools,json
from pathlib import Path
import numpy as np
import torch
from tracking_checkpoint import load_checkpoint,checkpoint_files
from tracking_variants import checkpoints_from_args,validate_variant,VARIANT_LABELS
from tracking_recorded import DEFAULT_DATASET,read_episode,make_batch,noise_for
from tracking_online_common import write_json,sha256
from tracking_visuals import plot_offline

def metrics(pred,target):
    e=(pred.astype(np.float64)-target.astype(np.float64))**2
    return {'action_mse':float(e.mean()),'xyz_imitation_mse':float(e[...,:3].mean()),
        'xyz_coordinate_rmse_mm':float(np.sqrt(e[...,:3].mean())*1000),
        'xyz_distance_rmse_mm':float(np.sqrt(e[...,:3].sum(-1).mean())*1000),
        'rotation_channel_mse':float(e[...,3:6].mean()),'gripper_mse':float(e[...,6].mean())}

def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    eps=json.loads((a.dataset/'meta/split_assignments.json').read_text())['test']
    report={'evaluation':'paired offline IID imitation','closed_loop':False,'episodes':eps,'seed':42,'action_steps':20,
            'velocity_source':'recorded current object velocity; not camera-estimated','models':{},'pairwise':{}}
    rows=[];saved={}
    for model,root in checkpoints_from_args(a).items():
        _,_,weights,_=checkpoint_files(root,require_complete=True)
        policy=load_checkpoint(root);validate_variant(policy.config,model)
        assert policy.config.chunk_size==20
        preds=[];targets=[];episode_ids=[];anchor_ids=[]
        for ep in eps:
            data=read_episode(a.dataset,ep);anchors=list(range(len(data[0])-20+1))
            ep_pred=[];ep_target=[]
            for start in range(0,len(anchors),a.batch_size):
                ids=anchors[start:start+a.batch_size];batch=make_batch(data,ids)
                policy.reset()
                with torch.inference_mode():pred=policy.predict_action_chunk(batch,noise=noise_for(ep,ids))
                assert torch.isfinite(pred).all()
                ep_pred.append(pred.cpu().numpy());ep_target.append(batch['action'].cpu().numpy())
            pred=np.concatenate(ep_pred);target=np.concatenate(ep_target)
            rows.append({'model':model,'episode':ep,'valid_anchors':len(anchors),**metrics(pred,target)})
            preds.append(pred);targets.append(target);episode_ids.extend([ep]*len(anchors));anchor_ids.extend(anchors)
            print('OFFLINE',model,ep,rows[-1],flush=True)
        pred=np.concatenate(preds);target=np.concatenate(targets)
        report['models'][model]={'display_name':VARIANT_LABELS[model],'checkpoint':str(weights),'sha256':sha256(weights),
            'valid_anchors':len(pred),**metrics(pred,target)}
        np.savez_compressed(a.output/(VARIANT_LABELS[model]+'_predictions.npz'),prediction=pred,target=target,episode=np.array(episode_ids),anchor=np.array(anchor_ids))
        saved[model]={r['episode']:r['xyz_distance_rmse_mm'] for r in rows if r['model']==model}
        del policy;torch.cuda.empty_cache()
    for left,right in itertools.combinations(report['models'],2):
        delta=np.array([saved[right][ep]-saved[left][ep] for ep in eps]);rng=np.random.default_rng(42)
        boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
        report['pairwise'][right+'_minus_'+left]={'mean_episode_xyz_distance_rmse_delta_mm':float(delta.mean()),'episode_bootstrap_ci95_mm':np.quantile(boot,[.025,.975]).tolist()}
    with (a.output/'per_episode.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(a.output/'comparison.json',report);plot_offline(report,rows,a.output/'iid_xyz_error.png')
    (a.output/'README.md').write_text('# Held-out offline imitation\n\nAll four validation-selected checkpoints use identical recorded test anchors and seeded noise. '
        'Velocity means the current recorded robot-base velocity, not a camera estimate. No test frames were used for model selection. '
        'Action MSE mixes translation, Euler rotation and gripper channels; XYZ errors are reported separately. '
        'These are IID imitation results; faster OOD is evaluated in live_tracking. No grasp-success metric is used.\n')
    print('FOUR-MODEL OFFLINE EVALUATION COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','velocity','residual','coordinate','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=DEFAULT_DATASET);p.add_argument('--batch-size',type=int,default=8)
    main(p.parse_args())
