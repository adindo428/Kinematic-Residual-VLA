"""Compare live/validation decoders on recorded inputs; no camera or robot IO."""
import argparse,json
from pathlib import Path
import sys
import numpy as np
import torch
from torchcodec.decoders import VideoDecoder
import pyarrow.parquet as pq
from real_policy_worker import load_policy,infer,prepare_live_batch
from real_robot.image_preprocessing import load_spec
from real_robot.action_objective import sampled_absolute_actions
from utils.real_robot_contract import absolute_pose

PROJECT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    runs=PROJECT/'scratch/real_robot/finetuning-runs/action-v5-fullsampler-20261110'
    contract=read(runs/'launch_contract.json');dataset=Path(contract['local_verified_dataset']);spec=load_spec(dataset/'meta/image_preprocessing.json')
    session=PROJECT/'scratch/real_robot/real-experiments/post-smolvla-v6-20261112/sessions/pilot-striped_ball-iid_slow-krv100-20261113T003328Z'
    live_file=sorted((session/'exchange/inference_inputs').glob('*.npz'))[0]
    with np.load(live_file,allow_pickle=False) as z:
        payload={k:z[k] for k in ['front','wrist','state','environment','required','valid']};meta=json.loads(str(z['metadata']))
    metadata={x['episode_index']:x for x in map(json.loads,(dataset/'meta/episodes.jsonl').read_text().splitlines())}
    usable=read(dataset/'meta/quality_policy.json')['usable_frames'];splits=read(dataset/'meta/splits.json')
    candidates={};by_ep={}
    # Find held-out descent/closing anchors with actual low commanded targets.
    for ep in splits['validation']:
        t=pq.read_table(dataset/f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet',columns=['action','observation.state','observation.environment_state'])
        arrays={key:np.asarray(t[key].to_pylist(),dtype=np.float32) for key in t.column_names}
        q=list(map(json.loads,(dataset/f'quality/episode_{ep:06d}.jsonl').read_text().splitlines()))
        valid=set(usable[str(ep)]);task=json.loads(metadata[ep]['tasks'][0]);name=task['objects']
        for i in sorted(valid):
            if max(0,i-2) not in valid:continue
            current=arrays['observation.state'][i,2];future=arrays['action'][i:i+20,2]
            kind='descent' if current>.04 and future.min()<current-.03 else 'closing' if q[i]['manipulation_phase']=='closing_candidate' and arrays['action'][i,2]<.05 else None
            if kind is None:continue
            key=name,kind
            # Up to two spaced samples per object/stage.
            previous=candidates.setdefault(key,[])
            if len(previous)<2 and all(ep!=old[0] or abs(i-old[1])>20 for old in previous):previous.append((ep,i))
        by_ep[ep]=(arrays,q,task)
    selected=[(name,kind,ep,i) for (name,kind),rows in candidates.items() for ep,i in rows]
    report=dict(hardware_accessed=False,live_input=str(live_file),models={})
    for family in ['dom','domv','krv100']:
        checkpoint=runs/'evaluation'/family;model,cfg=load_policy(checkpoint,PROJECT/'scratch/real_robot/finetuning-checkpoints')
        noise=torch.randn(1,20,32,device='cuda',generator=torch.Generator(device='cuda').manual_seed(12345))
        live_prediction=infer(model,cfg,payload,meta,spec,noise)
        batch=prepare_live_batch(cfg,payload,meta,spec);state=batch['observation.state'].clone()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):validation_prediction=sampled_absolute_actions(model,batch,noise)
        np.testing.assert_allclose(live_prediction,validation_prediction[0].float().cpu().numpy(),atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(batch['observation.state'],state,atol=0,rtol=0)
        result=dict(live_validation_decode_max_error=float(abs(live_prediction-validation_prediction[0].float().cpu().numpy()).max()),
            live_state_unchanged=True,live_current_z_mm=float(state[0,-1,2])*1000,
            live_predicted_z_mm=(live_prediction[:,2]*1000).tolist(),heldout_stages=[])
        decoders={}
        for task_name,kind,ep,i in selected:
            arrays,q,task=by_ep[ep];history=[max(0,i-2),i]
            item={'observation.state':torch.from_numpy(arrays['observation.state'][history])[None].cuda(),
                  'task':[task['instruction']]}
            if cfg.condition_on_object_velocity or cfg.enable_residual_flow:
                item['observation.environment_state']=torch.from_numpy(arrays['observation.environment_state'][history])[None].cuda()
                item['dynamic_object.velocity_required']=torch.tensor([[q[k]['velocity_required'] for k in history]],device='cuda')
                from real_robot.static_object import quality_valid as static_quality_valid
                item['dynamic_object.velocity_valid']=torch.tensor([[bool(q[k]['velocity_estimated'] and not q[k]['velocity_zero_filled']) or q[k].get('velocity_repaired',False) or static_quality_valid(q[k]) for k in history]],device='cuda')
            for camera in ['opst_cam','wrist_cam']:
                key=ep,camera
                if key not in decoders:decoders[key]=VideoDecoder((dataset/f'videos/chunk-{ep//1000:03d}/observation.images.{camera}/episode_{ep:06d}.mp4').read_bytes(),num_ffmpeg_threads=1)
                item['observation.images.'+camera]=decoders[key].get_frames_at(history).data[None].contiguous().cuda().float()/255
            raw_state=item['observation.state'][:,-1:,:].clone()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=model._get_action_chunk(model._prepare_batch(item),noise=noise)
                prediction=absolute_pose(prediction,raw_state,cfg.euler_delta_convention)
            prediction=prediction[0].float().cpu().numpy();truth=arrays['action'][i:i+20]
            result['heldout_stages'].append(dict(task=task_name,stage=kind,episode=ep,frame=i,current_z_mm=float(raw_state[0,0,2])*1000,
                target_z_mm=(truth[:,2]*1000).tolist(),predicted_z_mm=(prediction[:len(truth),2]*1000).tolist(),
                mean_signed_z_error_mm=float((prediction[:len(truth),2]-truth[:,2]).mean()*1000)))
        report['models'][family]=result
        (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(family,'decode parity',result['live_validation_decode_max_error'],'live z',result['live_current_z_mm'],'prediction min z',min(result['live_predicted_z_mm']),flush=True)
        for row in result['heldout_stages']:print(' ',row['task'],row['stage'],'state',round(row['current_z_mm'],1),'target min',round(min(row['target_z_mm']),1),'pred min',round(min(row['predicted_z_mm']),1),'bias',round(row['mean_signed_z_error_mm'],1),flush=True)
        del model,item,batch;torch.cuda.empty_cache()


if __name__=='__main__':main()
