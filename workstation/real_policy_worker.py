"""GPU process for continuous learned-policy inference; opens no robot devices."""
import argparse
import hashlib
from contextlib import ExitStack
import json
from pathlib import Path
import time
from unittest.mock import patch
import sys
import numpy as np
import torch
from safetensors.torch import load_model
from smoke_real_finetune import DynamicVLAPolicy, get_policy_cfg, AutoConfig, AutoTokenizer, PROJECT
sys.path.insert(0,str(PROJECT))
from real_robot.image_preprocessing import load_spec, transform
from utils.real_robot_contract import absolute_pose
from real_robot.policy_recording import InferenceArchive
from real_robot.policy_precision import apply_checkpoint_precision


def atomic_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def load_policy(checkpoint,bundle):
    cfg=get_policy_cfg(cfg_file=checkpoint/'config.json');cfg.enable_streaming=False
    # Training configs store False because buffers were initialized from the dataset.
    # Inference must construct locally and restore those same saved buffers.
    cfg.use_checkpoint_normalization=True
    assert cfg.vlm_model_name=='HuggingFaceTB/SmolLM2-360M' and cfg.fastvlm_inference_mode
    assert cfg.euler_delta_convention=='wrapped_xyz_v1' and cfg.require_tracking_quality
    assert cfg.n_obs_steps==2 and cfg.chunk_size==20
    with ExitStack() as stack:
        for cls in [AutoConfig,AutoTokenizer]:
            original=cls.from_pretrained
            def local(name,*pos,_original=original,**kw):
                asset=bundle/'assets'/str(name)
                if asset.is_dir():name=str(asset);kw['local_files_only']=True
                return _original(name,*pos,**kw)
            stack.enter_context(patch.object(cls,'from_pretrained',side_effect=local))
        policy=DynamicVLAPolicy(cfg)
    # Widen before loading FP32 master snapshots, preserving the trained values.
    apply_checkpoint_precision(policy,checkpoint)
    load_model(policy,str(checkpoint/'model.safetensors'),strict=True)
    return policy.eval().cuda(),cfg


def prepare_live_batch(cfg,payload,meta,spec,device='cuda'):
    batch={}
    for role,key in [('front','observation.images.opst_cam'),('wrist','observation.images.wrist_cam')]:
        frames=[transform(frame,spec,role)[:,:,::-1].copy() for frame in payload[role]]
        batch[key]=torch.from_numpy(np.stack(frames)).permute(0,3,1,2).contiguous().unsqueeze(0).to(device).float()/255
    for name,key in [('state','observation.state'),('environment','observation.environment_state')]:
        if name=='environment' and not (cfg.condition_on_object_velocity or cfg.enable_residual_flow):continue
        batch[key]=torch.as_tensor(payload[name],dtype=torch.float32,device=device).unsqueeze(0)
    for name in ['required','valid']:
        if not (cfg.condition_on_object_velocity or cfg.enable_residual_flow):continue
        key='dynamic_object.velocity_'+name
        batch[key]=torch.as_tensor(payload[name],dtype=torch.bool,device=device).unsqueeze(0)
    if getattr(cfg,'residual_phase_gate_mode','constant')=='causal_rules':
        gate=np.asarray(payload['phase_gate'])
        if gate.shape!=(cfg.chunk_size,1) or not np.isfinite(gate).all() or np.any((gate<0)|(gate>1)):raise ValueError('Invalid causal phase gate')
        batch[cfg.residual_phase_gate_key]=torch.as_tensor(gate,dtype=torch.float32,device=device).unsqueeze(0)
    batch['task']=[meta['instruction']]
    return batch


def infer(policy,cfg,payload,meta,spec,noise=None):
    batch=prepare_live_batch(cfg,payload,meta,spec)
    reference_state=batch['observation.state'][:,-1:,:].clone()
    with torch.inference_mode(),torch.autocast('cuda',dtype=getattr(policy,'execution_autocast_dtype',torch.bfloat16)):
        actions=policy._get_action_chunk(policy._prepare_batch(batch),noise=noise)
        if cfg.use_delta_action:
            actions=absolute_pose(actions,reference_state,cfg.euler_delta_convention)
    result=actions[0].float().cpu().numpy()
    if result.shape!=(20,7) or not np.isfinite(result).all():raise ValueError('Invalid model action chunk')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--exchange',type=Path,required=True)
    p.add_argument('--bundle',type=Path,default=PROJECT/'scratch/real_robot/finetuning-checkpoints')
    p.add_argument('--image-spec',type=Path,required=True);p.add_argument('--once',action='store_true')
    p.add_argument('--record-inputs',action='store_true',help='Losslessly archive each inferred input and full predicted chunk asynchronously')
    args=p.parse_args();args.exchange.mkdir(parents=True,exist_ok=True)
    policy,cfg=load_policy(args.checkpoint,args.bundle);spec=load_spec(args.image_spec)
    atomic_json(args.exchange/'worker.json',{'state':'ready','checkpoint':str(args.checkpoint.resolve()),'hardware_accessed':False})
    object_inputs_required=bool(cfg.condition_on_object_velocity or cfg.enable_residual_flow)
    payload_keys=['front','wrist','state']+(['environment','required','valid'] if object_inputs_required else [])
    if cfg.residual_phase_gate_mode=='causal_rules':payload_keys.append('phase_gate')
    previous=None;archive=InferenceArchive(args.exchange/'inference_inputs') if args.record_inputs else None
    try:
      with (args.exchange/'inference.jsonl').open('a',buffering=1) as log:
        while not (args.exchange/'STOP').exists():
            if archive is not None:archive.check()
            try:
                with np.load(args.exchange/'observation.npz',allow_pickle=False) as z:
                    meta=json.loads(str(z['metadata']));payload={k:z[k] for k in payload_keys}
            except FileNotFoundError:time.sleep(.005);continue
            key=(meta['token'],meta['index'])
            if key==previous:time.sleep(.002);continue
            previous=key
            if not 0<=time.monotonic()-meta['created_monotonic_s']<=.25:
                atomic_json(args.exchange/'worker.json',{'state':'waiting','reason':'observation stale'});continue
            started=time.monotonic();actions=infer(policy,cfg,payload,meta,spec);done=time.monotonic()
            result=dict(meta,inference_started_s=started,inference_finished_s=done,inference_seconds=done-started)
            with (args.exchange/'actions.tmp').open('wb') as f:np.savez(f,actions=actions,metadata=json.dumps(result))
            (args.exchange/'actions.tmp').replace(args.exchange/'actions.npz')
            atomic_json(args.exchange/'worker.json',{'state':'running','token':meta['token'],'index':meta['index'],'inference_seconds':done-started})
            # Publish control output first. Recording/compression never delays
            # that output or runs on the motor-control process.
            result['actions_absolute']=actions.tolist()
            result['observed_state']=payload['state'][-1].tolist()
            result['object_inputs_used_by_policy']=object_inputs_required
            if object_inputs_required:result['observed_environment']=payload['environment'][-1].tolist()
            if archive is not None:
                name=f'trial-{meta["trial"]:03d}-{hashlib.sha256(meta["token"].encode()).hexdigest()[:12]}-{meta["index"]:07d}.npz'
                result['input_archive']='inference_inputs/'+name
                archive.submit(name,dict(payload,actions=actions,metadata=np.array(json.dumps(result))))
            log.write(json.dumps(result)+'\n')
            if args.once:break
    finally:
        if archive is not None:archive.close()


if __name__=='__main__':main()
