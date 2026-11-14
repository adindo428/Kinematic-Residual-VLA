"""Native family inference process: no motor, camera or robot-device access."""
import argparse
from collections import deque
from contextlib import nullcontext
import math
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from family_policy import load_released_policy
from real_robot.family_live import family_batch,select_family_checkpoint,family_object_inputs_required
from real_robot.image_preprocessing import load_spec
from real_robot.policy_recording import InferenceArchive


def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def load_policy(campaign_path,family,checkpoint,expected_sha=None):
    selected,record,campaign=select_family_checkpoint(campaign_path,family,'best')
    checkpoint=Path(checkpoint).resolve()
    if checkpoint!=selected:
        selected,record,campaign=select_family_checkpoint(campaign_path,family,'last')
        if checkpoint!=selected:raise ValueError('Checkpoint is not a selected completed snapshot')
    checksum=digest(checkpoint)
    if expected_sha and expected_sha!=checksum:raise ValueError('Frozen experiment checkpoint changed')
    root=Path(campaign_path).resolve().parent;entry=next(e for e in campaign['entries'] if e['family']==family)
    stats=json.loads((root/'real_statistics.json').read_text())
    model=load_released_policy(root,entry,stats,device='cpu')
    if getattr(model,'velocity_conditioning',None)!=entry.get('velocity_conditioning'):
        raise ValueError('This runtime does not implement the checkpoint neural velocity conditioning contract')
    snapshot=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if snapshot['architecture_signature']!=model.architecture_signature or snapshot['statistics']!=stats:
        raise ValueError('Selected architecture or normalization differs from training')
    if snapshot.get('velocity_conditioning')!=entry.get('velocity_conditioning'):
        raise ValueError('Selected checkpoint conditioning differs from requested runtime')
    if snapshot['epoch']!=record['epoch']:raise ValueError('Selected epoch differs from checkpoint')
    if record.get('weights',snapshot['selected_weights'])!=snapshot['selected_weights']:raise ValueError('Selected raw/EMA weights differ')
    model.native.load_state_dict(snapshot['model'],strict=True);del snapshot
    model.to('cuda').eval().requires_grad_(False);model.native.policy.config.device='cuda'
    record.update(checkpoint_sha256=checksum,architecture_signature=model.architecture_signature)
    return model,record


def infer(model,payload,meta,spec,noise=None,reference=False,rtc=None):
    needs_object=bool(model.native.recipe['kr']) or bool(getattr(model,'velocity_conditioning',None))
    batch=family_batch(payload,meta,spec,object_inputs_required=needs_object)
    guided=rtc is not None and rtc.get('mode')=='guided'
    if guided:
        from real_robot.rtc import guide_smolvla
        context=guide_smolvla(model,batch,payload.get('rtc_prefix_actions',np.empty((0,7),np.float32)),
            inference_delay=rtc.get('inference_delay_steps',4),execution_horizon=rtc.get('execution_horizon',10),
            max_guidance_weight=rtc.get('max_guidance_weight',10.))
    else:context=nullcontext(None)
    # Guidance differentiates only the noisy action suffix; checkpoint weights
    # remain frozen. inference_mode would prevent the required suffix VJP.
    with (torch.no_grad() if guided else torch.inference_mode()),torch.autocast('cuda',dtype=torch.bfloat16):
        with context as diagnostics:
            actions=model.sample_absolute(batch,noise=noise,reference=reference)
        if guided:rtc.update(diagnostics)

    result=actions[0].float().cpu().numpy()
    if result.shape!=(20,7) or not np.isfinite(result).all():raise ValueError('Invalid absolute Cartesian action chunk')
    return result


def rtc_observation_ready(meta,completed):
    """Wait for controller acknowledgement so guidance sees the accepted plan."""
    return (completed is None or meta['token']!=completed[0]
        or meta.get('rtc_plan_source_index',-1)>=completed[1])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--family',required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--expected-sha256')
    p.add_argument('--image-spec',type=Path,required=True);p.add_argument('--exchange',type=Path)
    p.add_argument('--saved-input',type=Path);p.add_argument('--preflight-output',type=Path)
    p.add_argument('--record-inputs',action='store_true');p.add_argument('--once',action='store_true')
    p.add_argument('--rtc',choices=['off','guided'],default='off')
    p.add_argument('--rtc-execution-horizon',type=int,default=10)
    p.add_argument('--rtc-max-guidance-weight',type=float,default=10.)
    args=p.parse_args()
    if args.rtc!='off' and not args.family.startswith('smolvla_'):p.error('Guided RTC requires native SmolVLA')
    if not 1<=args.rtc_execution_horizon<=20 or not math.isfinite(args.rtc_max_guidance_weight) or args.rtc_max_guidance_weight<=0:p.error('Invalid RTC settings')
    model,selection=load_policy(args.campaign,args.family,args.checkpoint,args.expected_sha256)
    spec=load_spec(args.image_spec)
    needs_object=family_object_inputs_required(args.family,selection.get('velocity_conditioning'))
    payload_keys=['front','wrist','state']+(['environment','required','valid'] if needs_object else [])
    selection['object_inputs_used_by_policy']=needs_object
    if args.saved_input:
        with np.load(args.saved_input,allow_pickle=False) as z:
            payload={k:z[k] for k in payload_keys};meta=json.loads(str(z['metadata']))
            if 'rtc_prefix_actions' in z:payload['rtc_prefix_actions']=z['rtc_prefix_actions']
        noise=torch.randn(1,20,32,device='cuda',generator=torch.Generator(device='cuda').manual_seed(12345))
        explicit=infer(model,payload,meta,spec,noise);reference=infer(model,payload,meta,spec,noise,True)
        np.testing.assert_allclose(explicit,reference,atol=2e-5,rtol=2e-5)
        rtc=dict(mode=args.rtc,execution_horizon=args.rtc_execution_horizon,max_guidance_weight=args.rtc_max_guidance_weight)
        times=[]
        for _ in range(5):
            torch.cuda.synchronize();start=time.monotonic();infer(model,payload,meta,spec,noise,rtc=rtc);torch.cuda.synchronize();times.append(time.monotonic()-start)
        write(args.preflight_output,dict(passed=True,hardware_accessed=False,selection=selection,saved_input=str(args.saved_input),
            rtc=rtc,action_shape=list(explicit.shape),finite=True,native_sampler_max_error=float(abs(explicit-reference).max()),
            warmed_inference_seconds=times,median_inference_seconds=float(np.median(times)),
            interpretation='Saved real observation, not a physical rollout; controller drops elapsed 25Hz chunk entries.'))
        return
    if args.exchange is None:p.error('--exchange required for live inference')
    args.exchange.mkdir(parents=True,exist_ok=True)
    write(args.exchange/'worker.json',dict(state='ready',hardware_accessed=False,selection=selection))
    previous=None;completed=None;delays=deque(maxlen=20);archive=InferenceArchive(args.exchange/'inference_inputs') if args.record_inputs else None
    try:
        with (args.exchange/'inference.jsonl').open('a',buffering=1) as log:
            while not (args.exchange/'STOP').exists():
                if archive is not None:archive.check()
                try:
                    with np.load(args.exchange/'observation.npz',allow_pickle=False) as z:
                        payload={k:z[k] for k in payload_keys};meta=json.loads(str(z['metadata']))
                        if 'rtc_prefix_actions' in z:payload['rtc_prefix_actions']=z['rtc_prefix_actions']
                except FileNotFoundError:time.sleep(.005);continue
                key=meta['token'],meta['index']
                if args.rtc=='guided' and not rtc_observation_ready(meta,completed):time.sleep(.002);continue
                if key==previous:time.sleep(.002);continue
                if previous is not None and key[0]!=previous[0]:delays.clear()
                previous=key
                if not 0<=time.monotonic()-meta['created_monotonic_s']<=.25:
                    write(args.exchange/'worker.json',dict(state='waiting',reason='observation stale'));continue
                start=time.monotonic()
                delay_steps=math.ceil(float(np.percentile(delays,95))/.04)+1 if delays else 4
                rtc=dict(mode=args.rtc,inference_delay_steps=delay_steps,execution_horizon=args.rtc_execution_horizon,
                    max_guidance_weight=args.rtc_max_guidance_weight)
                actions=infer(model,payload,meta,spec,rtc=rtc);done=time.monotonic()
                total_delay=done-meta['created_monotonic_s'];delays.append(total_delay)
                result=dict(meta,inference_started_s=start,inference_finished_s=done,inference_seconds=done-start,
                    policy_observation_offsets=[0],family=args.family,velocity_conditioning=selection['velocity_conditioning'],
                    rtc=rtc,observation_to_result_seconds=total_delay,async_inference=True)
                with (args.exchange/'actions.tmp').open('wb') as f:np.savez(f,actions=actions,metadata=json.dumps(result))
                (args.exchange/'actions.tmp').replace(args.exchange/'actions.npz')
                completed=key
                write(args.exchange/'worker.json',dict(state='running',token=meta['token'],index=meta['index'],inference_seconds=done-start))
                result.update(actions_absolute=actions.tolist(),observed_state=payload['state'][-1].tolist(),object_inputs_used_by_policy=needs_object)
                if needs_object:result['observed_environment']=payload['environment'][-1].tolist()
                if archive is not None:
                    name=f'trial-{meta["trial"]:03d}-{hashlib.sha256(meta["token"].encode()).hexdigest()[:12]}-{meta["index"]:07d}.npz'
                    result['input_archive']='inference_inputs/'+name
                    archive.submit(name,dict(payload,actions=actions,metadata=np.array(json.dumps(result))))
                log.write(json.dumps(result)+'\n')
                if args.once:break
    finally:
        if archive is not None:archive.close()


if __name__=='__main__':main()
