"""Device-free GPU check of guided RTC on two saved, aligned real observations."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from real_family_policy_worker import load_policy,infer
from real_robot.image_preprocessing import load_spec


def read(path):
    with np.load(path,allow_pickle=False) as archive:
        payload={k:archive[k] for k in archive.files if k!='metadata'}
        return payload,json.loads(str(archive['metadata']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--image-spec',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    paths=sorted(args.inputs.glob('*.npz'));pair=None
    for a,b in zip(paths,paths[1:]):
        old,oldmeta=read(a);current,meta=read(b)
        offset=meta['index']-oldmeta['index']
        if meta['token']==oldmeta['token'] and 1<=offset<=10:
            pair=(a,b,old,current,meta,offset);break
    if pair is None:raise ValueError('Need two saved observations from one trial with overlapping action horizons')
    a,b,old,payload,meta,offset=pair
    model,selection=load_policy(args.campaign,'smolvla_baseline',args.checkpoint)
    spec=load_spec(args.image_spec)
    noise=torch.randn(1,20,32,device='cuda',generator=torch.Generator(device='cuda').manual_seed(12345))
    unchanged=noise.clone();original=model.native.policy.model.denoise_step
    off=infer(model,payload,meta,spec,noise)
    empty=dict(mode='guided');empty_result=infer(model,payload,meta,spec,noise,rtc=empty)
    np.testing.assert_array_equal(off,empty_result)
    payload['rtc_prefix_actions']=old['actions'][offset:].copy()
    times=[]
    for _ in range(4):
        rtc=dict(mode='guided',inference_delay_steps=4,execution_horizon=10,max_guidance_weight=10.)
        torch.cuda.synchronize();start=time.monotonic()
        guided=infer(model,payload,meta,spec,noise,rtc=rtc)
        torch.cuda.synchronize();times.append(time.monotonic()-start)
    assert rtc['guided_denoise_steps']==model.native.policy.config.num_steps
    assert model.native.policy.model.denoise_step==original
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    torch.testing.assert_close(noise,unchanged,rtol=0,atol=0)
    report=dict(passed=True,hardware_accessed=False,selection=selection,previous_input=str(a),current_input=str(b),
        source_index_offset=offset,rtc=rtc,empty_prefix_native_max_error=float(abs(off-empty_result).max()),
        guided_finite=bool(np.isfinite(guided).all()),guided_action_shape=list(guided.shape),
        guided_change_xyz_mm=float(np.linalg.norm(guided[:,:3]-off[:,:3],axis=1).mean()*1000),
        inference_seconds=times,warmed_median_seconds=float(np.median(times[1:])),
        action_horizon_seconds=.8,checkpoint_frozen=True,
        interpretation='Offline saved observations only; does not establish physical rollout success.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
