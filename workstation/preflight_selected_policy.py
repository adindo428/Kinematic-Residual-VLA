"""Exercise the DOM live worker on one saved real observation; no hardware IO."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from real_policy_worker import load_policy,infer
from real_robot.image_preprocessing import load_spec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--saved-input',type=Path,required=True)
    p.add_argument('--image-spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parents[1]/'scratch/real_robot/finetuning-checkpoints')
    a=p.parse_args();policy,cfg=load_policy(a.checkpoint,a.bundle);spec=load_spec(a.image_spec)
    with np.load(a.saved_input,allow_pickle=False) as z:
        keys=['front','wrist','state']+(['environment','required','valid'] if cfg.condition_on_object_velocity or cfg.enable_residual_flow else [])
        payload={k:z[k] for k in keys};meta=json.loads(str(z['metadata']))
    torch.manual_seed(12345);infer(policy,cfg,payload,meta,spec);times=[]
    for _ in range(5):
        torch.cuda.synchronize();start=time.monotonic();actions=infer(policy,cfg,payload,meta,spec);torch.cuda.synchronize();times.append(time.monotonic()-start)
    result=dict(passed=True,hardware_accessed=False,checkpoint=str(a.checkpoint),saved_input=str(a.saved_input),
        action_shape=list(actions.shape),finite=bool(np.isfinite(actions).all()),warmed_inference_seconds=times,
        median_inference_seconds=float(np.median(times)),interpretation='Saved observation only; physical pilot still required.')
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
