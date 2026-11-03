#!/usr/bin/env python3
"""Local policy RPC worker in the existing DynamicVLA inference environment."""
import argparse
import collections
import copy
import json
import logging
from pathlib import Path
import time
import numpy as np
import safetensors.torch
import torch
import zmq
from tracking_checkpoint import load_checkpoint
from tracking_variants import VARIANT_LABELS, checkpoints_from_args, validate_variant
from tracking_online_common import sha256, write_json


class Worker:
    def __init__(self, checkpoints):
        self.checkpoints = checkpoints
        self.policy = None
        self.variant = None
        self.history = collections.deque(maxlen=3)

    def reset(self, variant, seed):
        if variant not in self.checkpoints:
            raise ValueError(f'Unknown model {variant}')
        if variant != self.variant:
            self.policy = None
            torch.cuda.empty_cache()
            root = Path(self.checkpoints[variant])
            weights = root/'model.best_validation_action_mse.safetensors'
            self.policy = load_checkpoint(root).cuda().eval()
            c = self.policy.config
            if c.n_obs_steps != 2 or c.n_action_steps != 20 or not c.use_delta_action or c.action_feature.shape != (7,):
                raise ValueError('Checkpoint does not match two-frame, 20-step delta Euler tracking contract')
            validate_variant(c, variant)
            if c.enable_residual_flow and self.policy.get_action_normalization_std(torch.device('cpu'),torch.float32) is None:
                raise ValueError('Missing residual action scale')
            self.metadata = {'variant':variant, 'display_name':VARIANT_LABELS[variant], 'formulation':c.residual_flow_formulation, 'weights':str(weights), 'weights_sha256':sha256(weights),
                             'config_sha256':sha256(root/'config.json'), 'neural_velocity_conditioning':bool(c.condition_on_object_velocity), 'analytic_velocity_drift':bool(c.enable_residual_flow)}
            self.variant = variant
        self.policy.reset(); self.history.clear(); self.seed=int(seed); self.next_frame=0
        return self.metadata

    @torch.no_grad()
    def observe(self, message):
        frame = int(message['frame'])
        if frame != self.next_frame:
            raise ValueError(f'Expected observation {self.next_frame}, got {frame}')
        self.next_frame += 1
        state = np.asarray(message['state'], dtype=np.float32)
        env = np.asarray(message['environment_state'], dtype=np.float32)
        if state.shape != (6,) or env.shape != (9,) or not np.isfinite(state).all() or not np.isfinite(env).all():
            raise ValueError('Invalid live state/velocity')
        self.history.append((state,env,message['images']))
        if frame % self.policy.config.n_action_steps:
            return {'replanned':False}
        entries = [self.history[max(0,len(self.history)-3)],self.history[-1]]
        batch = {'observation.state':torch.from_numpy(np.stack([x[0] for x in entries])[None]).cuda(),
                 'observation.environment_state':torch.from_numpy(np.stack([x[1] for x in entries])[None]).cuda(),
                 'task':['Hover above the moving object and tilt according to its lateral velocity.']}
        for camera in ('opst_cam','wrist_cam'):
            array=np.stack([x[2][camera] for x in entries])
            if array.shape != (2,360,480,3) or array.dtype!=np.uint8:
                raise ValueError(f'Invalid image shape/dtype: {camera}')
            batch[f'observation.images.{camera}']=torch.from_numpy(array.copy()).permute(0,3,1,2).contiguous()[None].cuda().float()/255.
        generator=torch.Generator(device='cpu').manual_seed(self.seed+frame)
        noise=torch.randn(1,self.policy.config.chunk_size,self.policy.config.max_action_dim,generator=generator).cuda()
        torch.cuda.synchronize(); start=time.perf_counter()
        self.policy.reset()
        chunk=self.policy.predict_action_chunk(batch,noise=noise)[0].cpu().numpy()
        if chunk.shape != (self.policy.config.chunk_size,7) or not np.isfinite(chunk).all():
            raise ValueError('Invalid predicted action chunk')
        chunk[:,:6] += state[None,:6]
        return {'replanned':True,'chunk_absolute':chunk,'inference_seconds':time.perf_counter()-start,
                'anchor_pose':state,'anchor_time':float(message['time'])}


def smoke(args):
    import pyarrow.parquet as pq
    import av
    root=args.dataset
    split=json.loads((root/'meta/split_assignments.json').read_text())
    episode=split['test'][0]
    table=pq.read_table(root/f'data/chunk-{episode//1000:03d}/episode_{episode:06d}.parquet').to_pylist()
    images={}
    for camera in ('opst_cam','wrist_cam'):
        path=root/f'videos/chunk-{episode//1000:03d}/observation.images.{camera}/episode_{episode:06d}.mp4'
        with av.open(str(path)) as container:
            images[camera]=[]
            for frame in container.decode(video=0):
                images[camera].append(frame.to_ndarray(format='rgb24'))
                if len(images[camera])==21:break
        if len(images[camera])!=21:raise ValueError('Smoke episode has too few frames')
    worker=Worker(checkpoints_from_args(args))
    reports={}
    for variant in worker.checkpoints:
        metadata=worker.reset(variant,42)
        replies=[]
        for frame in range(21):
            sample=table[frame]
            reply=worker.observe({'frame':frame,'time':frame/25,
                'state':np.array(sample['observation.state'],dtype=np.float32),
                'environment_state':np.array(sample['observation.environment_state'],dtype=np.float32),
                'images':{key:images[key][frame] for key in images}})
            if reply['replanned']:
                replies.append({'frame':frame,'shape':list(reply['chunk_absolute'].shape),'latency_s':reply['inference_seconds']})
        assert [r['frame'] for r in replies]==[0,20]
        reports[variant]={'checkpoint':metadata,'replans':replies}
    write_json(args.output/'policy_smoke.json',reports)
    logging.info('POLICY ADAPTER SMOKE PASSED %s',reports)


def main(args):
    if args.smoke:
        smoke(args); return
    context=zmq.Context()
    socket=context.socket(zmq.REP); socket.setsockopt(zmq.LINGER,0); socket.bind(args.endpoint)
    worker=Worker(checkpoints_from_args(args))
    logging.info('Policy worker listening at %s',args.endpoint)
    while True:
        message=socket.recv_pyobj()
        try:
            if message['cmd']=='reset': result=worker.reset(message['variant'],message['seed'])
            elif message['cmd']=='observe': result=worker.observe(message)
            elif message['cmd']=='stop': socket.send_pyobj({'ok':True}); break
            else: raise ValueError('Unknown command')
            socket.send_pyobj({'ok':True,'result':result})
        except Exception as exc:
            logging.exception('Policy RPC failed')
            socket.send_pyobj({'ok':False,'error':str(exc)})
    socket.close();context.term()


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline');p.add_argument('--residual');p.add_argument('--coordinate-velocity')
    p.add_argument('--coordinate');p.add_argument('--velocity');p.add_argument('--endpoint');p.add_argument('--smoke',action='store_true')
    p.add_argument('--output',type=Path,default=Path('.'))
    p.add_argument('--dataset',type=Path,default=Path('/scratch/aparame/.cache/huggingface/lerobot/adi2440/dom-franka-tracking'))
    a=p.parse_args()
    if not any(getattr(a,key,None) for key in VARIANT_LABELS): p.error('At least one checkpoint required')
    if not a.smoke and not a.endpoint: p.error('--endpoint required')
    main(a)
