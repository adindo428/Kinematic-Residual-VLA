#!/usr/bin/env python3
"""Strict trained-backbone inference with DOM's latency-aware action queue."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import logging
from pathlib import Path
import sys
import time
from types import SimpleNamespace


class Adapter:
    def __init__(self,model,recipe,torch):
        self.model,self.recipe,self.torch=model,recipe,torch
        self.native=recipe['method']=='dom'
        cfg=model.policy.config
        self.config=SimpleNamespace(input_features=cfg.input_features,output_features=cfg.output_features,
            n_action_steps=20,n_obs_steps=2 if self.native else 1,enable_streaming=True,use_delta_action=True)
        self.executor=ThreadPoolExecutor(max_workers=1)
        self.future=None;self.generation=0;self.reset()

    def reset(self):
        self.generation+=1;self.queue=deque();self.action_index=-1
        # The worker owns the model and resets it before each prediction. A
        # prior trial's in-flight result is discarded by its generation tag.

    def infer(self,obs,generation):
        torch=self.torch;begin=time.monotonic()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            if self.native:
                batch={k:(v.to('cuda') if torch.is_tensor(v) else v) for k,v in obs.items()}
            else:
                state=obs['observation.state'][:,-1,:6]
                env=obs['observation.environment_state'][:,-1]
                batch={**{k:v[:,-1] for k,v in obs.items() if k.startswith('observation.images.')},
                    'observation.state':state,'action':torch.zeros(1,20,7),
                    'action_is_pad':torch.zeros(1,20,dtype=torch.bool),
                    'velocity':env[:,6:9],'offset':torch.zeros(1,dtype=torch.long),'task':obs['task']}
                self.model.policy.reset()
            prediction=self.model.predict(batch)
            if tuple(prediction.shape)!=(1,20,7) or not torch.isfinite(prediction).all():
                raise ValueError('Invalid reconstructed physical action chunk')
            # Both native DOM predict_action_chunk and the cross-VLA wrapper
            # return physical deltas. KR is reconstructed inside predict(),
            # then the observed reference pose is added exactly once here.
            prediction=prediction.clone()
            prediction[...,:6]+=obs['observation.state'][:,-1:,:6].to(prediction.device)
            actions=prediction.transpose(0,1).cpu()
        elapsed=time.monotonic()-begin
        if not self.native and obs.get('dt_scale') and float(obs['dt_scale'])>1:
            time.sleep(elapsed*(float(obs['dt_scale'])-1))
        return generation,obs['index'],actions,elapsed

    def select_action(self,obs):
        if self.future is not None and self.future.done():
            generation,index,actions,elapsed=self.future.result();self.future=None
            if generation==self.generation:
                skip=max(0,obs['index']-index);actions=actions[skip:];index+=skip
                # Same merge rule as the original DynamicVLA streaming queue.
                # Drop wholly expired chunks instead of indexing an empty one.
                if len(actions):
                    previous=list(self.queue)
                    current=[{'index':index+i,'action':a} for i,a in enumerate(actions)]
                    self.queue.clear()
                    if not previous:self.queue.extend(a for a in current if a['index']>self.action_index)
                    elif index>previous[-1]['index']:self.queue.extend(current)
                    elif index>previous[0]['index']:
                        self.queue.extend(previous[:index-previous[-1]['index']]+current)
                    else:self.queue.extend(current[previous[0]['index']-index:])
                logging.info('Chunk inference %.4fs; skipped %d stale actions',elapsed,skip)
        if self.future is None:
            self.future=self.executor.submit(self.infer,obs,self.generation)
        if not self.queue:return None
        action=self.queue.popleft();self.action_index=action['index'];return action['action']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,required=True);parser.add_argument('--model',required=True)
    parser.add_argument('--port',type=int,default=27000);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--probe',action='store_true');args=parser.parse_args()
    catalog=json.loads(args.catalog.read_text());entry=catalog['models'][args.model]
    import torch
    torch.set_num_threads(2)
    if 'L40S' not in torch.cuda.get_device_name(0):raise ValueError('L40S inference allocation required')
    sys.path.insert(0,str(Path(entry['code_root'])))
    if entry['method']=='dom':
        sys.path.insert(0,catalog['dom_source'])
        from native_dom import Policy
    else:
        sys.path.insert(1,catalog['vlash_source'])
        from policy import Policy
    recipe=json.loads(Path(entry['recipe']).read_text())
    checkpoint_path=Path(entry['checkpoint']);stat=checkpoint_path.stat()
    if stat.st_size!=entry['checkpoint_bytes'] or stat.st_mtime_ns!=entry['checkpoint_mtime_ns']:
        raise ValueError('Pinned checkpoint changed')
    saved=torch.load(checkpoint_path,map_location='cpu',mmap=True,weights_only=False)
    if saved['identity']!=entry['identity'] or saved['progress']!=entry['progress']:
        raise ValueError('Checkpoint identity/progress changed')
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model=Policy(entry['code_root'],recipe,'cuda')
    model.load_state_dict(saved['model'],strict=True);del saved
    model.eval();model.requires_grad_(False)
    adapter=Adapter(model,recipe,torch)
    dummy={k:torch.zeros((1,adapter.config.n_obs_steps,*v.shape))
           for k,v in adapter.config.input_features.items()}
    dummy.update({'task':['Pick up the moving object.'],'observation.environment_state':
                  torch.zeros(1,adapter.config.n_obs_steps,9),'index':0,'dt_scale':1})
    _,_,prediction,_=adapter.infer(dummy,adapter.generation)
    if args.probe:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'policy_probe.json').write_text(json.dumps({'strict_checkpoint_loaded':True,
            'model':args.model,'progress':entry['progress'],'identity':entry['identity'],
            'checkpoint_sha256':entry['checkpoint_sha256'],'finite_physical_shape':list(prediction.shape),
            'gpu':torch.cuda.get_device_name(0),'peak_reserved_bytes':torch.cuda.max_memory_reserved()},indent=2)+'\n')
        adapter.executor.shutdown();return
    source=Path(catalog['official_source']);sys.path.append(str(source))
    spec=importlib.util.spec_from_file_location('official_rpc',source/'scripts/inference.py')
    rpc=importlib.util.module_from_spec(spec);spec.loader.exec_module(rpc)
    transform=rpc._get_transformed_observation
    def with_object_state(observation,rotation,features,device='cpu'):
        import numpy as np
        value=transform(observation,rotation,features,device)
        obj=observation['observation.environment_state']['object']
        environment=np.concatenate([obj['pos'],rpc.utils.helpers.get_rotation_vector(obj['quat'],rotation),
                                    obj['velocity']],axis=-1).astype(np.float32)
        value['observation.environment_state']=torch.from_numpy(environment).to(device)
        return value
    rpc._get_transformed_observation=with_object_state
    sockets=None;start=time.monotonic()
    while sockets is None:
        try:sockets=rpc.get_zmq_sockets('127.0.0.1',args.port,args.port+1)
        except RuntimeError:
            if time.monotonic()-start>900:raise
            time.sleep(5)
    sockets[1].send_pyobj({'vla':args.model,'epoch':entry['progress']['epoch']})
    try:
        n,results=rpc.run_tests(*sockets,{'delta_timestamps':[-2,0] if adapter.native else [0]},
            adapter,str(checkpoint_path),entry['progress']['epoch'],'euler',True,str(args.output))
        logging.info('Evaluation completed. Total tests run: %d',n)
    finally:adapter.executor.shutdown(wait=True)


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='[%(levelname)s] %(asctime)s %(message)s');main()
