"""Completed SmolVLA baseline -> official DOM RPC, with physical delta chunks."""
import argparse,json,hashlib,sys,logging,time,importlib.util
from pathlib import Path
from collections import deque
from types import SimpleNamespace

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def batch_from_observation(obs,torch):
 state=obs['observation.state'][:,-1,:6]
 assert state.shape==(1,6) and torch.isfinite(state).all()
 return {**{k:v[:,-1] for k,v in obs.items() if k.startswith('observation.images.')},'observation.state':state,'action':torch.zeros((1,20,7),device=state.device),'action_is_pad':torch.zeros((1,20),dtype=torch.bool,device=state.device),'velocity':torch.zeros((1,3),device=state.device),'offset':torch.zeros(1,dtype=torch.long,device=state.device),'task':obs['task']}

class Adapter:
 def __init__(self,policy,torch):
  self.policy=policy;self.torch=torch
  # Omitting use_delta_action asks the pinned RPC converter to add the
  # chunk-start pose exactly once. Policy.predict already undoes normalization.
  self.config=SimpleNamespace(input_features=policy.policy.config.input_features,output_features=policy.policy.config.output_features,n_action_steps=20,n_obs_steps=1,enable_streaming=False)
  self.reset()
 def reset(self):self.queue=deque();self.policy.policy.reset()
 def select_action(self,obs):
  if not self.queue:
   with self.torch.inference_mode(),self.torch.autocast('cuda',dtype=self.torch.bfloat16,cache_enabled=False):actions=self.policy.predict(batch_from_observation(obs,self.torch))
   assert tuple(actions.shape)==(1,20,7) and self.torch.isfinite(actions).all(),'Invalid physical actions'
   self.queue.extend(actions.unbind(1))
  return self.queue.popleft().detach().cpu()

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--port',type=int,default=27000);p.add_argument('--probe',action='store_true');a=p.parse_args();cfg=json.load(open(a.config));root=Path(cfg['training_root']);out=root/'runs'/'smolvla_baseline'
 for path,want in cfg['files'].items():assert sha(path)==want,'Frozen input changed: '+path
 assert sha(out/'latest.pt')==cfg['checkpoint_sha256'],'Checkpoint changed'
 import torch
 sys.path.insert(0,str(root));from policy import Policy
 recipe=json.load(open(root/'recipes/smolvla_baseline.json'));assert recipe['method']=='smolvla' and not recipe['kr']
 checkpoint=torch.load(out/'latest.pt',map_location='cpu',mmap=True,weights_only=False);completed=json.load(open(out/'completed.json'));latest=json.load(open(out/'latest.json'));provenance=json.load(open(out/'provenance.json'))
 assert checkpoint['progress']==completed['progress']==latest['progress'] and checkpoint['progress']['epoch']==10
 assert checkpoint['identity']==latest['identity']==provenance['identity']
 model=Policy(root,recipe,'cuda');model.load_state_dict(checkpoint['model'],strict=True);model.eval();model.requires_grad_(False);adapter=Adapter(model,torch)
 if a.probe:
  from data import DOM
  from torch.utils.data import DataLoader
  row=next(iter(DataLoader(DOM(root,'validation'),batch_size=1,num_workers=0)))
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):actions=model.predict(row)
  assert tuple(actions.shape)==(1,20,7) and torch.isfinite(actions).all()
  a.output.mkdir(parents=True,exist_ok=True);(a.output/'policy_probe.json').write_text(json.dumps({'strict_completed_checkpoint_loaded':True,'progress':checkpoint['progress'],'identity':checkpoint['identity'],'checkpoint_sha256':cfg['checkpoint_sha256'],'finite_physical_action_shape':list(actions.shape),'gpu':torch.cuda.get_device_name(0),'baseline_only':True},indent=2));return
 source=Path(cfg['official_source']);sys.path.insert(0,str(source));spec=importlib.util.spec_from_file_location('official_rpc',source/'scripts/inference.py');rpc=importlib.util.module_from_spec(spec);spec.loader.exec_module(rpc)
 sockets=None;start=time.monotonic()
 while sockets is None:
  try:sockets=rpc.get_zmq_sockets('127.0.0.1',a.port,a.port+1)
  except RuntimeError:
   if time.monotonic()-start>900:raise
   time.sleep(5)
 sockets[1].send_pyobj({'vla':'smolvla_baseline_20k10_pilot','epoch':10})
 n,results=rpc.run_tests(*sockets,{'delta_timestamps':[0]},adapter,str(out),10,'euler',True,str(a.output))
 logging.info('Evaluation completed. Total tests run: %d',n)
 (a.output/'client-summary.json').write_text(json.dumps({'trials':n,'environments':list(results),'pilot_only':True},indent=2))
if __name__=='__main__':
 logging.basicConfig(level=logging.INFO,format='[%(levelname)s] %(asctime)s %(message)s');main()
