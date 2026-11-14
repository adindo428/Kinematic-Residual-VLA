#!/usr/bin/env python3
"""Sequential, synchronized compute-only profile; run only after scored eval."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def distribution(values):
 values=sorted(values)
 def q(p):
  k=(len(values)-1)*p;i=int(k);return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(k-i)
 return {'samples':len(values),'mean_ms':statistics.mean(values),'p50_ms':q(.5),'p95_ms':q(.95),'std_ms':statistics.pstdev(values)}


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--catalog',type=Path,required=True);parser.add_argument('--model',required=True)
 parser.add_argument('--output',type=Path,required=True);parser.add_argument('--warmup',type=int,default=20)
 parser.add_argument('--samples',type=int,default=200);parser.add_argument('--repeats',type=int,default=5)
 args=parser.parse_args();catalog=json.loads(args.catalog.read_text());entry=catalog['models'][args.model]
 import torch
 torch.set_num_threads(2);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 assert 'RTX A6000' in torch.cuda.get_device_name(0)
 stat=Path(entry['weight_path']).stat()
 assert stat.st_size==entry['weight_bytes'] and stat.st_mtime_ns==entry['weight_mtime_ns']
 native=entry['client_kind']=='native'
 if native:
  sys.path.insert(0,entry['source'])
  from utils.helpers import get_policy_cfg
  from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy
  from safetensors.torch import load_model
  cfg=get_policy_cfg(cfg_file=Path(entry['checkpoint'])/'config.json');cfg.enable_streaming=False
  model=DynamicVLAPolicy.from_pretrained(entry['checkpoint'],config=cfg).cuda()
  load_model(model,str(Path(entry['checkpoint'])/'model.safetensors'),strict=True)
  steps=cfg.n_obs_steps
 else:
  sys.path.insert(0,entry['code_root']);sys.path.insert(1,catalog['vlash_source'])
  from policy_inference import Policy
  recipe=json.loads(Path(entry['recipe']).read_text())
  saved=torch.load(entry['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
  assert saved['identity']==entry['identity'] and saved['progress']==entry['progress']
  if entry['client_kind']=='completed':
   from transformers import AutoTokenizer
   from completed_vla_table1_client import pinned_tokenizer
   refs=json.loads((Path(entry['code_root'])/'checkpoints.local.json').read_text())
   context=pinned_tokenizer(AutoTokenizer,refs['google/paligemma-3b-pt-224']['path'])
  else:context=nullcontext()
  with context:model=Policy(entry['code_root'],recipe,'cuda')
  model.load_state_dict(saved['model'],strict=True);del saved
  cfg=model.policy.config;steps=1
 parameters=sum(p.numel() for p in model.parameters())
 requires_grad_as_loaded=sum(p.numel() for p in model.parameters() if p.requires_grad)
 model.eval().requires_grad_(False)
 gpu_uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 def uncontended():
  rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines()
  foreign=[row for row in rows if row.split(',')[-1].strip()==gpu_uuid and int(row.split(',')[0])!=os.getpid()]
  assert not foreign,'Another compute process used the profiling GPU; discard this attempt and retry when idle.'
 prompts=['Pick up the moving drink can.','Place the lemon in the yellow box.','Put the avocado and apple in the container.']
 generator=torch.Generator(device='cpu').manual_seed(42)
 examples=[]
 for i in range(12):
  obs={key:(torch.rand((1,steps,*feature.shape),generator=generator) if key.startswith('observation.images.')
      else torch.zeros(1,steps,*feature.shape)) for key,feature in cfg.input_features.items()}
  env=torch.zeros(1,steps,9);env[...,0]=.4;env[...,6]=[-.1,0.,.1][i%3]
  obs.update(task=[prompts[i%3]],**{'observation.environment_state':env},index=0,dt_scale=1.)
  examples.append(obs)
 def infer(obs):
  model.reset() if native else model.policy.reset()
  with torch.inference_mode(),(nullcontext() if native else torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False)):
   if native:
    batch={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in obs.items()}
    prediction=model.predict_action_chunk(batch)
   else:
    batch={k:v[:,-1] for k,v in obs.items() if k.startswith('observation.images.')}
    batch.update({'observation.state':obs['observation.state'][:,-1,:6],
     'action':torch.zeros(1,20,7),'action_is_pad':torch.zeros(1,20,dtype=torch.bool),
     'velocity':obs['observation.environment_state'][:,-1,6:9],
     'offset':torch.zeros(1,dtype=torch.long),'task':obs['task']})
    prediction=model.predict(batch)
   physical=prediction.clone();physical[...,:6]+=obs['observation.state'][:,-1:,:6].to(physical.device)
   return physical.cpu()
 modes=['enabled']
 kr=bool(entry.get('kr',False)) if native else bool(recipe['kr'])
 if kr:modes.append('analytic_disabled')
 report={'model':args.model,'created_utc':datetime.now(timezone.utc).isoformat(),'strict_checkpoint_loaded':True,
  'gpu':torch.cuda.get_device_name(0),'total_parameters':parameters,
  'requires_grad_parameters_as_loaded':requires_grad_as_loaded,
  'parameter_dtypes':sorted({str(p.dtype) for p in model.parameters()}),
  'autocast':'native checkpoint implementation' if native else 'bfloat16',
  'input_kind':'fixed synthetic compute-only: 12 RGB/state/velocity examples, three prompts, seed42',
  'observation_window':steps,'batch_size':1,'action_horizon':20,'warmup_chunks_per_mode':args.warmup,
  'sampling_configuration':{key:getattr(cfg,key,None) for key in ['num_steps','num_inference_steps','chunk_size','n_action_steps']},
  'timed_chunks_per_run':args.samples,'repeats':args.repeats,'velocity_source':'supplied oracle; no estimator',
  'timing_boundary':'prepared CPU observations to physical action chunk on CPU; CUDA synchronize before and after',
  'analytic_disabled_meaning':'Same checkpoint, same neural conditioning; only analytic residual-flow/reconstruction flag disabled; compute-only intervention.'}
 for mode in modes:
  if mode=='analytic_disabled':
   if native:
    model.config.enable_residual_flow=False;model.model.config.enable_residual_flow=False
   else:model.recipe['kr']=False
  torch.manual_seed(42);torch.cuda.manual_seed_all(42)
  for i in range(args.warmup):
   output=infer(examples[i%len(examples)])
   assert tuple(output.shape)==(1,20,7) and torch.isfinite(output).all()
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
  records=[];runs=[]
  for repeat in range(args.repeats):
   uncontended()
   torch.manual_seed(42+repeat);torch.cuda.manual_seed_all(42+repeat)
   times=[]
   for i in range(args.samples):
    torch.cuda.synchronize();tick=time.perf_counter()
    output=infer(examples[i%len(examples)])
    torch.cuda.synchronize();elapsed=(time.perf_counter()-tick)*1000
    assert tuple(output.shape)==(1,20,7) and torch.isfinite(output).all()
    times.append(elapsed);records.append({'run':repeat,'sample':i,'example':i%len(examples),'wall_ms':elapsed})
   runs.append(distribution(times))
   uncontended()
  report[mode]={**distribution([x['wall_ms'] for x in records]),'runs':runs,'raw_samples':records,
   'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
 args.output.parent.mkdir(parents=True,exist_ok=True)
 temporary=args.output.with_suffix('.json.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(args.output)


if __name__=='__main__':main()
